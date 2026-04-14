"""
Action Translator for runtime optimization actions.

Action mapping (from prod.json):
    0 -> do_nothing   : no API call
    1 -> update_queue : POST /qos/queue/{switch_id}
    2 -> failover     : POST /router/{switch_id}
    3 -> reroute      : POST /router/{switch_id}
"""

import logging
from dataclasses import dataclass, field

from ai_layer.network_interface.ryu_client import RyuClient

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    success: bool
    action_id: int
    action_name: str
    message: str
    metadata: dict = field(default_factory=dict)


class ActionTranslator:
    """Converts runtime action IDs into concrete Ryu API operations."""

    def __init__(self, client: RyuClient, config: dict):
        self.client = client
        self.config = config
        self.default_switch = config["environment"]["network"]["switch_dpid"]
        self._actions_cfg = config["environment"]["action_space"]["actions"]

        self._handlers = {
            0: self._do_nothing,
            1: self._update_queue,
            2: self._failover,
            3: self._reroute,
        }

    def execute(self, action_id: int) -> ActionResult:
        handler = self._handlers.get(action_id)
        if handler is None:
            return ActionResult(
                success=False,
                action_id=action_id,
                action_name="unknown",
                message=f"Invalid action_id: {action_id}",
            )

        action_key = str(action_id)
        action_name = self._actions_cfg.get(action_key, {}).get("name", "unknown")

        try:
            metadata = handler(action_key) or {}
            logger.info("Action %d (%s) executed", action_id, action_name)
            return ActionResult(
                success=True,
                action_id=action_id,
                action_name=action_name,
                message=f"Executed {action_name}",
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Action %d (%s) failed: %s", action_id, action_name, exc)
            return ActionResult(
                success=False,
                action_id=action_id,
                action_name=action_name,
                message=str(exc),
            )

    def _do_nothing(self, _action_key: str) -> dict:
        return {}

    def _update_queue(self, action_key: str) -> dict:
        target = self._actions_cfg[action_key].get("target", {})
        switch_id = str(target.get("switch_dpid", self.default_switch))
        qos_config = target.get("qos_config", {})
        if not qos_config:
            raise ValueError("Missing qos_config for update_queue action")
        self.client.apply_qos(switch_id, qos_config)
        return {"operation": "update_queue", "switch_id": switch_id}

    def _failover(self, action_key: str) -> dict:
        target = self._actions_cfg[action_key].get("target", {})
        switch_id = str(target.get("switch_dpid", self.default_switch))
        route = target.get("route", {})
        if not route:
            raise ValueError("Missing route payload for failover action")
        self.client.post_router_entry(switch_id, route)
        return {
            "operation": "failover",
            "switch_id": switch_id,
            "failover_active": bool(target.get("set_failover_active", True)),
        }

    def _reroute(self, action_key: str) -> dict:
        target = self._actions_cfg[action_key].get("target", {})
        switch_id = str(target.get("switch_dpid", self.default_switch))
        route = target.get("route", {})
        if not route:
            raise ValueError("Missing route payload for reroute action")
        self.client.post_router_entry(switch_id, route)
        return {
            "operation": "reroute",
            "switch_id": switch_id,
            "failover_active": bool(target.get("set_failover_active", False)),
        }
