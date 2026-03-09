"""
Action Translator — maps RL action IDs (0-4) to Ryu REST API commands.

Action mapping (from prod.json):
    0 → do_nothing        — no API call
    1 → route_to_queue_0  — POST /qos/rules  (high-priority queue)
    2 → route_to_queue_1  — POST /qos/rules  (low-priority queue)
    3 → apply_rate_limit  — POST /qos/queue   (500 kbps cap)
    4 → remove_rate_limit — POST /qos/queue   (restore 1 Mbps)

Uses RyuClient to execute commands. Returns ActionResult for every call.
"""

import logging
from dataclasses import dataclass

from ai_layer.network_interface.ryu_client import RyuClient

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    success: bool
    action_id: int
    action_name: str
    message: str


class ActionTranslator:
    """Converts integer action IDs into Ryu REST API calls."""

    def __init__(self, client: RyuClient, config: dict):
        """
        Args:
            client: RyuClient instance.
            config: Full prod.json config dict.
        """
        self.client = client
        self.dpid = config["environment"]["network"]["switch_dpid"]

        # Read action definitions from config
        actions_cfg = config["environment"]["action_space"]["actions"]
        self._actions_cfg = actions_cfg

        self._handlers = {
            0: self._do_nothing,
            1: self._route_to_queue_0,
            2: self._route_to_queue_1,
            3: self._apply_rate_limit,
            4: self._remove_rate_limit,
        }

    def execute(self, action_id: int) -> ActionResult:
        """Execute the action corresponding to action_id.

        Args:
            action_id: Integer 0-4.

        Returns:
            ActionResult with success status and description.
        """
        handler = self._handlers.get(action_id)
        if handler is None:
            return ActionResult(
                success=False,
                action_id=action_id,
                action_name="unknown",
                message=f"Invalid action_id: {action_id}",
            )

        name = self._actions_cfg[str(action_id)]["name"]
        try:
            handler()
            logger.info("Action %d (%s) executed", action_id, name)
            return ActionResult(
                success=True, action_id=action_id, action_name=name,
                message=f"Executed {name}",
            )
        except Exception as exc:
            logger.error("Action %d (%s) failed: %s", action_id, name, exc)
            return ActionResult(
                success=False, action_id=action_id, action_name=name,
                message=str(exc),
            )

    # ------------------------------------------------------------------ #
    #  Action handlers
    # ------------------------------------------------------------------ #

    def _do_nothing(self):
        pass

    def _route_to_queue_0(self):
        """Send UDP traffic to high-priority queue (queue_id=0)."""
        target = self._actions_cfg["1"]["target"]
        self.client.post_qos_rule(self.dpid, {
            "match": {
                "nw_proto": 17,               # UDP
                "tp_dst": target["port"],      # 5002
                "nw_dst": target["destination_ip"],
            },
            "actions": {"queue": target["queue_id"]},  # queue 0
        })

    def _route_to_queue_1(self):
        """Send UDP traffic to low-priority queue (queue_id=1)."""
        target = self._actions_cfg["2"]["target"]
        self.client.post_qos_rule(self.dpid, {
            "match": {
                "nw_proto": 17,
                "tp_dst": target["port"],
                "nw_dst": target["destination_ip"],
            },
            "actions": {"queue": target["queue_id"]},  # queue 1
        })

    def _apply_rate_limit(self):
        """Apply 500 kbps rate limit on s1-eth1."""
        target = self._actions_cfg["3"]["target"]
        self.client.apply_qos(self.dpid, {
            "port_name": target["port_name"],
            "type": "linux-htb",
            "max_rate": str(target["max_rate_bps"]),
            "queues": [{"max_rate": str(target["max_rate_bps"])}],
        })

    def _remove_rate_limit(self):
        """Restore full 1 Mbps on s1-eth1."""
        target = self._actions_cfg["4"]["target"]
        self.client.apply_qos(self.dpid, {
            "port_name": target["port_name"],
            "type": "linux-htb",
            "max_rate": str(target["max_rate_bps"]),
            "queues": [{"max_rate": str(target["max_rate_bps"])}],
        })
