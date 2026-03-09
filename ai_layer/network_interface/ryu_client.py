import time
import logging
import requests

logger = logging.getLogger(__name__)


class RyuClientError(Exception):
    """Base exception for Ryu REST client errors."""


class RyuConnectionError(RyuClientError):
    """Failed to connect to Ryu controller."""


class RyuResponseError(RyuClientError):
    """Invalid or error response from Ryu controller."""


class RyuClient:
    """REST client for communicating with a Ryu SDN controller."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'environment.ryu_controller' section from prod.json.
        """
        self.base_url = config["base_url"].rstrip("/")
        self.timeout = config.get("timeout_seconds", 5)
        self.retries = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay_seconds", 1)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _request(self, method: str, path: str, json_body: dict = None) -> dict:
        """Send an HTTP request with retry logic.

        Returns the parsed JSON response body, or an empty dict on 204/no-content.
        """
        url = f"{self.base_url}{path}"

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.request(
                    method, url, json=json_body, timeout=self.timeout
                )
                resp.raise_for_status()

                if resp.status_code == 204 or not resp.content:
                    return {}
                return resp.json()

            except requests.ConnectionError as exc:
                logger.warning("Connection failed (attempt %d/%d): %s", attempt, self.retries, exc)
                if attempt == self.retries:
                    raise RyuConnectionError(f"Cannot reach Ryu at {url}") from exc
                time.sleep(self.retry_delay)

            except requests.Timeout as exc:
                logger.warning("Request timed out (attempt %d/%d): %s", attempt, self.retries, exc)
                if attempt == self.retries:
                    raise RyuConnectionError(f"Timeout reaching Ryu at {url}") from exc
                time.sleep(self.retry_delay)

            except requests.HTTPError as exc:
                raise RyuResponseError(
                    f"{method} {url} returned {resp.status_code}: {resp.text}"
                ) from exc

            except ValueError as exc:
                raise RyuResponseError(f"Invalid JSON from {url}") from exc

    # ------------------------------------------------------------------ #
    #  GET endpoints – telemetry
    # ------------------------------------------------------------------ #

    def get_port_stats(self, dpid: str) -> dict:
        """GET /stats/port/{dpid}  →  port statistics."""
        return self._request("GET", f"/stats/port/{dpid}")

    def get_flow_stats(self, dpid: str) -> dict:
        """GET /stats/flow/{dpid}  →  flow statistics."""
        return self._request("GET", f"/stats/flow/{dpid}")

    def get_queue_stats(self, dpid: str) -> dict:
        """GET /qos/queue/{dpid}  →  queue statistics."""
        return self._request("GET", f"/qos/queue/{dpid}")

    # ------------------------------------------------------------------ #
    #  POST endpoints – actions
    # ------------------------------------------------------------------ #

    def install_flow(self, dpid: str, flow_rule: dict) -> dict:
        """POST /stats/flowentry/add  →  install a flow rule.

        Args:
            dpid:      Switch datapath ID.
            flow_rule: Dict with match/actions fields.
                       dpid is injected automatically.
        """
        body = {"dpid": dpid, **flow_rule}
        return self._request("POST", "/stats/flowentry/add", json_body=body)

    def delete_flow(self, dpid: str, flow_rule: dict) -> dict:
        """POST /stats/flowentry/delete  →  remove a flow rule."""
        body = {"dpid": dpid, **flow_rule}
        return self._request("POST", "/stats/flowentry/delete", json_body=body)

    def apply_qos(self, switch_id: str, qos_config: dict) -> dict:
        """POST /qos/queue/{switch_id}  →  configure a QoS queue.

        Args:
            switch_id:  Switch identifier (e.g. '0000000000000001').
            qos_config: Dict with port_name, max_rate, etc.
        """
        return self._request("POST", f"/qos/queue/{switch_id}", json_body=qos_config)

    def post_qos_rule(self, switch_id: str, rule: dict) -> dict:
        """POST /qos/rules/{switch_id}  →  add a QoS routing rule."""
        return self._request("POST", f"/qos/rules/{switch_id}", json_body=rule)

    def reset_network(self) -> dict:
        """POST /network/reset  →  reset network state."""
        return self._request("POST", "/network/reset")
