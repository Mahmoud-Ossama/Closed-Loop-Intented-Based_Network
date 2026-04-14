"""
Telemetry Parser for operational 6D state.

State layout:
    [0] latency_norm
    [1] packet_loss_norm
    [2] throughput_norm
    [3] main_link_util
    [4] backup_link_util
    [5] failover_active

Raw telemetry inputs:
    - GET /links/utilization
    - GET /latency/{src}/{dst}
"""

from typing import Dict

import numpy as np


class TelemetryParser:
    """Converts Ryu telemetry JSON into normalized operational state."""

    MAIN_LINKS = ["core -> sp1", "sp1 -> lf1"]
    BACKUP_LINKS = ["core -> sp2", "sp2 -> lf1"]

    def __init__(
        self,
        main_link_capacity_mbps: float,
        backup_link_capacity_mbps: float,
        latency_min_ms: float,
        latency_max_ms: float,
        packet_loss_max_percent: float,
    ):
        self.main_capacity = max(float(main_link_capacity_mbps), 1e-6)
        self.backup_capacity = max(float(backup_link_capacity_mbps), 1e-6)
        self.latency_min_ms = float(latency_min_ms)
        self.latency_max_ms = max(float(latency_max_ms), self.latency_min_ms + 1e-6)
        self.packet_loss_max_percent = max(float(packet_loss_max_percent), 1e-6)

    def parse_link_utilization(self, response: dict) -> Dict[str, float]:
        """Parse /links/utilization into {link_name: tx_mbps}."""
        links = response.get("links", [])
        result: Dict[str, float] = {}
        for item in links:
            link = item.get("link")
            tx_mbps = float(item.get("tx_mbps", 0.0))
            if link:
                result[link] = tx_mbps
        return result

    def parse_latency(self, response: dict) -> Dict[str, float]:
        """Parse /latency/{src}/{dst} response into numeric values."""
        latency_ms = float(response.get("latency_ms", 0.0))
        loss_percent = float(response.get("packet_loss_percent", 0.0))
        return {
            "latency_ms": latency_ms,
            "packet_loss_percent": loss_percent,
        }

    def build_state(
        self,
        link_util_response: dict,
        latency_response: dict,
        failover_active: bool,
    ) -> np.ndarray:
        """Build a 6D operational state vector in [0, 1]."""
        link_map = self.parse_link_utilization(link_util_response)
        perf = self.parse_latency(latency_response)

        main_tx = float(sum(link_map.get(name, 0.0) for name in self.MAIN_LINKS) / len(self.MAIN_LINKS))
        backup_tx = float(sum(link_map.get(name, 0.0) for name in self.BACKUP_LINKS) / len(self.BACKUP_LINKS))

        main_util = main_tx / self.main_capacity
        backup_util = backup_tx / self.backup_capacity

        latency_norm = (perf["latency_ms"] - self.latency_min_ms) / (self.latency_max_ms - self.latency_min_ms)
        loss_norm = perf["packet_loss_percent"] / self.packet_loss_max_percent

        throughput_mbps = max(0.0, link_map.get("core -> sp1", 0.0)) + max(0.0, link_map.get("core -> sp2", 0.0))
        throughput_cap = self.main_capacity + self.backup_capacity
        throughput_norm = throughput_mbps / max(throughput_cap, 1e-6)

        state = np.array(
            [
                latency_norm,
                loss_norm,
                throughput_norm,
                main_util,
                backup_util,
                1.0 if bool(failover_active) else 0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(state, 0.0, 1.0)
