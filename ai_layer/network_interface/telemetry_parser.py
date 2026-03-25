"""
Telemetry Parser for the updated network model.

Builds a normalized 12-dimensional state vector:
    [0] util_ran_agg
    [1] util_agg_core
    [2] util_core_sp1
    [3] util_core_sp2
    [4] util_sp1_lf1
    [5] util_sp2_lf1
    [6] latency_norm
    [7] packet_loss_norm
    [8] traffic_load
    [9] service_urllc
    [10] service_embb
    [11] service_mmtc

Raw telemetry inputs:
    - GET /links/utilization
    - GET /latency/{src}/{dst}
"""

from typing import Dict, List

import numpy as np


class TelemetryParser:
    """Converts updated Ryu JSON telemetry into RL state vectors."""

    LINK_ORDER: List[str] = [
        "RAN -> agg",
        "agg -> core",
        "core -> sp1",
        "core -> sp2",
        "sp1 -> lf1",
        "sp2 -> lf1",
    ]

    def __init__(self, link_capacity_mbps: float = 100.0, latency_cap_ms: float = 200.0):
        """
        Args:
            link_capacity_mbps: Fixed link capacity for utilization normalization.
            latency_cap_ms: Max latency used for normalization/clipping.
        """
        self.link_capacity_mbps = max(float(link_capacity_mbps), 1e-6)
        self.latency_cap_ms = max(float(latency_cap_ms), 1e-6)

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
        # API returns percent; convert to [0,1] fraction.
        loss_frac = float(response.get("packet_loss_percent", 0.0)) / 100.0
        return {
            "latency_ms": latency_ms,
            "packet_loss": loss_frac,
        }

    def build_state(
        self,
        link_util_response: dict,
        latency_response: dict,
        service_type: str,
    ) -> np.ndarray:
        """Build the 12-dim normalized state vector.

        Args:
            link_util_response: Response from /links/utilization.
            latency_response: Response from /latency/{src}/{dst}.
            service_type: One of {"URLLC", "eMBB", "mMTC"}.
        """
        state = np.zeros(12, dtype=np.float32)

        link_map = self.parse_link_utilization(link_util_response)
        for idx, name in enumerate(self.LINK_ORDER):
            tx_mbps = link_map.get(name, 0.0)
            state[idx] = tx_mbps / self.link_capacity_mbps

        perf = self.parse_latency(latency_response)
        state[6] = perf["latency_ms"] / self.latency_cap_ms
        state[7] = perf["packet_loss"]

        state[8] = float(np.mean(state[:6]))

        service_norm = service_type.strip().lower()
        if service_norm == "urllc":
            state[9:12] = [1.0, 0.0, 0.0]
        elif service_norm == "embb":
            state[9:12] = [0.0, 1.0, 0.0]
        elif service_norm == "mmtc":
            state[9:12] = [0.0, 0.0, 1.0]

        return np.clip(state, 0.0, 1.0)
