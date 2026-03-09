"""
Telemetry Parser — converts raw Ryu REST responses into a normalized
5-dimensional RL state vector.

State vector layout (all values in [0, 1]):
    [0] link1_utilization   — port 1 tx_bytes / link_capacity
    [1] link2_utilization   — port 2 tx_bytes / link_capacity
    [2] link3_utilization   — port 3 tx_bytes / link_capacity
    [3] packet_loss_rate    — total dropped / total packets
    [4] total_traffic_load  — sum of link utilizations (capped at 1.0)

Designed to be pluggable: if the Ryu response format changes,
only the _extract_* helpers need updating.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ------------------------------------------------------------------ #
#  Data classes – structured intermediates
# ------------------------------------------------------------------ #

@dataclass
class PortStatistics:
    port_no: int
    rx_packets: int
    tx_packets: int
    rx_bytes: int
    tx_bytes: int
    rx_dropped: int
    tx_dropped: int


@dataclass
class FlowEntry:
    priority: int
    match: dict
    actions: list
    packet_count: int
    byte_count: int


# ------------------------------------------------------------------ #
#  Parser
# ------------------------------------------------------------------ #

class TelemetryParser:
    """Converts raw Ryu JSON telemetry into RL state vectors."""

    def __init__(self, link_capacity_bps: int = 1_000_000, num_ports: int = 3):
        """
        Args:
            link_capacity_bps: Max link bandwidth in bytes/sec (default 1 Mbps).
            num_ports:         Number of switch ports to track.
        """
        # Ryu reports bytes, and link_capacity_bps is in bits.
        # Convert capacity to bytes/sec for correct utilization ratio.
        self.link_capacity_bytes = link_capacity_bps / 8
        self.num_ports = num_ports

    # ----- structured parsing ---------------------------------------- #

    def parse_port_stats(self, response: dict, dpid: str) -> Dict[int, PortStatistics]:
        """Parse raw port stats response into {port_no: PortStatistics}.

        Expected input format (from Ryu ofctl_rest):
            { "<dpid>": [ {port_no, rx_packets, ...}, ... ] }
        """
        port_list = response.get(dpid, [])
        result = {}
        for p in port_list:
            ps = PortStatistics(
                port_no=p["port_no"],
                rx_packets=p.get("rx_packets", 0),
                tx_packets=p.get("tx_packets", 0),
                rx_bytes=p.get("rx_bytes", 0),
                tx_bytes=p.get("tx_bytes", 0),
                rx_dropped=p.get("rx_dropped", 0),
                tx_dropped=p.get("tx_dropped", 0),
            )
            result[ps.port_no] = ps
        return result

    def parse_flow_stats(self, response: dict, dpid: str) -> List[FlowEntry]:
        """Parse raw flow stats response into a list of FlowEntry."""
        flow_list = response.get(dpid, [])
        return [
            FlowEntry(
                priority=f.get("priority", 0),
                match=f.get("match", {}),
                actions=f.get("actions", []),
                packet_count=f.get("packet_count", 0),
                byte_count=f.get("byte_count", 0),
            )
            for f in flow_list
        ]

    # ----- state vector ---------------------------------------------- #

    def build_state(self, port_stats: Dict[int, PortStatistics]) -> np.ndarray:
        """Build the 5-dim normalized state vector from parsed port stats.

        Returns:
            np.ndarray of shape (5,), dtype float32, values clipped to [0, 1].
        """
        state = np.zeros(5, dtype=np.float32)

        # Link utilizations (ports 1-3)
        for i in range(self.num_ports):
            port_no = i + 1
            ps = port_stats.get(port_no)
            if ps is not None:
                state[i] = ps.tx_bytes / self.link_capacity_bytes

        # Packet loss rate
        total_packets = sum(
            ps.rx_packets + ps.tx_packets for ps in port_stats.values()
        )
        total_dropped = sum(
            ps.rx_dropped + ps.tx_dropped for ps in port_stats.values()
        )
        state[3] = total_dropped / max(total_packets, 1)

        # Total traffic load (sum of 3 link utilizations, normalised to 1)
        raw_load = state[0] + state[1] + state[2]
        state[4] = raw_load / self.num_ports  # average keeps it in [0,1] range

        return np.clip(state, 0.0, 1.0)

    # ----- convenience: raw response → state in one call ------------- #

    def response_to_state(self, port_response: dict, dpid: str) -> np.ndarray:
        """Shortcut: raw Ryu port-stats JSON → normalised state vector."""
        port_stats = self.parse_port_stats(port_response, dpid)
        return self.build_state(port_stats)
