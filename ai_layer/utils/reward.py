"""
Service-aware reward function for the updated SDN state vector.

State layout:
    [0:6]  link utilizations
    [6]    latency_norm
    [7]    packet_loss
    [8]    traffic_load
    [9:12] service one-hot [URLLC, eMBB, mMTC]
"""

import numpy as np


def _active_service(state: np.ndarray) -> str:
    svc = state[9:12]
    idx = int(np.argmax(svc)) if np.any(svc > 0) else 0
    return ["URLLC", "eMBB", "mMTC"][idx]


def compute_reward(state: np.ndarray, config: dict = None) -> float:
    """Compute service-aware reward from a 12-dim normalized state."""
    utils = state[:6]
    latency = float(state[6])
    loss = float(state[7])
    traffic = float(state[8])

    clip_lo, clip_hi = -10.0, 10.0
    if config is not None:
        clip_lo, clip_hi = config.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])

    congestion = float(np.max(utils))
    balance = float(1.0 / (1.0 + np.std(utils)))
    service = _active_service(state)

    if service == "URLLC":
        # Prioritize low latency and low loss.
        reward = (-2.0 * latency) + (-5.0 * loss) + (-1.0 * congestion)
    elif service == "eMBB":
        # Prioritize high throughput while avoiding severe congestion.
        reward = (+3.0 * traffic) + (-1.0 * congestion) + (-1.0 * loss)
    else:  # mMTC
        # Prioritize stability and acceptable loss.
        reward = (+2.0 * balance) + (-1.0 * loss) + (-0.5 * congestion)

    return float(np.clip(reward, clip_lo, clip_hi))
