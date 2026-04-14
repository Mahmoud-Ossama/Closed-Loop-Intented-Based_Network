"""
Operational reward function for 6D SDN state.

State layout:
    [0] latency_norm
    [1] packet_loss_norm
    [2] throughput_norm
    [3] main_link_util
    [4] backup_link_util
    [5] failover_active
"""

import numpy as np


def _component(cfg: dict, name: str, default_enabled=True, **defaults) -> dict:
    comp = (cfg or {}).get("components", {}).get(name, {})
    out = dict(defaults)
    out["enabled"] = bool(comp.get("enabled", default_enabled))
    for k, v in defaults.items():
        out[k] = type(v)(comp.get(k, v))
    return out


def compute_reward_details(state: np.ndarray, config: dict = None) -> dict:
    """Return reward with decomposed components for operational optimization."""
    latency = float(state[0])
    loss = float(state[1])
    throughput = float(state[2])
    main_util = float(state[3])
    backup_util = float(state[4])
    failover = float(state[5])

    clip_lo, clip_hi = -10.0, 10.0
    if config is not None:
        clip_lo, clip_hi = config.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])

    lat_cfg = _component(config, "latency_penalty", weight=2.0)
    loss_cfg = _component(config, "packet_loss_penalty", weight=3.0)
    util_cfg = _component(config, "utilization_penalty", weight=1.0)
    thr_cfg = _component(config, "throughput_bonus", weight=1.0)
    cong_cfg = _component(config, "congestion_threshold", threshold=0.9, penalty=2.0)
    failover_cfg = _component(config, "failover_penalty", weight=0.2)

    mean_util = float((main_util + backup_util) / 2.0)
    max_util = float(max(main_util, backup_util))

    components = {
        "latency_penalty": (-(latency * lat_cfg["weight"])) if lat_cfg["enabled"] else 0.0,
        "packet_loss_penalty": (-(loss * loss_cfg["weight"])) if loss_cfg["enabled"] else 0.0,
        "utilization_penalty": (-(mean_util ** 2) * util_cfg["weight"]) if util_cfg["enabled"] else 0.0,
        "throughput_bonus": ((throughput * thr_cfg["weight"])) if thr_cfg["enabled"] else 0.0,
        "congestion_penalty": (
            -float(cong_cfg["penalty"]) if (cong_cfg["enabled"] and max_util > float(cong_cfg["threshold"])) else 0.0
        ),
        "failover_penalty": (-(failover * failover_cfg["weight"])) if failover_cfg["enabled"] else 0.0,
    }

    total_unclipped = float(sum(components.values()))
    total = float(np.clip(total_unclipped, clip_lo, clip_hi))

    return {
        **components,
        "total_unclipped": total_unclipped,
        "total": total,
    }


def compute_reward(state: np.ndarray, config: dict = None) -> float:
    """Backward-compatible scalar reward helper."""
    return float(compute_reward_details(state, config)["total"])
