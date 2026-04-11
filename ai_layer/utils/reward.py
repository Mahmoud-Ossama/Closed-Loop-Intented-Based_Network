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


def _component(cfg: dict, name: str, default_enabled=True, **defaults) -> dict:
    comp = (cfg or {}).get("components", {}).get(name, {})
    out = dict(defaults)
    out["enabled"] = bool(comp.get("enabled", default_enabled))
    for k, v in defaults.items():
        out[k] = type(v)(comp.get(k, v))
    return out


def compute_reward_details(state: np.ndarray, config: dict = None) -> dict:
    """Return service-aware reward with decomposed components."""
    utils = state[:6]
    latency = float(state[6])
    loss = float(state[7])
    traffic = float(state[8])

    clip_lo, clip_hi = -10.0, 10.0
    if config is not None:
        clip_lo, clip_hi = config.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])

    util_cfg = _component(config, "utilization_penalty", weight=1.0)
    loss_cfg = _component(config, "packet_loss_penalty", weight=10.0)
    lat_cfg = _component(config, "latency_penalty", weight=2.0)
    bal_cfg = _component(config, "balance_bonus", weight=1.0)
    cong_cfg = _component(config, "congestion_threshold", penalty=2.0, threshold=0.8)

    congestion = float(np.max(utils))
    mean_util = float(np.mean(utils))
    balance = float(1.0 / (1.0 + np.std(utils)))
    service = _active_service(state)

    base = {
        "utilization_penalty": (-(mean_util ** 2) * util_cfg["weight"]) if util_cfg["enabled"] else 0.0,
        "packet_loss_penalty": (-(loss * loss_cfg["weight"])) if loss_cfg["enabled"] else 0.0,
        "latency_penalty": (-(latency * lat_cfg["weight"])) if lat_cfg["enabled"] else 0.0,
        "balance_bonus": ((balance * bal_cfg["weight"])) if bal_cfg["enabled"] else 0.0,
        "congestion_penalty": (
            -float(cong_cfg["penalty"]) if (cong_cfg["enabled"] and congestion > float(cong_cfg["threshold"])) else 0.0
        ),
        "traffic_bonus": 0.0,
    }

    if service == "URLLC":
        # Keep URLLC latency/loss sensitive, but still reward balance.
        service_mult = {
            "utilization_penalty": 0.6,
            "packet_loss_penalty": 1.4,
            "latency_penalty": 1.5,
            "balance_bonus": 0.8,
            "congestion_penalty": 1.0,
        }
    elif service == "eMBB":
        # eMBB values throughput but still avoids severe loss/congestion.
        service_mult = {
            "utilization_penalty": 1.0,
            "packet_loss_penalty": 0.8,
            "latency_penalty": 0.4,
            "balance_bonus": 0.6,
            "congestion_penalty": 1.0,
        }
        base["traffic_bonus"] = 2.5 * traffic
    else:  # mMTC
        # mMTC favors stability with moderate sensitivity to loss/latency.
        service_mult = {
            "utilization_penalty": 0.6,
            "packet_loss_penalty": 0.9,
            "latency_penalty": 0.5,
            "balance_bonus": 1.5,
            "congestion_penalty": 0.8,
        }

    scaled = {
        "utilization_penalty": base["utilization_penalty"] * service_mult["utilization_penalty"],
        "packet_loss_penalty": base["packet_loss_penalty"] * service_mult["packet_loss_penalty"],
        "latency_penalty": base["latency_penalty"] * service_mult["latency_penalty"],
        "balance_bonus": base["balance_bonus"] * service_mult["balance_bonus"],
        "congestion_penalty": base["congestion_penalty"] * service_mult["congestion_penalty"],
        "traffic_bonus": base["traffic_bonus"],
    }

    total_unclipped = float(sum(scaled.values()))
    total = float(np.clip(total_unclipped, clip_lo, clip_hi))

    return {
        "service": service,
        **scaled,
        "total_unclipped": total_unclipped,
        "total": total,
    }


def compute_reward(state: np.ndarray, config: dict = None) -> float:
    """Backward-compatible scalar reward helper."""
    return float(compute_reward_details(state, config)["total"])
