"""
Reward function for the SDN RL environment.

R = R_util + R_loss + R_balance + R_congestion

Components (from prod.json):
    R_util      = -mean(u1, u2, u3)²          weight 1.0
    R_loss      = -packet_loss_rate × 100      weight 100.0
    R_balance   = 1 / (1 + std(u1, u2, u3))   weight 1.0
    R_congestion = -5 if any(u) > 0.8 else 0   threshold 0.8

Final reward clipped to [-10, 10].
"""

import numpy as np


def compute_reward(state: np.ndarray, config: dict = None) -> float:
    """Compute reward from a 5-dim normalised state vector.

    Args:
        state: [link1_util, link2_util, link3_util, packet_loss, traffic_load]
        config: Optional reward_function config from prod.json.
                Uses spec defaults if not provided.
    """
    u1, u2, u3, loss, _ = state

    # Defaults from spec / prod.json
    loss_weight = 100.0
    congestion_threshold = 0.8
    congestion_penalty = 5.0
    clip_lo, clip_hi = -10.0, 10.0

    if config is not None:
        comps = config.get("components", {})
        loss_weight = comps.get("packet_loss_penalty", {}).get("weight", loss_weight)
        congestion_threshold = comps.get("congestion_threshold", {}).get("threshold", congestion_threshold)
        congestion_penalty = comps.get("congestion_threshold", {}).get("penalty", congestion_penalty)
        clip_lo, clip_hi = config.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])

    utils = np.array([u1, u2, u3])
    mean_util = utils.mean()

    r_util = -(mean_util ** 2)
    r_loss = -loss * loss_weight
    r_balance = 1.0 / (1.0 + utils.std())
    r_congestion = -congestion_penalty if utils.max() > congestion_threshold else 0.0

    reward = r_util + r_loss + r_balance + r_congestion
    return float(np.clip(reward, clip_lo, clip_hi))
