"""
Mock SDN Environment for offline training with the updated 12-feature state.

Simulates:
    - 6 link utilizations across main/backup paths
    - latency and packet loss dynamics
    - traffic load fluctuations
    - service-aware one-hot encoding (URLLC/eMBB/mMTC)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_layer.utils.reward import compute_reward_details


class MockSDNEnv(gym.Env):
    """Offline simulated SDN environment for training/testing."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        env_cfg = config["environment"]
        mon_cfg = env_cfg.get("monitoring", {})

        self.max_steps = env_cfg["episode"]["max_steps"]
        self.reward_cfg = config.get("reward_function", None)
        self.active_service = mon_cfg.get("active_service", "URLLC")
        repeat_cfg = self.reward_cfg.get("components", {}).get("action_repeat_penalty", {}) if self.reward_cfg else {}
        self.repeat_penalty_weight = (
            float(repeat_cfg.get("weight", 0.0)) if bool(repeat_cfg.get("enabled", False)) else 0.0
        )
        outcome_cfg = self.reward_cfg.get("components", {}).get("outcome_improvement_bonus", {}) if self.reward_cfg else {}
        self.outcome_bonus_weight = (
            float(outcome_cfg.get("weight", 0.0)) if bool(outcome_cfg.get("enabled", False)) else 0.0
        )
        self.outcome_bonus_max = float(outcome_cfg.get("max_bonus", 0.2)) if outcome_cfg else 0.2

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._utils = np.zeros(6, dtype=np.float32)
        self._lat = 0.0
        self._loss = 0.0
        self._current_step = 0
        self._rng = np.random.default_rng()
        self._last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._current_step = 0
        self._last_action = None

        if options and "service_type" in options:
            self.active_service = str(options["service_type"])

        self._utils = self._rng.uniform(0.10, 0.55, size=6).astype(np.float32)
        self._lat = float(self._rng.uniform(0.02, 0.20))
        self._loss = float(self._rng.uniform(0.0, 0.03))

        return self._build_state(), {"service_type": self.active_service}

    def step(self, action: int):
        self._current_step += 1
        prev_congestion = float(np.max(self._utils))

        self._apply_action(action)
        self._simulate_traffic()

        state = self._build_state()
        reward_details = compute_reward_details(state, self.reward_cfg)

        repeat_penalty = 0.0
        if self.repeat_penalty_weight > 0.0 and self._last_action is not None and action == self._last_action:
            repeat_penalty = -self.repeat_penalty_weight

        outcome_bonus = 0.0
        post_congestion = float(np.max(state[:6]))
        if self.outcome_bonus_weight > 0.0 and action in (1, 2):
            improvement = max(0.0, prev_congestion - post_congestion)
            outcome_bonus = min(self.outcome_bonus_max, improvement * self.outcome_bonus_weight)

        reward = reward_details["total"] + repeat_penalty + outcome_bonus

        clip_lo, clip_hi = -10.0, 10.0
        if self.reward_cfg is not None:
            clip_lo, clip_hi = self.reward_cfg.get("normalization", {}).get("clip_range", [clip_lo, clip_hi])
        reward = float(np.clip(reward, clip_lo, clip_hi))
        reward_details["action_repeat_penalty"] = repeat_penalty
        reward_details["outcome_improvement_bonus"] = outcome_bonus
        reward_details["total"] = reward
        self._last_action = action

        terminated = False
        truncated = self._current_step >= self.max_steps
        return state, reward, terminated, truncated, {
            "service_type": self.active_service,
            "reward_components": reward_details,
        }

    def render(self):
        s = self._build_state()
        labels = [
            "u_ran_agg", "u_agg_core", "u_core_sp1", "u_core_sp2",
            "u_sp1_lf1", "u_sp2_lf1", "lat", "loss", "traffic",
            "svc_u", "svc_e", "svc_m",
        ]
        parts = " | ".join(f"{l}={v:.3f}" for l, v in zip(labels, s))
        print(f"[Step {self._current_step:3d}] {parts}")

    def _apply_action(self, action: int):
        if action == 0:
            self._utils += self._rng.uniform(0.00, 0.03, size=6)

        elif action == 1:
            # Favor main path (sp1 branch)
            self._utils[2] += self._rng.uniform(0.05, 0.12)
            self._utils[4] += self._rng.uniform(0.05, 0.12)
            self._utils[3] -= self._rng.uniform(0.03, 0.10)
            self._utils[5] -= self._rng.uniform(0.03, 0.10)

        elif action == 2:
            # Favor backup path (sp2 branch)
            self._utils[3] += self._rng.uniform(0.05, 0.12)
            self._utils[5] += self._rng.uniform(0.05, 0.12)
            self._utils[2] -= self._rng.uniform(0.03, 0.10)
            self._utils[4] -= self._rng.uniform(0.03, 0.10)

        elif action == 3:
            # Apply stricter shaping at Core
            self._utils *= self._rng.uniform(0.72, 0.90)

        elif action == 4:
            # Relax shaping
            self._utils *= self._rng.uniform(1.05, 1.18)

    def _simulate_traffic(self):
        noise = self._rng.normal(0.0, 0.025, size=6).astype(np.float32)
        self._utils += noise
        self._utils = np.clip(self._utils, 0.0, 1.0)

        max_util = float(np.max(self._utils))
        self._lat = float(np.clip(0.04 + 0.6 * max_util + self._rng.normal(0.0, 0.03), 0.0, 1.0))

        if max_util > 0.8:
            self._loss = float(self._rng.uniform(0.03, 0.15))
        elif max_util > 0.55:
            self._loss = float(self._rng.uniform(0.01, 0.05))
        else:
            self._loss = float(self._rng.uniform(0.0, 0.015))

    def _build_state(self) -> np.ndarray:
        state = np.zeros(12, dtype=np.float32)
        state[:6] = self._utils
        state[6] = self._lat
        state[7] = self._loss
        state[8] = float(np.mean(self._utils))

        svc = self.active_service.strip().lower()
        if svc == "urllc":
            state[9:12] = [1.0, 0.0, 0.0]
        elif svc == "embb":
            state[9:12] = [0.0, 1.0, 0.0]
        elif svc == "mmtc":
            state[9:12] = [0.0, 0.0, 1.0]

        return np.clip(state, 0.0, 1.0)
