"""
SDN Gymnasium Environment for the updated telemetry model.

Observation: np.array of 12 floats in [0, 1]
Action:      Discrete(5)

step() cycle:
    1. Execute action via ActionTranslator
    2. Wait stabilization_delay seconds
    3. Collect /links/utilization + /latency/{src}/{dst}
    4. Parse telemetry into 12-dim state via TelemetryParser
    5. Compute service-aware reward
"""

import logging
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_layer.network_interface.action_translator import ActionTranslator
from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser
from ai_layer.utils.reward import compute_reward_details

logger = logging.getLogger(__name__)


class SDNEnv(gym.Env):
    """Gymnasium environment wrapping the live SDN network."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        env_cfg = config["environment"]
        net_cfg = env_cfg["network"]
        mon_cfg = env_cfg.get("monitoring", {})
        main_pair = mon_cfg.get("main_pair", {})

        self.stabilization_delay = net_cfg["stabilization_delay_seconds"]
        self.max_steps = env_cfg["episode"]["max_steps"]
        self.active_service = mon_cfg.get("active_service", "URLLC")
        self.latency_src = main_pair.get("src", "G6_D1")
        self.latency_dst = main_pair.get("dst", "URLLC")

        self.client = RyuClient(env_cfg["ryu_controller"])
        self.parser = TelemetryParser(
            link_capacity_mbps=net_cfg.get("link_capacity_mbps", 100.0),
            latency_cap_ms=mon_cfg.get("latency_cap_ms", 200.0),
        )
        self.translator = ActionTranslator(self.client, config)
        self.reward_cfg = config.get("reward_function", None)
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

        self._current_step = 0
        self._state = np.zeros(12, dtype=np.float32)
        self._last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._last_action = None

        if options and "service_type" in options:
            self.active_service = str(options["service_type"])

        try:
            self.client.reset_network()
        except Exception:
            logger.debug("Network reset not available, continuing")

        time.sleep(self.stabilization_delay)
        self._state = self._observe()

        return self._state, {"service_type": self.active_service}

    def step(self, action: int):
        prev_congestion = float(np.max(self._state[:6]))
        result = self.translator.execute(action)
        if not result.success:
            logger.warning("Action %d failed: %s", action, result.message)

        time.sleep(self.stabilization_delay)
        self._state = self._observe()
        reward_details = compute_reward_details(self._state, self.reward_cfg)

        repeat_penalty = 0.0
        if self.repeat_penalty_weight > 0.0 and self._last_action is not None and action == self._last_action:
            repeat_penalty = -self.repeat_penalty_weight

        outcome_bonus = 0.0
        post_congestion = float(np.max(self._state[:6]))
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

        self._current_step += 1
        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            "step": self._current_step,
            "service_type": self.active_service,
            "action_name": result.action_name,
            "action_success": result.success,
            "reward": reward,
            "reward_components": reward_details,
        }
        return self._state, reward, terminated, truncated, info

    def render(self):
        labels = [
            "u_ran_agg", "u_agg_core", "u_core_sp1", "u_core_sp2",
            "u_sp1_lf1", "u_sp2_lf1", "lat", "loss", "traffic",
            "svc_u", "svc_e", "svc_m",
        ]
        parts = [f"{l}={v:.3f}" for l, v in zip(labels, self._state)]
        print(f"[Step {self._current_step:3d}] {' | '.join(parts)}")

    def _observe(self) -> np.ndarray:
        links = self.client.get_link_utilization()
        latency = self.client.get_latency(self.latency_src, self.latency_dst)
        return self.parser.build_state(links, latency, self.active_service)
