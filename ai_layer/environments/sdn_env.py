
"""
SDN Gymnasium Environment — connects the RL agent to the network
via ryu_client, telemetry_parser, and action_translator.

Observation: np.array of 5 floats in [0, 1]
Action:      Discrete(5)

step() cycle:
    1. Execute action via ActionTranslator
    2. Wait stabilization_delay seconds
    3. Collect telemetry via RyuClient
    4. Parse telemetry into state via TelemetryParser
    5. Compute reward
"""

import time
import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_layer.network_interface.ryu_client import RyuClient
from ai_layer.network_interface.telemetry_parser import TelemetryParser
from ai_layer.network_interface.action_translator import ActionTranslator
from ai_layer.utils.reward import compute_reward

logger = logging.getLogger(__name__)


class SDNEnv(gym.Env):
    """Gymnasium environment wrapping the live SDN network."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        """
        Args:
            config: Full prod.json config dict.
        """
        super().__init__()

        self.config = config
        env_cfg = config["environment"]
        net_cfg = env_cfg["network"]

        # Network params
        self.dpid = net_cfg["switch_dpid"]
        self.stabilization_delay = net_cfg["stabilization_delay_seconds"]
        self.max_steps = env_cfg["episode"]["max_steps"]

        # Components
        self.client = RyuClient(env_cfg["ryu_controller"])
        self.parser = TelemetryParser(
            link_capacity_bps=net_cfg["link_capacity_bps"],
            num_ports=net_cfg["num_ports"],
        )
        self.translator = ActionTranslator(self.client, config)

        # Reward config
        self.reward_cfg = config.get("reward_function", None)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Episode state
        self._current_step = 0
        self._state = np.zeros(5, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Gym interface
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0

        # Optionally reset the network
        try:
            self.client.reset_network()
        except Exception:
            logger.debug("Network reset not available, continuing")

        # Wait for network to settle, then observe
        time.sleep(self.stabilization_delay)
        self._state = self._observe()

        return self._state, {}

    def step(self, action: int):
        # 1. Execute action
        result = self.translator.execute(action)
        if not result.success:
            logger.warning("Action %d failed: %s", action, result.message)

        # 2. Wait for network to stabilize
        time.sleep(self.stabilization_delay)

        # 3-4. Collect telemetry → state
        self._state = self._observe()

        # 5. Compute reward
        reward = compute_reward(self._state, self.reward_cfg)

        # Episode bookkeeping
        self._current_step += 1
        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            "step": self._current_step,
            "action_name": result.action_name,
            "action_success": result.success,
            "reward": reward,
        }

        return self._state, reward, terminated, truncated, info

    def render(self):
        labels = ["link1_util", "link2_util", "link3_util", "pkt_loss", "traffic"]
        parts = [f"{l}={v:.3f}" for l, v in zip(labels, self._state)]
        print(f"[Step {self._current_step:3d}] {' | '.join(parts)}")

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _observe(self) -> np.ndarray:
        """Fetch telemetry and return state vector."""
        raw = self.client.get_port_stats(self.dpid)
        return self.parser.response_to_state(raw, self.dpid)
