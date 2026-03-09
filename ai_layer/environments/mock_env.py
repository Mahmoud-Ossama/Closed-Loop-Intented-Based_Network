"""
Mock SDN Environment — simulates network states for offline RL training.

Mimics the same Gymnasium interface as SDNEnv (observation_space, action_space)
but requires NO network, NO Ryu controller, NO mock server.

Simulated dynamics:
    - Congestion builds over time if no action is taken
    - Actions shift utilizations across links
    - Packet loss correlates with congestion
    - Random traffic fluctuations each step
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_layer.utils.reward import compute_reward


class MockSDNEnv(gym.Env):
    """Offline simulated SDN environment for training / testing."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        env_cfg = config["environment"]

        self.max_steps = env_cfg["episode"]["max_steps"]
        self.reward_cfg = config.get("reward_function", None)

        # Gym spaces — identical to SDNEnv
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Internal simulation state
        self._utils = np.zeros(3, dtype=np.float32)  # link utilizations
        self._loss = 0.0
        self._current_step = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    #  Gym interface
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._current_step = 0

        # Start with moderate random utilizations
        self._utils = self._rng.uniform(0.1, 0.5, size=3).astype(np.float32)
        self._loss = self._rng.uniform(0.0, 0.01)

        return self._build_state(), {}

    def step(self, action: int):
        self._current_step += 1

        # 1. Apply action effects
        self._apply_action(action)

        # 2. Simulate traffic dynamics
        self._simulate_traffic()

        # 3. Build state and compute reward
        state = self._build_state()
        reward = compute_reward(state, self.reward_cfg)

        terminated = False
        truncated = self._current_step >= self.max_steps

        return state, reward, terminated, truncated, {}

    def render(self):
        s = self._build_state()
        labels = ["link1_util", "link2_util", "link3_util", "pkt_loss", "traffic"]
        parts = " | ".join(f"{l}={v:.3f}" for l, v in zip(labels, s))
        print(f"[Step {self._current_step:3d}] {parts}")

    # ------------------------------------------------------------------ #
    #  Simulation logic
    # ------------------------------------------------------------------ #

    def _apply_action(self, action: int):
        """Simulate the effect of each action on link utilizations."""
        if action == 0:
            # do_nothing — congestion drifts up slightly
            self._utils += self._rng.uniform(0.01, 0.05, size=3)

        elif action == 1:
            # route_to_queue_0 — shift traffic toward link 1
            self._utils[0] += self._rng.uniform(0.05, 0.15)
            self._utils[1] -= self._rng.uniform(0.02, 0.10)
            self._utils[2] -= self._rng.uniform(0.02, 0.10)

        elif action == 2:
            # route_to_queue_1 — shift traffic toward link 2
            self._utils[0] -= self._rng.uniform(0.02, 0.10)
            self._utils[1] += self._rng.uniform(0.05, 0.15)
            self._utils[2] -= self._rng.uniform(0.02, 0.10)

        elif action == 3:
            # apply_rate_limit — reduce overall utilization
            self._utils *= self._rng.uniform(0.7, 0.9)

        elif action == 4:
            # remove_rate_limit — utilization rises
            self._utils *= self._rng.uniform(1.05, 1.2)

    def _simulate_traffic(self):
        """Add random traffic fluctuations and derive packet loss."""
        # Random per-link noise
        noise = self._rng.normal(0.0, 0.03, size=3).astype(np.float32)
        self._utils += noise

        # Clamp utilizations to [0, 1]
        self._utils = np.clip(self._utils, 0.0, 1.0)

        # Packet loss = f(congestion): rises sharply when any link > 0.8
        max_util = self._utils.max()
        if max_util > 0.8:
            self._loss = self._rng.uniform(0.02, 0.08)
        elif max_util > 0.5:
            self._loss = self._rng.uniform(0.005, 0.02)
        else:
            self._loss = self._rng.uniform(0.0, 0.005)

    def _build_state(self) -> np.ndarray:
        """Return the 5-dim state vector matching SDNEnv format."""
        traffic_load = float(self._utils.mean())
        return np.array(
            [self._utils[0], self._utils[1], self._utils[2],
             self._loss, traffic_load],
            dtype=np.float32,
        )
