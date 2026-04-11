import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
	state: np.ndarray
	action: int
	reward: float
	next_state: np.ndarray
	done: bool


class ReplayBuffer:
	"""Fixed-size FIFO replay buffer for DQN transitions."""

	def __init__(self, capacity: int):
		self.capacity = int(capacity)
		self.buffer = deque(maxlen=self.capacity)

	def add(
		self,
		state: np.ndarray,
		action: int,
		reward: float,
		next_state: np.ndarray,
		done: bool,
	) -> None:
		self.buffer.append(
			Transition(
				state=np.asarray(state, dtype=np.float32),
				action=int(action),
				reward=float(reward),
				next_state=np.asarray(next_state, dtype=np.float32),
				done=bool(done),
			)
		)

	def sample(self, batch_size: int):
		"""Return a random minibatch as numpy arrays."""
		if batch_size > len(self.buffer):
			raise ValueError(
				f"batch_size ({batch_size}) cannot exceed buffer size ({len(self.buffer)})"
			)

		batch = random.sample(self.buffer, batch_size)

		states = np.stack([t.state for t in batch]).astype(np.float32)
		actions = np.array([t.action for t in batch], dtype=np.int64)
		rewards = np.array([t.reward for t in batch], dtype=np.float32)
		next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
		dones = np.array([t.done for t in batch], dtype=np.float32)

		return states, actions, rewards, next_states, dones

	def __len__(self) -> int:
		return len(self.buffer)
