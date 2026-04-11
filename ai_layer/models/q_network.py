import torch
import torch.nn as nn


class QNetwork(nn.Module):
	"""Simple MLP Q-network: state_dim -> 64 -> 64 -> action_dim."""

	def __init__(self, state_dim: int, action_dim: int):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, action_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Return Q-values for all actions."""
		return self.model(x)
