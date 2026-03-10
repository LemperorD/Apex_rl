# Copyright (c) 2026 GitHub@Apex_rl Developer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay buffer for Off-policy RL algorithms.

Supports multi-dimensional observations (e.g., images) and flexible storage.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class ReplayBuffer:
    """Buffer for storing replay data during Off-policy RL training.

    Stores transitions (observations, actions, rewards, dones, values, log_probs)
    and computes advantages using Generalized Advantage Estimation (GAE).

    Supports multi-dimensional observations (images, vectors, etc.).

    Attributes:
        num_envs (int): Number of parallel environments.
        capacity (int): Maximum number of transitions to store in the buffer.
        obs_shape (tuple): Shape of observations (can be multi-dimensional).
        action_dim (int): Dimension of the action space.
        device (torch.device): Device for tensors.
        is_priority (bool): Whether to use prioritized experience replay.
        full (bool): Whether the buffer is full.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        is_priority: bool = False,
    ):
        """Initialize the replay buffer.

        Args:
            num_envs: Number of parallel environments.
            capacity: Maximum number of transitions to store in the buffer.
            obs_shape: Shape of observations (e.g., (48,) for vectors, (3, 84, 84) for images).
            action_dim: Dimension of the action space.
            device: Device for tensors.
            is_priority: Whether to use prioritized experience replay.
        """
        self.num_envs = num_envs
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.is_priority = is_priority

        # Buffers for replay data
        # Shape: (capacity, *obs_shape)
        self.observations = torch.zeros(
            (capacity, *obs_shape), device=device, dtype=torch.float32,
        )

        self.next_observations = torch.zeros(
            (capacity, *obs_shape), dtype=torch.float32, device=device,
        )

        self.actions = torch.zeros(
            (capacity, action_dim), dtype=torch.float32, device=device,
        )

        self.rewards = torch.zeros(
            capacity, device=device, dtype=torch.float32, 
        )

        self.dones = torch.zeros(
            capacity, device=device, dtype=torch.float32, 
        )

        # Buffer pointer & size
        self.pos = 0
        self.size = 0
        self.full = False

        if is_priority:
            self.priorities = torch.zeros(
                capacity, device=device, dtype=torch.float32,
            )
            self.alpha = 0.6
            self.beta = 0.4
            self.beta_increment = 1e-4
            self.eps = 1e-6

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a transition to the replay buffer.

        Args:
            observations: Observations. Shape: (num_envs, *obs_shape)
            actions: Actions. Shape: (num_envs, action_dim) or (num_envs,)
            rewards: Rewards. Shape: (num_envs,)
            next_observations: Next observations. Shape: (num_envs, *obs_shape)
            dones: Done flags. Shape: (num_envs,)
        """
        n = self.num_envs
        start = self.pos
        end = self.pos + n

        if end <= self.capacity: 
            self.observations[start:end] = observations
            self.actions[start:end] = actions
            self.rewards[start:end] = rewards
            self.next_observations[start:end] = next_observations
            self.dones[start:end] = dones
            if self.is_priority:
                self.priorities[start:end] = self.priorities.max() if self.size > 0 else 1.0
        else:
            first_part = self.capacity - start
            second_part = n - first_part
            self.observations[start:] = observations[:first_part]
            self.observations[:second_part] = observations[first_part:]
            self.actions[start:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[start:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_observations[start:] = next_observations[:first_part]
            self.next_observations[:second_part] = next_observations[first_part:]
            self.dones[start:] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]
            if self.is_priority:
                max_p = self.priorities.max() if self.size > 0 else 1.0
                self.priorities[start:] = max_p
                self.priorities[:second_part] = max_p

        # update buffer pointers
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of samples.

        Returns:
            Tuple containing:
                observations
                actions
                rewards
                next_observations
                dones
                indices
                importance sampling weights
        """

        if self.is_priority:
            priorities = self.priorities[:self.size] ** self.alpha
            probs = priorities / priorities.sum()

            indices = torch.multinomial(probs, batch_size, replacement=True)

            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()

            # anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)

        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)
            weights = torch.ones(batch_size, device=self.device)

        obs = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_observations[indices]
        dones = self.dones[indices]

        return obs, actions, rewards, next_obs, dones, indices, weights
    
    def update_priorities(
        self,
        indices: torch.Tensor,
        td_errors: torch.Tensor,
    ) -> None:
        """Update priorities for prioritized replay.

        Args:
            indices: Indices sampled from the buffer.
            td_errors: TD errors corresponding to the samples.
        """

        if not self.is_priority:
            return

        priorities = torch.abs(td_errors) + self.eps
        self.priorities[indices] = priorities

    def clear(self) -> None:
        """Clear the buffer."""

        self.pos = 0
        self.size = 0
        self.full = False

        if self.is_priority:
            self.priorities.zero_()

    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return self.size
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return buffer state."""

        state = {
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "pos": self.pos,
            "size": self.size,
        }

        if self.is_priority:
            state["priorities"] = self.priorities

        return state
    
    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load buffer state."""

        self.observations.copy_(state["observations"])
        self.next_observations.copy_(state["next_observations"])
        self.actions.copy_(state["actions"])
        self.rewards.copy_(state["rewards"])
        self.dones.copy_(state["dones"])

        self.pos = state["pos"]
        self.size = state["size"]

        if self.is_priority and "priorities" in state:
            self.priorities.copy_(state["priorities"])

    def to(self, device: torch.device) -> "ReplayBuffer":
        """Move all tensors to a new device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        self.observations = self.observations.to(device)
        if self.privileged_observations is not None:
            self.privileged_observations = self.privileged_observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.log_probs = self.log_probs.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        return self
