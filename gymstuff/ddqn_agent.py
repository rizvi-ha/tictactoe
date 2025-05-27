import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple feed‑forward network used by the Double DQN agent."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed‑size cyclic buffer that stores experience tuples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", ("state", "action", "reward", "next_state", "done")
        )

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            self.experience(state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.array([e.state for e in batch]), dtype=torch.float32)
        actions = torch.as_tensor([e.action for e in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor([e.reward for e in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(
            np.array([e.next_state for e in batch]), dtype=torch.float32
        )
        dones = torch.as_tensor([e.done for e in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    """Double DQN agent.
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        device: str = "cpu",
        hidden_dims = (128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_freq: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
    ) -> None:
        self.device = torch.device(device)
        self.action_dim = action_dim

        # Q‑networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

        # Hyper‑parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ε‑greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0  # number of training‑time action selections

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _flatten_obs(obs: dict) -> np.ndarray:
        """Convert env observation dict → flat float32 vector."""
        return np.concatenate(
            (
                obs["board"].astype(np.float32),
                obs["history_x"].astype(np.float32),
                obs["history_o"].astype(np.float32),
                obs["current_player"].astype(np.float32),
            )
        )

    def _epsilon(self) -> float:
        """Current ε according to exponential decay schedule."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.steps_done / self.epsilon_decay
        )

    # ----------------------------------------------------------- public api
    def act(self, obs: dict, *, greedy: bool = False) -> int:
        state = self._flatten_obs(obs)
        board = obs["board"]
        legal_spots = [i for i, v in enumerate(board) if v == 0]

        eps = 0.0 if greedy else self._epsilon()
        if not greedy:
            self.steps_done += 1
        if random.random() < eps:
            return random.choice(legal_spots)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)[0]
            assert len(q_vals) == self.action_dim, \
                f"Expected {self.action_dim} Q-values, got {len(q_vals)}"

        legal_q_vals = []
        for i, q_val in enumerate(q_vals):
            if i in legal_spots:
                legal_q_vals.append((q_val, i))

        return max(legal_q_vals, key=lambda x: x[0])[1] if legal_q_vals else exit("Somehow no legal moves left!")

    def store(self, *args, **kwargs):
        self.replay.push(*args, **kwargs)

    def update(self):
        # Only train when enough samples collected
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states = rewards.to(self.device), next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        q_sa = self.policy_net(states).gather(1, actions)

        # Double‑DQN target
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions).detach()
        target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard‑update target network
        if self.steps_done % self.target_update_freq == 0 and self.steps_done:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ----------------------------------------------------------- persistence
    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        sd = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)
