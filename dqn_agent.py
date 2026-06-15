"""
dqn_agent.py
============
Deep Q-Network (DQN) reinforcement learning agent for the modular hybrid
energy system environment defined in formulas.py.

Algorithm components implemented:
  - Experience replay buffer (Lin 1992) — breaks temporal correlations in training data
  - Target network (Mnih et al. 2015) — stabilises Q-value regression targets
  - Epsilon-greedy exploration — balances exploration and exploitation
  - Bellman equation for TD target computation — bootstrapped Q-value updates
"""

import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Two-hidden-layer fully connected Q-network.

    Maps a state vector of length state_size to Q-values for each of
    action_size discrete actions.
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer  (Lin 1992)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (s, a, r, s', done) tuples.

    Randomly sampling a batch from this buffer breaks the temporal
    auto-correlation of consecutive environment transitions, which is the
    key insight from Lin (1992) that stabilises neural-network Q-learning.
    """

    def __init__(self, capacity: int = 50_000):
        self._buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience tuple."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Draw a random batch and return five separate numpy arrays:
        states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with experience replay and a frozen target network.

    The policy network (self.policy_net) is updated every train_step call.
    The target network (self.target_net) is a periodically-copied snapshot
    of the policy network used to compute stable Bellman targets — this
    is the stabilisation technique introduced by Mnih et al. (2015).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 10,
        buffer_capacity: int = 50_000,
    ):
        self.action_size        = action_size
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network — trained every step (Mnih et al. 2015)
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        # Target network — frozen copy used only for computing TD targets
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.update_target_network()   # initialise target = policy
        self.target_net.eval()         # target network is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss — robust to large Q-value outliers

        # Experience replay buffer (Lin 1992)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self._episode_count = 0   # tracks when to refresh the target network

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon a random action is chosen (exploration);
        otherwise the action with the highest predicted Q-value is selected
        (exploitation).  Returns an integer action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Experience storage ────────────────────────────────────────────────────

    def store_experience(self, state, action, reward, next_state, done):
        """Push one (s, a, r, s', done) tuple into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self):
        """
        Sample a random mini-batch and perform one gradient descent step.

        Bellman target (Mnih et al. 2015):
            y = r  +  gamma × max_a Q_target(s', a) × (1 − done)

        The target network Q_target is used (not the policy network) to
        compute the right-hand side.  This decoupling prevents the moving
        target problem that destabilises naive Q-learning with neural networks.

        Returns the scalar loss, or None if the buffer is not yet large
        enough to fill one batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors on the correct device
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions actually taken
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: y = r + gamma × max_a' Q_target(s', a') × (1 − done)
        with torch.no_grad():
            q_next_max = self.target_net(next_states_t).max(dim=1).values
            q_target   = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    # ── Target network management ─────────────────────────────────────────────

    def update_target_network(self):
        """
        Hard copy of policy network weights into the target network.
        Called every target_update_freq episodes (Mnih et al. 2015).
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Multiplicative epsilon decay, clamped to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save policy network weights to filepath."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str):
        """Load policy network weights from filepath."""
        self.policy_net.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from formulas import GreenfieldEnergyEnv

    # ── Initialise environment and agent ───────────────────────────────────────
    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    print("\nDQN Policy Network architecture:")
    print(agent.policy_net)

    # ── Training loop: 3 episodes × 8 760 timesteps ───────────────────────────
    N_EPISODES = 3

    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        total_reward   = 0.0
        total_fuel     = 0.0
        episode_losses = []

        for _ in range(env.n_timesteps):
            # Epsilon-greedy action selection
            action = agent.select_action(state)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer (experience replay — Lin 1992)
            agent.store_experience(state, action, reward, next_state, float(done))

            # One gradient update on a sampled mini-batch
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            total_fuel   += sum(info["fuel_consumed_per_source"].values())

            state = next_state
            if done:
                break

        # End-of-episode bookkeeping
        agent.decay_epsilon()
        agent._episode_count += 1
        if agent._episode_count % agent.target_update_freq == 0:
            agent.update_target_network()

        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        print(
            f"Episode {episode:>2} | "
            f"Total Reward: {total_reward:>10.2f} | "
            f"Fuel Consumed: {total_fuel:>9.2f} L | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Mean Loss: {mean_loss:.6f}"
        )

    print("\nDQN agent trained successfully across 3 episodes without errors.")
