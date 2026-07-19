import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):

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


class ReplayBuffer:

    def __init__(self, capacity: int = 50_000):
        self._buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
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


class DQNAgent:

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

        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self._episode_count = 0


    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())


    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_max = self.target_net(next_states_t).max(dim=1).values
            q_target   = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.policy_net.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )


if __name__ == "__main__":
    from configuration import GreenfieldEnergyEnv

    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    print("\nDQN Policy Network architecture:")
    print(agent.policy_net)

    N_EPISODES = 3

    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        total_reward   = 0.0
        total_fuel     = 0.0
        episode_losses = []

        for _ in range(env.n_timesteps):
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, float(done))

            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            total_fuel   += sum(info["fuel_consumed_per_source"].values())

            state = next_state
            if done:
                break

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

