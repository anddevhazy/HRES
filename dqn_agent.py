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
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrioritizedReplayBuffer:

    def __init__(
        self,
        capacity:  int   = 100_000,
        alpha:     float = 0.6,
        beta_start: float = 0.4,
        beta_end:   float = 1.0,
        beta_steps: int   = 100_000,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._step      = 0

        self._buffer    = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._pos        = 0

    @property
    def beta(self) -> float:
        fraction = min(1.0, self._step / self.beta_steps)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        max_prio = self._priorities.max() if self._buffer else 1.0
        if len(self._buffer) < self.capacity:
            self._buffer.append(None)
        self._buffer[self._pos]     = (state, action, reward, next_state, done)
        self._priorities[self._pos] = max_prio
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        n      = len(self._buffer)
        prios  = self._priorities[:n]
        probs  = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self._buffer[i] for i in indices]


        weights  = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        self._step += 1

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self._priorities[idx] = abs(float(err)) + 1e-6

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:

    def __init__(
        self,
        state_size:        int   = 22,
        action_size:       int   = 7,
        lr:                float = 5e-4,
        gamma:             float = 0.97,
        epsilon:           float = 1.0,
        epsilon_min:       float = 0.05,
        epsilon_decay:     float = 0.995,
        batch_size:        int   = 128,
        target_update_freq: int  = 10,
        buffer_capacity:   int   = 100_000,
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.replay_buffer  = PrioritizedReplayBuffer(capacity=buffer_capacity)
        self._episode_count = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        (states, actions, rewards, next_states,
         dones, indices, weights) = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)
        weights_t     = torch.FloatTensor(weights).to(self.device)


        q_current = (self.policy_net(states_t)
                     .gather(1, actions_t.unsqueeze(1))
                     .squeeze(1))


        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            q_next       = (self.target_net(next_states_t)
                            .gather(1, next_actions)
                            .squeeze(1))
            q_target     = rewards_t + self.gamma * q_next * (1.0 - dones_t)


        td_errors = (q_target - q_current).detach().cpu().numpy()
        loss      = (self.loss_fn(q_current, q_target) * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()


        self.replay_buffer.update_priorities(indices, td_errors)

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def step_scheduler(self):
        self.scheduler.step()

    def save(self, filepath: str):
        torch.save({
            "policy_net":    self.policy_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "episode_count": self._episode_count,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "epsilon" in ckpt:
            self.epsilon = ckpt["epsilon"]
        if "episode_count" in ckpt:
            self._episode_count = ckpt["episode_count"]


if __name__ == "__main__":
    from configuration import GreenfieldEnergyEnv

    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    print("\nDQN Policy Network architecture:")
    print(agent.policy_net)
    print(f"\nDevice : {agent.device}")
    print(f"Gamma  : {agent.gamma}  (high — agent plans ahead)")
    print(f"Buffer : PrioritizedReplayBuffer (capacity=100,000)")

    N_EPISODES = 3
    for episode in range(1, N_EPISODES + 1):
        state          = env.reset()
        total_reward   = 0.0
        total_fuel     = 0.0
        episode_losses = []

        for _ in range(env.n_timesteps):
            action                         = agent.select_action(state)
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
            f"Episode {episode:>2} | Reward {total_reward:>10.2f} | "
            f"Fuel {total_fuel:>9.2f} L | ε {agent.epsilon:.4f} | "
            f"Loss {mean_loss:.6f}"
        )

    print("\nDQN agent smoke-test passed.")
