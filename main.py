import gymnasium as gym
import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import wandb


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super(PolicyNet, self).__init__()
        self.fc1: Linear = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2: Linear = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super(ValueNet, self).__init__()
        self.fc1: Linear = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2: Linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x: torch.Tensor = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    """PPO算法,采用截断方式"""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        lmbda: float,
        epochs: int,
        eps: float,
        gamma: float,
        device: torch.device,
    ):
        self.actor: PolicyNet = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic: ValueNet = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer: Adam = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer: Adam = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.gamma: float = gamma
        self.lmbda: float = lmbda
        self.epochs: int = epochs  # 一条序列的数据用来训练轮数
        self.eps: float = eps  # PPO中截断范围的参数
        self.device: torch.device = device

    def take_action(self, state: np.ndarray):
        state: torch.Tensor = torch.tensor(np.array([state]), dtype=torch.float).to(
            self.device
        )
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(
            np.array(transition_dict["states"]), dtype=torch.float
        ).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            np.array(transition_dict["next_states"]), dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach())
            )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    run.log(
                        {
                            "return": np.mean(return_list[-10:]),
                        },
                        step=int(num_episodes / 10 * i + i_episode + 1),
                    )
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode=None)
    env.action_space.seed(0)
    env.observation_space.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )
    run = wandb.init(
        project="mdp-homomorphic-networks",
        config={
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "num_episodes": num_episodes,
            "hidden_dim": hidden_dim,
            "gamma": gamma,
            "lmbda": lmbda,
            "epochs": epochs,
            "eps": eps,
        },
    )

    return_list = train_on_policy_agent(env, agent, num_episodes)
