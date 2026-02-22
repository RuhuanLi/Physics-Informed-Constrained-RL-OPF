import torch
import torch.nn as nn
import random
from collections import deque
import copy
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)

        return (
            torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(a, dtype=np.float32)).to(self.device),
            torch.FloatTensor(r).unsqueeze(1).to(self.device),
            torch.from_numpy(np.array(s2, dtype=np.float32)).to(self.device),
            torch.FloatTensor(d).unsqueeze(1).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, actor, critic, device="cpu", capacity=100000):

        self.actor = actor.to(device)
        self.critic = critic.to(device)

        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        self.buffer = ReplayBuffer(capacity, device)

        self.gamma = 0.99
        self.tau = 0.005

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.actor.G.device
        ).unsqueeze(0)

        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, batch_size):

        if len(self.buffer) < batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(batch_size)

        # -------- Critic --------
        with torch.no_grad():
            next_a = self.actor_target(s2)
            target_Q = self.critic_target(s2, next_a)
            y = r + self.gamma * (1 - d) * target_Q

        current_Q = self.critic(s, a)
        critic_loss = nn.MSELoss()(current_Q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------- Actor --------
        # 这个时候就是得重新计算一次actor 不能用读取的 因为如果使用读取的话就没有梯度了 从state开始计算
        actions = self.actor(s)
        # 原始的critic算出来的Q 还要添加后续的violation
        actor_Q = self.critic(s, actions)

        actor_loss = -actor_Q.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -------- Soft update --------
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
