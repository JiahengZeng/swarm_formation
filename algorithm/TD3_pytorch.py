import copy
import time
import envfile
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import utils
# from torch.utils.tensorboard import SummaryWriter

"""项目测试文件：该项目的目标是完成五种形状编队的无人机强化学习的编队重构，以应对当无人机受到损失后的队形维持问题；
五种队形包括：一字型、二字型、三角形、纵列式、双纵列式
在编队重构的过程中可以避障；
开发人员:Central South University-JiaHeng Zeng"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='For formation test')
parser.add_argument('--env', help='choose a env', type=str, default='uavFormation-v1')
parser.add_argument('--render', help='whether to render', type=bool, default=True)
parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--record', help='weather to record', type=int, default=0)
parser.add_argument('--agents_number', help='the number of UAVs', type=int, default=50)
parser.add_argument('--targets_number', help='the number of targets', type=int, default=50)
parser.add_argument('--horizon', help='total steps to stop the game', type=int, default=2000)
parser.add_argument('--log_dir', help='the path to save the record data', type=str, default='.')
parser.add_argument('--map', type=str, default='emptyMap')
parser.add_argument('--LR_A', type=float, default=1e-4)
parser.add_argument('--LR_C', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--replace_iter_a', type=int, default=1100)
parser.add_argument('--replace_iter_c', type=int, default=1000)
parser.add_argument('--memory_capacity', type=int, default=5000)
args = parser.parse_args()

env = envfile.make(args.env,
                   render=args.render,
                   record=args.record,
                   mapID=args.map,
                   directory=args.log_dir,
                   horizon=args.horizon,
                   num_agents=args.agents_number,
                   num_targets=args.targets_number,
                   is_training=False)
STATE_DIM = 11
ACTION_DIM = 2
ACTION_BOUND = [-1, 1]

device = torch.device("cpu")

directory = './plot/'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, 10)
        self.l4 = nn.Linear(10, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.1,
            noise_clip=0.1,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        # self.writer = SummaryWriter(directory + time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time())))
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():

            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor losse
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.total_it)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


policy = TD3(STATE_DIM, ACTION_DIM, ACTION_BOUND[1])


def main():
    done = {'__all__': False}
    obs = env.reset(nb_agents=args.agents_number, nb_targets=args.targets_number)
    if args.load:
        policy.load(f'./algorithm/results/uav')

    while not done['__all__']:
        if args.render:
            env.render()
        action_dictionary = {}
        for agent_id, _ in obs.items():
            action_dictionary[agent_id] = policy.actor.forward(torch.tensor(obs[agent_id][0], dtype=torch.float32))\
                .detach().numpy()
        obs, reward, done, information = env.step(action_dictionary)


if __name__ == '__main__':
    main()
