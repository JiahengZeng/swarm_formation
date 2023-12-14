from envfile.env.formation_env_base import uavFormationbase
import numpy as np
from gym import spaces
from envfile.env.maps import map_utils
from envfile.agent_models import *
from envfile.util import shape_regeneration

"""
针对无人机编队重构强化学习方法构建的环境
描述： 有多个无人机智能体，数量可变，和相同数量的无人机目标飞行位置；
uavFormation_v0:
最简单的编队重构环境，不包含任何障碍物，可以接受碰撞：
智能体观察状态(observation state): [d, alpha] * nb_agents
目标状态: [x, y]
"""


class uavFormationEnv1(uavFormationbase):
    def __init__(self, num_agents=4, num_targets=4, map_name='emptyMap', is_training=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets, map_name=map_name, is_training=is_training)

        self.id = 'uavFomation-v0'
        self.nb_agents = num_agents
        self.nb_targets = num_targets

        self.agent_dim = 3
        self.target_dim = 2
        self.nb_obstacle = METADATA['obstacle_number']
        self.nb_sensors = METADATA['sensor_number']
        self.r_sensors = METADATA['r_sensors']
        self.x_obs_rand = np.random.randint(50, 100, self.nb_obstacle)
        self.y_obs_rand = np.random.randint(0, 450, self.nb_obstacle)

        self.destroy_number = METADATA['destroy_number']  # 被击毁的无人机架数
        self.destroy_happened = METADATA['destroy_happened']  # 击毁发生的步长时间
        self.target_init_velocity = METADATA['target_init_vel'] * np.ones((2,))
        self.stop_distance = METADATA['stop_distance']
        self.limit = {}
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [
            np.concatenate((self.MAP.mapmin, [-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
            np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0]  # 目标与无人机的最大相对速度
        self.limit['state'] = [np.array(([0.0, -np.pi, -rel_vel_limit, -10 * np.pi, -50.0, 0.0])),
                               np.array(([600.0, np.pi, rel_vel_limit, 10 * np.pi, 50.0, 2.0]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1],
                                            dtype=np.float32)  # 规定智能体可以观察到的范围
        self.episode_record = 0  # 记录迭代到了第几个时间步长
        # 建立一个智能体的初始化类列表
        self.setup_agents()
        self.done_dict = {}
        for ii in range(len(self.agents)):
            self.done_dict[self.agents[ii].agent_id] = True
        # 建立一个目标的初始化类列表
        self.setup_targets()
        # 建立奖励
        self.get_reward()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id='agent-' + str(i),
                                dim=self.agent_dim, sampling_period=self.sampling_period,
                                limit=self.limit['agent'],
                                collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                                obstacle_x=self.x_obs_rand,
                                obstacle_y=self.y_obs_rand
                                )
                       for i in range(self.num_agents)]  # 此时未定义policy，因此初始位置不受policy影响

    def setup_targets(self):
        self.targets = [TargetLocation(target_id='target-' + str(i),
                                       dim=self.target_dim, sampling_period=self.sampling_period,
                                       limit=self.limit['target'])
                        for i in range(self.num_targets)]

    def get_reward(self, observed=None, is_training=True):
        return reward_fun(self.nb_agents, is_training)

    def reset(self, **kwargs):
        """
        通过重置reset，在地图中随机产生指定数量的智能体和目标位置；智能体在地图中被分配一个随机的位置，目标则被分配到随机与目标的相邻位置；
        返回一个观察的状态字典，id作为键keys，观察observation作为对应的值(values)；
        """
        try:  # 如果把reset中添加了'nb_agents、nb_targets参数，则每轮的智能体数量不会改变'
            self.nb_agents = kwargs['nb_agents']
            self.nb_targets = kwargs['nb_targets']
        except:  # 否则无人机和目标的数量都会发生随机变化
            self.nb_agents = np.random.random_integers(1, self.num_agents)
            self.nb_targets = np.random.random_integers(1, self.num_targets)
        assert self.nb_agents == self.nb_targets, '目标位置和无人机的数目应当一致！'
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # 初始化智能体和观测列表
        for ii in range(self.nb_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
            self.agents[ii].sensor_reset(self.nb_sensors, self.r_sensors)

        # 初始化目标位置的内容
        for nn in range(self.nb_targets):
            self.targets[nn].reset(init_pose['targets'][nn])
        # 初始化智能体对相应目标位置的观察内容
        for jj in range(self.nb_agents):
            r, alpha = util.relative_distance_polar(self.targets[jj].state[:2],
                                                    xy_base_uav=self.agents[jj].state[:2],
                                                    theta_base_uav=self.agents[jj].state[2])
            obs_dict[self.agents[jj].agent_id].append([1, r / (1000 * np.sqrt(2)), alpha, 0, 1, 1, 1, 1, 1, 1, 1])
        for agent_id in obs_dict:
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])
        return obs_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__': False}
        info_dict = {}
        self.episode_record += 1

        # 目标位置的移动(t -> t+1)
        for jj in range(self.nb_targets):
            x_change, y_change = self.targets[jj].update()

        # 智能体的移动(t -> t+1)并且观察目标位置
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            # 判断无人机是否已经到达了目标位置
            if np.sqrt(np.sum((np.array(self.agents[ii].state[:2]) - np.array(
                    self.targets[ii].state[:2])) ** 2)) < 2:
                self.done_dict[self.agents[ii].agent_id] = True
            else:
                self.done_dict[self.agents[ii].agent_id] = False
            # 根据是否到了目标位置来确定是否要更新行动
            if self.done_dict[self.agents[ii].agent_id] is True:
                action_vw = [0, -1]
            else:
                action_vw = action_dict[agent_id]  # 角速度与线速度的list
            # 保持目标位置和智能体之间的距离
            # margin_pos = [t.state[:2] for t in self.targets[:self.nb_targets]]
            margin_pos = []
            for p, ids in enumerate(action_dict):  # 防止碰撞，一旦和别的无人机的位置信息重合了，则不更新位置
                if agent_id != ids:
                    margin_pos.append(np.array(self.agents[p].state[:2]))
            _ = self.agents[ii].update(action_vw, margin_pos)
            _ = self.agents[ii].sensor_update()

            # 更新对应目标位置观测
            r, alpha = util.relative_distance_polar(self.targets[ii].state[:2],
                                                    xy_base_uav=self.agents[ii].state[:2],
                                                    theta_base_uav=self.agents[ii].state[-1])
            # 更新obstacle位置观测
            obstacle_to_sensor = self.agents[ii].sensor_info[:, 0]

            # 状态变量为： 是否到目标位置, 归一化的距离, 目标的角度,  线速度, 归一化传感器数据
            # action 是v 和 w
            obs_dict[agent_id].append(np.hstack([1 if self.done_dict[self.agents[ii].agent_id] is True
                                                else 0, r / (1000 * np.sqrt(2)), alpha, action_vw[1], obstacle_to_sensor / 40]))  # 状态观测更新 np.hstack
            obs_dict[agent_id] = np.asarray(obs_dict[agent_id])

        # 在指定时间步长后，指定数量的无人机受到袭击，被摧毁，开始编队重构；
        if self.episode_record == self.destroy_happened:
            self.destroy_start_number = int(np.random.randint(1, self.nb_agents, 1))
            self.destroy_end_number = int(np.min([self.destroy_start_number + self.destroy_number, self.nb_agents]))  # 留下来的第一个编号
            for gg in range(self.destroy_end_number - self.destroy_start_number):
                del obs_dict['agent-' + str(self.destroy_start_number - 1 + gg)]
            del self.agents[self.destroy_start_number - 1: self.destroy_end_number - 1]
            del self.targets[self.destroy_start_number - 1: self.destroy_end_number - 1]
            really_destroy_number = np.min([self.nb_agents - self.destroy_start_number + 1, self.destroy_number])
            self.pre_number_agents = self.nb_agents
            self.pre_number_targets = self.nb_targets
            self.nb_agents -= really_destroy_number
            self.nb_targets -= really_destroy_number
            for ii in range(len(self.agents)):
                self.done_dict[self.agents[ii].agent_id] = False

        # 目标位置在部分无人机受到攻击后开始重新生成
        if self.episode_record == self.destroy_happened + 5:
            init_pose_target = shape_regeneration(pre_number=self.pre_number_targets, new_number=self.nb_targets,
                                                  center_position=np.array([self.MAP.origin[0], self.MAP.origin[1]]),
                                                  direction=METADATA['direction'],
                                                  min_distance=METADATA['flight_distance'],
                                                  circular_r=METADATA['circular_r'],
                                                  formation_name=METADATA['later_shape_name'],
                                                  destroy_start_number=self.destroy_start_number,
                                                  destroy_end_number=self.destroy_end_number)
            for nn in range(self.nb_targets):
                self.targets[nn].reset(np.array(init_pose_target[nn]) + np.array([x_change + 350, y_change, 0]))

        reward, done = self.get_reward(self.is_training)
        reward_dict['__all__'], done_dict['__all__'], info_dict['zjh'] = reward, done, 1
        return obs_dict, reward_dict, done_dict, info_dict


def reward_fun(nb_targets, is_training=True):
    reward = 1
    return reward, False
