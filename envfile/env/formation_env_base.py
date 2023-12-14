import os
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from envfile.env.maps import map_utils
from envfile import util
from envfile.metadata import METADATA
from envfile.util import shape_generation


class uavFormationbase(gym.Env):
    def __init__(self, num_agents=4, num_targets=4, map_name='emptyMap', is_training=True, **kwargs):
        self.seed()
        self.id = 'uavFormation-base'
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * len(METADATA['action_w']))
        self.action_map = {}
        for (i, v) in enumerate(METADATA['action_v']):
            for (j, w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v, w)
        assert (len(self.action_map.keys()) == self.action_space.n)  # 当前组合是以元组储存在字典中
        self.sensor_r = METADATA['sensor_r']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(
            map_path=os.path.join(map_dir_path, map_name),
            r_max=self.sensor_r
        )
        self.sampling_period = 0.5  # sec
        self.is_training = is_training
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.reset_num = 0
        self.num_agents = num_agents
        self.nb_agents = num_agents
        self.num_targets = num_targets
        self.nb_targets = num_targets
        self.agent_init_pos = np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        # needed for gym/core.py wrappers
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

    def seed(self, seed=None):  # 当seed为None则会随机产生一个seed值，然后根据这个值不断产生随机数，需要一直保留这个seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setup_agents(self):
        """为环境构建所有的智能体，一旦继承类不给函数重构则会报错"""
        raise NotImplementedError

    def setup_targets(self):
        """为环境构建所有的目标飞行位置，一旦继承类不进行函数重构则报错"""
        raise NotImplementedError

    def reset(self, init_random=True):
        """为每一迭代随机初始化，一旦继承类不给函数重构则会报错"""
        raise NotImplementedError

    def step(self, action_diction):
        """输入智能体的动作，并且更行整个环境以返回观察和奖励(obs, rewards)，一旦继承类不给函数重构则会报错"""
        raise NotImplementedError

    def observation(self, target, agent):
        """
        :param target: 目标飞行位置的全局坐标
        :param agent: 智能体的全局坐标
        :return: 智能体观察到的目标飞行位置与自己的相对坐标
        """
        r, alpha = util.relative_distance_polar(target.state[:2], xy_base_uav=agent.state[:2],
                                                theta_base_uav=agent.state[2])
        observe = np.array([r, alpha])
        return observe

    def gen_rand_pose(self, o_xy, c_theta, min_lin_dist, max_lin_dist, min_ang_dist, max_ang_dist):
        """根据一个点o产生一个随机的位置和角度
        参数设置
        -------
        o_xy: 选定点o的xy位置，根据o点来以一定距离计算生成的随机位置；
        c_theta: 选定一个参考的角度c，根据c来增加或减少一个角度以产生随机角度；
        min_lin_dist: 生成随机位置与点o的距离下限
        max_lin_dist: 生成随机位置与点o的距离上限
        min_ang_dist: 生成随机角度与c的角度差下限
        max_ang_dist: 生成随机角度与c的角度差上限
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2 * np.pi
        rand_ang = util.wrap_around(np.random.rand() * (max_ang_dist - min_ang_dist) + min_ang_dist + c_theta)
        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r * np.cos(rand_ang), rand_r * np.sin(rand_ang)]) + o_xy
        is_valid = not (map_utils.is_collision(self.MAP, rand_xy))
        return is_valid, [rand_xy[0], rand_xy[1], rand_ang]

    def get_init_pose_random(self, lin_dist_range_target=(METADATA['init_distance_min'], METADATA['init_distance_max']),
                             ang_dist_range_target=(-np.pi, np.pi), model='model2',
                             min_distance=METADATA['flight_distance'], shape_name=METADATA['shape_name'],
                             is_map_center=METADATA['is_map_center'], direction=METADATA['direction'],  # 默认方向为0
                             circular_r=METADATA['circular_r'], blocked=False, **kwargs):
        init_pose = {'agents': []}
        if model == 'model1':
            for ii in range(self.nb_agents):
                is_agent_valid = False
                if self.MAP.map is None and ii == 0:
                    if blocked:
                        raise ValueError('Unable to find a blocked initial condition. There is no obstacle in this map')
                    a_init = self.agent_init_pos[:2]  # 无人机只有一架的话将会占领中心位置
                else:
                    while (not is_agent_valid):  # 无人机剩余编号的初始位置
                        a_init = np.random.random((2,)) * (self.MAP.mapmax - self.MAP.mapmin) + self.MAP.mapmin
                        is_agent_valid = not (map_utils.is_collision(self.MAP, a_init))
                init_pose_agent = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
                init_pose['agents'].append(init_pose_agent)  # init_pose是存智能体初始状态的字典
        if model == 'model2':  # 如果采用model2则意味着是无人机部分损毁后的重构模式
            init_pose_target = shape_generation(self.nb_targets,
                                                center_position=np.array([self.MAP.origin[0], self.MAP.origin[1]]),
                                                is_map_center=is_map_center, direction=direction,
                                                min_distance=min_distance, shape_name=shape_name, circular_r=circular_r)
            init_pose['agents'] = init_pose_target

        init_pose['targets'] = []
        if model == 'model1':
            for jj in range(self.nb_targets):  # 方式1：在智能体无人机附近某一角度内生成
                is_agent_valid = False
                while (not is_agent_valid):
                    rand_agent = np.random.randint(self.nb_agents)  # 写一个目标分配
                    is_agent_valid, init_pose_target = self.gen_rand_pose(
                        init_pose['agents'][rand_agent][:2], init_pose['agents'][rand_agent][2],
                        lin_dist_range_target[0], lin_dist_range_target[1],
                        ang_dist_range_target[0], ang_dist_range_target[1]
                    )
                init_pose['targets'].append(init_pose_target)
        if model == 'model2':  # 方式二以预设的图形在指定地方生成目标位置
            init_pose_target = shape_generation(self.nb_targets,
                                                center_position=np.array([self.MAP.origin[0], self.MAP.origin[1]]),
                                                is_map_center=is_map_center, direction=direction,
                                                min_distance=min_distance, shape_name=shape_name, circular_r=circular_r)
            init_pose['targets'] = init_pose_target
        return init_pose

    def get_init_pose(self, init_pose_list=[], **kwargs):
        """
        获得无人机智能体和飞行目标的初始位置
        :param init_pose_list: 预先定义好的初始位置词典;
        lin_dist_range_target: 目标和智能体之间的最小最大距离的元组；
        ang_dist_range_target: 目标和智能体之间的最大最小（逆时针方向）的限制元组；
        :return:
        """
        if init_pose_list:
            self.reset_num += 1
            return init_pose_list[self.reset_num - 1]
        else:
            return self.get_init_pose_random(**kwargs)

    def get_reward(self, observed, is_training=True):
        return reward_fun(is_training)


def reward_fun(is_training):
    pass
