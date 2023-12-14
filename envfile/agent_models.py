"""动态对象模型
TargetLocation ：目标位置的模型
                状态 ： x,y
AgentSE2 : 智能体的SE(2)
        状态 x,y,theta
Agent2DFixedPath : 具有预定义路径的模型
Agent_InfoPlanner ： 来自Info库的模型

SE2Dynamics ：用控制输入更新动力学函数——线速度 与 角速度
SEDynamicsVel : 采用连续线速度与角速度动力学更新函数的模式

针对多无人机编队问题建设的智能体模型--中南大学曾家恒
"""

import numpy as np
from envfile.metadata import METADATA
import envfile.util as util


class Agent(object):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, margin=METADATA['margin']):
        self.agent_id = agent_id
        self.dim = dim
        self.sampling_period = sampling_period
        self.limit = limit
        self.collision_func = collision_func
        self.margin = margin

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos)**2, axis=1)) < self.margin)  # no update

    def reset(self, init_state):
        self.state = init_state
        # [x0, y0, direction0]


class AgentSE2(Agent):   # 给智能体建立的动态方程模型
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, obstacle_x, obstacle_y,
                 margin=METADATA['margin'], policy=None):
        super(AgentSE2, self).__init__(agent_id, dim, sampling_period, limit, collision_func, margin=margin)
        self.policy = policy
        self.obstacle_coords = [0] * len(obstacle_x)
        for ii in range(len(obstacle_x)):
            self.obstacle_coords[ii] = np.array([[obstacle_x[ii], obstacle_y[ii]], [obstacle_x[ii] + 50, obstacle_y[ii]],
                                                 [obstacle_x[ii] + 50, obstacle_y[ii]+ 50],
                                                 [obstacle_x[ii], obstacle_y[ii] + 50]])

    def reset(self, init_state):
        super().reset(init_state)
        if self.policy:
            self.policy.reset(init_state)

    def update(self, control_input=None, margin_pos=None):
        """
        control_input: [角速度， 线性速度]
        margin_pos: 距离目标的最小距离
        """
        if control_input is None:
            control_input = self.policy.get_control(self.state)
        new_state = SE2Dynamics(self.state, self.sampling_period, control_input)
        is_col = 0

        #  下面的if不会调用，暂时留着为了碰撞检查
        if self.collision_check(new_state[:2]):  # 目前永远是False
            is_col = 1
            new_state[:2] = self.state[:2]   # 撞墙了则不会前进
            if self.policy is not None:
                corrected_policy = self.policy.collision(new_state)  # 修正一下policy产生的控制
                if corrected_policy is not None:
                    new_state = SE2DynamicsVel(self.state, self.sampling_period, corrected_policy)
        elif margin_pos is not None:
            if self.margin_check(new_state[:2], margin_pos):
                new_state[:2] = self.state[:2]
        self.state = new_state
        self.range_check()  # 将state强行clip在范围内MAP.mapmin和np.pi
        return is_col  # 目前只更新了位置，返回该参数没有作用

    def sensor_reset(self, sensor_nb, sensor_r):
        self.sensor_nb = sensor_nb
        self.sensor_r = sensor_r
        self.sensor_info = sensor_r + np.zeros((sensor_nb, 3))  # distance, x, y
        self.sensor_update()

    def sensor_update(self):
        cx, cy, rotation = self.state
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, self.sensor_nb)
        # [1, 1, 1, 1, 1, 1, 1]
        xs = (np.zeros(self.sensor_nb, ) + self.sensor_r) * np.cos(sensor_theta)
        ys = (np.zeros(self.sensor_nb, ) + self.sensor_r) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])

        tmp_x = xys[:, 0]
        tmp_y = xys[:, 1]
        rotated_x = tmp_x * np.cos(self.state[2]) - tmp_y * np.sin(self.state[2])
        rotated_y = tmp_x * np.sin(self.state[2]) + tmp_y * np.cos(self.state[2])
        self.sensor_info[:, -2:] = np.vstack([rotated_x + cx, rotated_y + cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q  # 回归到原点处的传感器尖端坐标
            possible_sensor_distance = [self.sensor_r]
            possible_intersections = [self.sensor_info[si, -2:]]

            for o_index in range(len(self.obstacle_coords)):
                for oi in range(len(self.obstacle_coords[o_index])):
                    p = self.obstacle_coords[o_index][oi]
                    r = self.obstacle_coords[o_index][(oi + 1) % len(self.obstacle_coords[o_index])] \
                        - self.obstacle_coords[o_index][oi]
                    if np.cross(r, s) != 0:
                        t = np.cross((q - p), s) / np.cross(r, s)
                        u = np.cross((q - p), r) / np.cross(r, s)
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            intersection = q + u * s
                            possible_intersections.append(intersection)  # 雷达交互点
                            possible_sensor_distance.append(np.linalg.norm(u * s))  # 与机体的距离

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]


def SE2Dynamics(x, dt, u):
    assert (len(x) == 3)
    tw = dt * u[0]  # delta w
    velocity = (u[1] + 1) / 2 * 20
    if abs(tw) < 0.001:
        diff = np.array([dt * velocity * np.cos(x[2] + tw/2) + dt * METADATA['velocity_x'],
                         dt * velocity * np.sin(x[2] + tw/2) + dt * METADATA['velocity_y'], tw])  # 微小的时候算平均，额外加上了整体的移动也就是METADATA['VELOCITY_X']
    else:
        diff = np.array([velocity / u[0] * (np.sin(x[2] + tw) - np.sin(x[2])) + dt * METADATA['velocity_x'],
                         velocity / u[0] * (np.cos(x[2]) - np.cos(x[2] + tw)) + dt * METADATA['velocity_y'],
                         tw])  # 改变很大则不采用平均
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x


def SE2DynamicsVel(x, dt, u=None):
    """
    输入：x=[x坐标， y坐标， theta智能体的角度， v智能体的线加速度， w智能体的角加速度]；
        dt指时间间隔；u则提取加速度内容
    输出：智能体新的坐标位置与角度；后两位则继续保持了原先的线加速度与角加速度
    """
    assert(len(x) == 5)  # x = [x,y,theta,v,w]
    if u is None:
        u = x[-2:]
    odom = SE2Dynamics(x[:3], dt, u)
    return np.concatenate((odom, u))


class TargetLocation(object):
    def __init__(self, target_id, dim, sampling_period, limit, margin=METADATA['margin']):
        self.target_id = target_id
        self.dim = dim
        self.sampling_period = sampling_period
        self.limit = limit
        self.margin = margin
        self.state = None
        self.x_change = 0
        self.y_change = 0

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def reset(self, init_state):
        self.state = init_state

    def update(self):
        self.state[0] += self.sampling_period * METADATA['velocity_x_target']
        self.state[1] += self.sampling_period * METADATA['velocity_y_target']
        self.x_change += self.sampling_period * METADATA['velocity_x_target']
        self.y_change += self.sampling_period * METADATA['velocity_y_target']
        return self.x_change, self.y_change









