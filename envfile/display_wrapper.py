"""
用于多智能体强化学习编队重构的实时渲染————中南大学曾家恒
"""
from gym import Wrapper
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from envfile.metadata import *
import matplotlib.patches as patches

matplotlib.use('TkAgg')


class Display2D(Wrapper):
    def __init__(self, env, fig_id=0, skip=1):
        super(Display2D, self).__init__(env)
        self.fig_id = fig_id  # fig_id = 0: train, fig_id = 1: test
        self.env_core = env.env
        self.bin = self.env_core.MAP.mapres  # 单位长度
        if self.env_core.MAP.map is None:
            self.map = np.zeros(self.env_core.MAP.mapdim)  # 初始化地图矩阵
        else:
            self.map = self.env_core.MAP.map
        self.mapmin = self.env_core.MAP.mapmin
        self.mapmax = self.env_core.MAP.mapmax
        self.mapres = self.env_core.MAP.mapres
        self.fig = plt.figure(self.fig_id, figsize=(9.5, 9.5))
        self.n_frames = 0
        self.skip = skip


    def close(self):
        plt.close(self.fig)

    def render(self, mode='empty', record=False, traj_num=0, batch_outputs=None):
        if not hasattr(self, 'traj'):
            raise ValueError('Must do a env.reset() first before calling env.render()')
        num_agents = self.env_core.nb_agents
        if type(self.env_core.agents) == list:
            agent_pos = [self.env_core.agents[i].state for i in range(num_agents)]
        else:
            agent_pos = self.env_core.agents.state

        num_targets = self.env_core.nb_targets
        if type(self.env_core.targets) == list:
            target_pos = [self.env_core.targets[i].state[:2] for i in range(num_targets)]
        else:
            target_pos = self.env_core.targets.state[:, :2]

        if self.n_frames % self.skip == 0:
            self.fig.clf()
            ax = self.fig.subplots()
            ax.add_patch(patches.Rectangle((1000, 0), .1, .1))    # 20 ~ 450 y-axis
            for qq in range(self.env_core.nb_obstacle):
                ax.add_patch(patches.Rectangle((self.env_core.x_obs_rand[qq], self.env_core.y_obs_rand[qq]),
                                               50, 50, hatch="+"))
            for ii in range(num_agents):
                # 智能体的位置
                ax.plot(agent_pos[ii][0], agent_pos[ii][1], marker='o', markersize=2.4,
                    linestyle='None', markerfacecolor='r', markeredgecolor='r')
                ax.plot(self.traj[ii][0], self.traj[ii][1], 'g.', markersize=0.2)   # 轨迹生成
                # sensor_arc = patches.Arc((agent_pos[ii][0], agent_pos[ii][1]), METADATA['sensor_r'] * 2,
                #                          METADATA['sensor_r'] * 2,
                #                          angle=agent_pos[ii][2] / np.pi * 180, theta1=-METADATA['fov'] / 2,
                #                          theta2=METADATA['fov'] / 2, facecolor='gray')

                # 传感器的示意图
                # ax.add_patch(sensor_arc)
                # ax.plot([agent_pos[ii][0], agent_pos[ii][0] + METADATA['sensor_r'] * np.cos(
                #     agent_pos[ii][2] + 0.5 * METADATA['fov'] / 180.0 * np.pi)],
                #         [agent_pos[ii][1], agent_pos[ii][1] + METADATA['sensor_r'] * np.sin(
                #             agent_pos[ii][2] + 0.5 * METADATA['fov'] / 180.0 * np.pi)], 'k', linewidth=0.5)
                # ax.plot([agent_pos[ii][0], agent_pos[ii][0] + METADATA['sensor_r'] * np.cos(
                #     agent_pos[ii][2] - 0.5 * METADATA['fov'] / 180.0 * np.pi)],
                #         [agent_pos[ii][1], agent_pos[ii][1] + METADATA['sensor_r'] * np.sin(
                #             agent_pos[ii][2] - 0.5 * METADATA['fov'] / 180.0 * np.pi)], 'k', linewidth=0.5)
                self.traj[ii][0].append(agent_pos[ii][0])
                self.traj[ii][1].append(agent_pos[ii][1])

            # for jj in range(num_targets):
            #     ax.plot(target_pos[jj][0], target_pos[jj][1], marker='o', markersize=3,
            #         linestyle='None', markerfacecolor='none', markeredgecolor='b')
                # self.traj_y[jj][0].append(target_pos[jj][0])
                # self.traj_y[jj][1].append(target_pos[jj][1])

            ax.set_aspect('equal', 'box')
            ax.grid()
            ax.set_title(' '.join([mode.upper(), ': Trajectory', str(traj_num)]))

            if not record:
                plt.draw()
                plt.pause(0.000005)
        self.n_frames += 1

    def reset(self, **kwargs):
            self.traj = [[[], []]] * self.env_core.num_agents
            self.traj_y = [[[], []]] * self.env_core.num_targets
            return self.env.reset(**kwargs)


class Video2D(Wrapper):
    def __init__(self, env, dirname='', skip=1, dpi=80, local_view=False):
        super(Video2D, self).__init__(env)
        self.local_view = local_view
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fname = os.path.join(dirname, 'eval_%da%dt.mp4' % (env.nb_agents, env.nb_targets))
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        if self.local_view:
            self.moviewriter0 = animation.FFMpegWriter()
            fname0 = os.path.join(dirname, 'train_local_%d.mp4' % np.random.randint(0, 20))
            self.moviewriter0.setup(fig=env.fig0, outfile=fname0, dpi=dpi)
        self.n_frames = 0

    def render(self, *args, **kwargs):
        if self.n_frames % self.skip == 0:
            self.env.render(record=True, *args, **kwargs)
        self.moviewriter.grab_frame()
        if self.local_view:
            self.moviewriter0.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
