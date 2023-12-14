import gym
"""
通过使用gym.Wrapper来继承gym库的限制管理，唯一不同的是对gym中的时间限制文件time_limit.py进行更改
使得返回的判别完成与否内容是字典，而不是布尔类型；
"""


class uav_wrapper(gym.Wrapper):
    def __init__(self, env, horizon=None):
        super(uav_wrapper, self).__init__(env)
        if horizon is None and self.env.spec is not None:
            horizon = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = horizon
        self._horizon = horizon
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._horizon:
            info['TimeLimit.truncated'] = not done
            done['__all__'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

