from envfile.TimeRepackaging import uav_wrapper


def make(env_name, render=False, record=False, directory='', mapID='emptyMap',
         horizon=None, num_agents=4, **kwargs):
    """
    :param env_name: 环境的名称
    :param render: 是否渲染迭代过程
    :param record: 是否记录迭代过程信息
    :param directory: 存储信息的目标目录
    :param mapID: 选择地图
    :param horizon: 时间步长迭代上限
    :param num_agents: 智能体无人机的数量
    :param kwargs: 多余的参数输入
    :return: 返回可供交互的智能体环境，继承gym库
    """
    if horizon is None:  # 若没有上限输入则迭代上限设置为200步长
        horizon = 200

    if env_name == 'uavFormation-v0':
        from envfile.env.uavFormation_v0 import uavFormationEnv0
        env0 = uavFormationEnv0(num_agents=num_agents, map_name=mapID, **kwargs)
    elif env_name == 'uavFormation-v1':
        from envfile.env.uavFormation_v1 import uavFormationEnv1
        env0 = uavFormationEnv1(num_agents=num_agents, map_name=mapID, **kwargs)
    # ... some other env
    else:
        raise ValueError('No such environment exists.')

    env = uav_wrapper(env0, horizon=horizon)

    if render:
        from envfile.display_wrapper import Display2D
        env = Display2D(env, fig_id=0)
    if record:
        from envfile.display_wrapper import Video2D
        env = Video2D(env, dirname=directory)

    return env