import numpy as np

METADATA = {
    'version': 1,
    'sensor_r': 10.0,
    'fov': 90,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    'target_init_cov': 30.0,  # initial target diagonal Covariance.
    'target_init_vel': 0.0,  # target's initial velocity.
    'target_vel_limit': 2.0,  # velocity limit of targets.
    'init_distance_min': 5.0,  # the minimum distance btw targets and the agent.
    'init_distance_max': 10.0,  # the maximum distance btw targets and the agent.
    'init_belief_distance_min': 0.0,  # the minimum distance btw belief and the target.
    'init_belief_distance_max': 5.0,  # the maximum distance btw belief and the target.
    'margin': 1.0,  # a marginal distance btw targets and the agent.
    'margin2wall': 0.5,  # a marginal distance from a wall.
    'action_v': [2, 1.33, 0.67, 0],  # action primitives - linear velocities.
    'action_w': [np.pi/2, 0, -np.pi/2],  # action primitives - angular velocities.
    'const_q': 0.001,  # target noise constant in beliefs.
    'const_q_true': 0.01,  # target noise constant of actual targets.
    'flight_distance': 40,  # 无人机之间的飞行最小距离
    'shape_name': 'circular',  # 无人机飞行的形状 'circular'、'one_font'、'column'、'double_column'、'double_one'
    'later_shape_name': 'circular',
    'center_distance': 20,  # 双纵队或双一字队距中心的垂直离
    'is_map_center': True,  # 位于地图中间
    'direction': 0,  # 设置无人机的飞行方向
    'circular_r': 200,  # 无人机编队飞行半径
    'stop_distance': 0.05,  # 无人机与目标位置多近时不需要再调整位置
    'destroy_number': 10,  # 被击毁无人机的数量
    'destroy_happened': 10,  # 击毁无人机的时间步长
    'velocity_x': 3,  # x轴方向上的速度
    'velocity_y': 0,  # y轴方向上的速度
    'velocity_x_target': 3,  # 目标x轴方向上的速度
    'velocity_y_target': 0,     # 目标y轴方向上的速度
    'obstacle_number': 2,   # 障碍物的数量
    'sensor_number': 7,   # 传感器的数量
    'r_sensors': 40,       # 传感器感受野范围
}
