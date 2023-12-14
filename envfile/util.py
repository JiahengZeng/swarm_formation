import numpy as np
from envfile.metadata import METADATA

def wrap_around(x):
    # 把角度限制在 [-pi,pi) 内
    if x >= np.pi:
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    else:
        return x


def transfrom_2d(vec, theta_base_uav, xy_base_uav=None):
    """在该函数中：vec和xy_base_uav都是在全局坐标系下的坐标向量，通过角度和位置关系，
    求得vec相对目标位置的坐标变换;
    R^T * (vec - theta_base_uav)
    其中R是在全局坐标系下的旋转矩阵,由无人机相对于全球坐标系的角度theta决定"""
    if xy_base_uav is None:
        xy_base_uav = [0.0, 0.0]
    assert (len(vec) == 2)
    return np.matmul([[np.cos(theta_base_uav), np.sin(theta_base_uav)],
                      [-np.sin(theta_base_uav), np.cos(theta_base_uav)]],
                      vec - np.array(xy_base_uav))


def cartesian2polar(xy):
    r = np.sqrt(np.sum(xy**2))
    alpha = np.arctan2(xy[1], xy[0])
    return r, alpha


def relative_distance_polar(xy_target, theta_base_uav, xy_base_uav):
    """
    :param xy_target: 目标位置的全球坐标系-xy坐标
    :param theta_base_uav: 智能体无人机的角度
    :param xy_base_uav:  智能体无人机的全球坐标系-xy坐标
    :return: 返回目标位置相对于无人机的极坐标
    """
    xy_target_base = transfrom_2d(xy_target, theta_base_uav, xy_base_uav)
    return cartesian2polar(xy_target_base)


def shape_generation(nb_agents=None, center_position=None, is_map_center=False,
                     direction=0, min_distance=None, shape_name='one_font', circular_r=None):
    if nb_agents is None:
        ValueError('The number of agents can\'t be None.')
    if is_map_center is False:
        center_position = np.array([np.random.random(), np.random.random()])  # 还没设计好
    if shape_name == 'one_font':
        return one_font_generation(nb_agents, center_position, direction, min_distance)
    if shape_name == 'circular':
        if circular_r is None:
            ValueError('The circular_r can\'t be None.')
        return circular_generation(nb_agents, center_position, direction, circular_r, min_distance)
    if shape_name == 'column':
        return column_generation(nb_agents, center_position, direction, min_distance)
    if shape_name == 'double_column':
        return double_column_generation(nb_agents, center_position, direction, min_distance, METADATA['center_distance'])
    if shape_name == 'double_one':
        return double_one_font_generation(nb_agents, center_position, direction, min_distance, METADATA['center_distance'])


def circular_generation(nb_agents, center_position, direction, circular_r, min_distance):
    rho = circular_r
    interval = 2 * np.pi / nb_agents
    list_temporary = np.array([x for x in range(nb_agents)]) * interval + direction
    coordinate_list = []
    for angle in list_temporary:
        coordinate_list.append([center_position[0] + rho * np.cos(angle), center_position[1] + rho * np.sin(angle), direction])
    if np.sqrt(np.square(coordinate_list[0][0] - coordinate_list[1][0]) +
               np.square(coordinate_list[0][1] - coordinate_list[1][1])) < min_distance:  # 因为等距所以只需要判断一对最近目标
        ValueError('It is dangerous since uav\'s distance is too long!')
    return coordinate_list


def one_font_generation(nb_agents, center_position, direction, min_distance):
    total_length = (nb_agents - 1) * min_distance
    x0 = center_position[0] - total_length * np.cos(direction - np.pi/2) / 2
    y0 = center_position[1] - total_length * np.sin(direction - np.pi/2) / 2

    dx = min_distance * np.cos(direction - np.pi/2)
    dy = min_distance * np.sin(direction - np.pi/2)
    coordinate_list = []
    for ii in range(nb_agents):
        coordinate_list.append([x0, y0, direction])
        x0 += dx
        y0 += dy
    return coordinate_list


def double_one_font_generation(nb_agents, center_position, direction, min_distance, center_distance):
    total_length = (nb_agents / 2 - 1) * min_distance
    x0 = center_position[0] - total_length * np.cos(direction - np.pi / 2) / 2 + center_distance
    x1 = center_position[0] - total_length * np.cos(direction - np.pi / 2) / 2 - center_distance
    y0 = center_position[1] - total_length * np.sin(direction - np.pi / 2) / 2
    y1 = center_position[1] - total_length * np.sin(direction - np.pi / 2) / 2

    dx = min_distance * np.cos(direction - np.pi / 2)
    dy = min_distance * np.sin(direction - np.pi / 2)
    coordinate_list = []
    for ii in range(int(nb_agents / 2)):
        coordinate_list.append([x0, y0, direction])
        x0 += dx
        y0 += dy
    for ii in range(int(nb_agents / 2)):
        coordinate_list.append([x1, y1, direction])
        x1 += dx
        y1 += dy
    return coordinate_list


def column_generation(nb_agents, center_position, direction, min_distance):
    total_length = (nb_agents - 1) * min_distance
    x0 = center_position[0] - total_length * np.cos(direction) / 2
    y0 = center_position[1] - total_length * np.sin(direction) / 2

    dx = min_distance * np.cos(direction)
    dy = min_distance * np.sin(direction)
    coordinate_list = []
    for ii in range(nb_agents):
        coordinate_list.append([x0, y0, direction])
        x0 += dx
        y0 += dy
    return coordinate_list


def double_column_generation(nb_agents, center_position, direction, min_distance, center_distance):
    total_length = (nb_agents / 2 - 1) * min_distance
    x0 = center_position[0] - total_length * np.cos(direction) / 2
    x1 = center_position[0] + total_length * np.cos(direction) / 2
    y0 = center_position[1] - total_length * np.sin(direction) / 2 + center_distance
    y1 = center_position[1] - total_length * np.sin(direction) / 2 - center_distance

    dx = min_distance * np.cos(direction)
    dy = min_distance * np.sin(direction)
    coordinate_list = []
    for ii in range(int(nb_agents / 2)):
        coordinate_list.append([x0, y0, direction])
        x0 += dx
        y0 += dy
    for ii in range(int(nb_agents / 2)):
        coordinate_list.append([x1, y1, direction])
        x1 -= dx
        y1 += dy
    return coordinate_list


def shape_regeneration(pre_number, new_number, center_position, direction, min_distance,
                       circular_r, formation_name, destroy_start_number, destroy_end_number):
    if formation_name == 'one_font':
        total_length = (pre_number - 1) * min_distance
        x0 = center_position[0] - total_length * np.cos(direction - np.pi/2) / 2
        y0 = center_position[1] - total_length * np.sin(direction - np.pi/2) / 2

        dx = total_length / new_number * np.cos(direction - np.pi/2)
        dy = total_length / new_number * np.sin(direction - np.pi/2)
        coordinate_list = []
        for ii in range(new_number):
            coordinate_list.append([x0, y0, direction])
            x0 += dx
            y0 += dy
        return coordinate_list
    elif formation_name == 'circular':
        coordinate_list = circular_generation(new_number, center_position, direction, circular_r, min_distance)
        move_number = int((destroy_end_number - destroy_start_number) / 2)
        for jj in range(move_number - 1):
            coordinate_list.append(coordinate_list[0])
            del coordinate_list[0]
        return coordinate_list

    elif formation_name == 'column':
        total_length = (pre_number - 1) * min_distance
        x0 = center_position[0] - total_length * np.cos(direction) / 2
        y0 = center_position[1] - total_length * np.sin(direction) / 2

        dx = total_length / new_number * np.cos(direction)
        dy = total_length / new_number * np.sin(direction)
        coordinate_list = []
        for ii in range(new_number):
            coordinate_list.append([x0, y0, direction])
            x0 += dx
            y0 += dy
        return coordinate_list

    elif formation_name == 'double_column':
        total_length = (pre_number / 2 - 1) * min_distance
        x0 = center_position[0] - total_length * np.cos(direction) / 2
        x1 = center_position[0] + total_length * np.cos(direction) / 2
        y0 = center_position[1] - total_length * np.sin(direction) / 2 + METADATA['center_distance']
        y1 = center_position[1] - total_length * np.sin(direction) / 2 - METADATA['center_distance']

        dx = total_length / (new_number / 2 - 1) * np.cos(direction)
        dy = total_length / (new_number / 2 - 1) * np.sin(direction)
        coordinate_list = []
        for ii in range(int(new_number / 2)):
            coordinate_list.append([x0, y0, direction])
            x0 += dx
            y0 += dy
        for ii in range(int(new_number / 2)):
            coordinate_list.append([x1, y1, direction])
            x1 -= dx
            y1 += dy
        return coordinate_list

    elif formation_name == 'double_one':
        total_length = (pre_number / 2 - 1) * min_distance
        x0 = center_position[0] - total_length * np.cos(direction - np.pi / 2) / 2 + METADATA['center_distance']
        x1 = center_position[0] - total_length * np.cos(direction - np.pi / 2) / 2 - METADATA['center_distance']
        y0 = center_position[1] - total_length * np.sin(direction - np.pi / 2) / 2
        y1 = center_position[1] - total_length * np.sin(direction - np.pi / 2) / 2

        dx = total_length / (new_number / 2 - 1) * np.cos(direction - np.pi / 2)
        dy = total_length / (new_number / 2 - 1) * np.sin(direction - np.pi / 2)

        coordinate_list = []
        for ii in range(int(new_number / 2)):
            coordinate_list.append([x0, y0, direction])
            x0 += dx
            y0 += dy
        for ii in range(int(new_number / 2)):
            coordinate_list.append([x1, y1, direction])
            x1 += dx
            y1 += dy
        return coordinate_list
