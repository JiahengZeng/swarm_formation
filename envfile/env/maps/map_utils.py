"""
一些用于支撑栅格地图生成的函数
用于生成一个2D的地图，(xmin, ymin)位于地图的左下角
从左向右x轴的坐标增长，从下至上y的坐标轴增长；
在Matplotlib.pyplot 中原点(0, 0)位于左上角，因此在该地图中，地图是翻转的；
"""
import numpy as np
import yaml


class GridMap(object):
    def __init__(self, map_path, r_max=1.0, fov=np.pi, margin2wall=0.5):
        map_config = yaml.load(open(r"./envfile/env/maps/" + map_path + '.yaml', 'r'), Loader=yaml.FullLoader)
        self.mapdim = map_config['mapdim']
        self.mapres = np.array(map_config['mapres'])
        self.mapmin = np.array(map_config['mapmin'])
        self.mapmax = np.array(map_config['mapmax'])
        self.margin2wall = margin2wall
        self.origin = map_config['origin']
        self.r_max = r_max
        self.pi = fov
        if 'empty' in map_path: self.map = None

    def se2_to_cell(self, pos):  # 将实际位置转换为相关坐标
        pos = pos[:2]  # 取position的头两个即坐标(x,y)
        cell_idx = (pos - self.mapmin) / self.mapres - 0.5
        return round(cell_idx[0]), round(cell_idx[1])

    def cell_to_se2(self, cell_indx):  # 将相关坐标转换为实际值
        return (np.array(cell_indx) + 0.5) * self.mapres + self.mapmin


def is_collision(map_obj, pos):  # 判定是否在界内，如果在界内则返回True
    if not (in_bound(map_obj, pos)):
        return True
    else:
        if map_obj.map is not None:
            n = np.ceil(map_obj.margin2wall / map_obj.mapres).astype(np.int16)
            cell = np.minimum([map_obj.mapdim[0] - 1, map_obj.mapdim[1] - 1], map_obj.se2_to_cell(pos))
            for r_add in np.arange(-n[1], n[1], 1):
                for c_add in np.arange(-n[0], n[0], 1):
                    x_c = np.clip(cell[0] + r_add, 0, map_obj.mapdim[0] - 1).astype(np.int16)
                    y_c = np.clip(cell[1] + c_add, 0, map_obj.mapdim[1] - 1).astype(np.int16)
                    idx = x_c + map_obj.mapdim[0] * y_c
                    if map_obj.map_linear[idx] == 1:
                        return True
    return False


def in_bound(map_obj, pos):  # 出界判定，如果出界了则返回False
    return not ((pos[0] < map_obj.mapmin[0] + map_obj.margin2wall)
                or (pos[0] > map_obj.mapmax[0] - map_obj.margin2wall)
                or (pos[1] < map_obj.mapmin[1] + map_obj.margin2wall)
                or (pos[1] > map_obj.mapmax[1] - map_obj.margin2wall))
