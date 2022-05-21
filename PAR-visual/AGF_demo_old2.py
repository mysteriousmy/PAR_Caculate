import laspy
import numpy as np
import os
import math
import multiprocessing as mp
import datetime


def diffuse_v(data, tidx, pidx):
    newlist = []
    for t in range(len(tidx[:-1])):
        dtmin, dtmax = tidx[t], tidx[t+1]
        for p in range(len(pidx[:-1])):
            dpmin, dpmax = pidx[p], pidx[p+1]
            if t == len(tidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dpmin) & (data[:, 0] < dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] <= dtmax))[0]
            elif p == len(pidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dpmin) & (data[:, 0] <= dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] < dtmax))[0]
            else:
                condition_data = np.where((data[:, 0] >= dpmin) & (data[:, 0] < dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] < dtmax))[0]
            condition_data = data[condition_data]
            newlist.append(condition_data)
    return newlist


def readlas(lasfile):
    inFile = laspy.read(lasfile)
    x, y, z = inFile.x, inFile.y, inFile.z
    arr = np.c_[x, y, z]
    return arr


def atan2(x, y):
    # radiant
    if x > 0 and y > 0:  # 1
        return math.pi/2 - math.atan(y/x)
    elif x < 0 and y > 0:  # 2
        return math.pi*3/2 - math.atan(y/x)  # 减去负值
    elif x < 0 and y < 0:
        return math.pi*3/2 - math.atan(y/x)  # 加正值
    elif x > 0 and y < 0:
        return math.pi/2 - math.atan(y/x)
    elif x == 0 and y > 0:
        return 0
    elif x == 0 and y < 0:
        return math.pi
    elif y == 0 and x > 0:
        return math.pi/2
    elif y == 0 and x < 0:
        return math.pi*3/2
    elif x == 0 and y == 0:
        return 0


def opp(name, op, x, y, z, result_dict, result_lock):
    # set op as origin point
    newdata_x, newdata_y, newdata_z = x-op[0], y-op[1], z-op[2]
    newdata = np.c_[newdata_x, newdata_y, newdata_z]
    # descartes to polar coordinates
    # print(len(newdata))
    phi = list(map(lambda x, y: atan2(x, y)*180 /
                   math.pi, newdata[:, 0], newdata[:, 1]))
    r = list(map(lambda x, y, z: math.sqrt(x**2 + y**2 + z**2),
                 newdata[:, 0], newdata[:, 1], newdata[:, 2]))
    if r == 0:
        theta = 0
    else:
        theta = list(map(lambda z, r: math.acos(
            z/r)*180/math.pi, newdata[:, 2], r))
    halfball = np.c_[phi, theta, r].astype(np.float32)
    ball_result = {"op": op, "ops": halfball}
    with result_lock:
        result_dict[name] = ball_result
    return ball_result


def doballtest(name, halfball, realdata, tidx, pidx, result_dict, result_lock):
    enewlist = []
    Ngap_num = 0
    enewlist = diffuse_v(halfball["ops"], tidx, pidx)
    enewlist = list(filter(lambda x: len(x) == 0, enewlist))
    Ngap_num = len(enewlist)
    Ngap_num = Ngap_num / ((len(tidx)-1) * (len(pidx)-1))
    op_index = np.where((realdata[:, 0] == halfball["op"][0]) & (
        realdata[:, 1] == halfball["op"][1]) & (realdata[:, 2] == halfball["op"][2]))[0]
    Ngap = {"op_index": op_index, "Ngap_num": Ngap_num}
    with result_lock:
        result_dict[name] = Ngap
    return Ngap


def create_halfball_data_mp(ops, x, y, z, pool):
    param_dict = []
    for i in range(0, len(ops)):
        param_dict.append(
            ['task{}'.format(i), ops[i], x, y, z])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(opp, args=(
        name, op, x, y, z, managed_dict, managed_locker)) for name, op, x, y, z in param_dict]
    results = [p.get() for p in results]
    return managed_dict


def start_cacu_byops(managed_dict, realdata, tidx, pidx):
    param_dict = []
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    for namex, datas in managed_dict.items():
        param_dict.append(
            ['task{}'.format(namex), datas, realdata, tidx, pidx])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results2 = [pool.apply_async(doballtest, args=(
        name, halfball, realdata, tidx, pidx, managed_dict, managed_locker)) for name, halfball, realdata, tidx, pidx in param_dict]
    results2 = [p.get() for p in results2]
    result_item = []
    for namex, datas in managed_dict.items():
        result_item.append(datas)
    return result_item


def init_caculate(v2, realdata, one_dict):

    for i in v2:
        tmin, pmin = 0, 0
        tmax, pmax = 90, 360
        tidx = np.arange(tmin, tmax, i)
        tidx = np.r_[tidx, tmax]
        pidx = np.arange(pmin, pmax, i)
        pidx = np.r_[pidx, pmax]

        agf_result = start_cacu_byops(one_dict, realdata, tidx, pidx)
        return agf_result


def init_data(zhigh_percent, realdata, pool):
    # filter data above the calculated height layer
    filterindex = np.where(realdata[:, 2] >= realdata[:, 2].min(
    )+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]

    x, y, z = filter_xyz[:, 0], filter_xyz[:, 1], filter_xyz[:, 2]
    one_dict = create_halfball_data_mp(filter_xyz, x, y, z, pool)
    return one_dict
