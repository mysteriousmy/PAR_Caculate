from cmath import log
from hashlib import new
import re
import laspy
import numpy as np
import os
import multiprocessing as mp


def direct_v_two(data, v0, v1, dymax, dxmax, dymin, dxmin, zmax, meanp):
    xmin, ymin = dxmin, dymin
    xmax, ymax = dxmax, dymax
    xidx = np.arange(xmin, xmax, v1)
    xidx = np.r_[xidx, xmax]
    yidx = np.arange(ymin, ymax, v1)
    yidx = np.r_[yidx, ymax]
    condition_data = []
    newlist = []
    for y in range(len(yidx[:-1])):
        dymin, dymax = yidx[y], yidx[y+1]
        for x in range(len(xidx[:-1])):
            dxmin, dxmax = xidx[x], xidx[x+1]
            if y == len(yidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] <= dymax))[0]
            elif x == len(xidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] <= dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax))[0]
            else:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax))[0]
            condition_data = data[condition_data]
            if len(condition_data) != 0:
                newlist.append(condition_data)
    weighted = calculate_weighted(newlist, v0, v1, zmax, meanp)
    # weighted = 1 - (weighted / ((v0 / v1) ** 2)) 0.01 25: 8.9% 50: 16.9% 75:53.2%
    # 0.01 25: 21.9% 50: 29.32% 75:59.89%
    weighted = 1 - (weighted / (len(xidx[:-1]) * len(yidx[:-1])))
    return weighted


def direct_v(data, data_op, yidx, xidx, v0, v1):
    newlist = []
    for y in range(len(yidx[:-1])):
        dymin, dymax = yidx[y], yidx[y+1]
        for x in range(len(xidx[:-1])):
            dxmin, dxmax = xidx[x], xidx[x+1]
            if y == len(yidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] <= dymax))[0]
            elif x == len(xidx[:-1])-1:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] <= dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax))[0]
            else:
                condition_data = np.where((data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax))[0]
            condition_data = data[condition_data]
            if len(condition_data) != 0:
                newlist.append(condition_data)
    weighted = calculate_weighted(newlist, data_op, v0, v1)
    # weighted = 1 - (weighted / ((v0 / v1) ** 2)) 0.01 25: 8.9% 50: 16.9% 75:53.2%
    # 0.01 25: 21.9% 50: 29.32% 75:59.89%
    weighted = 1 - (weighted / ((v0 / v1) ** 2))
    # weighted = 1
    # if len(newlist) != 0:
    #     weighted = calculate_weighted(newlist, v0, v1, zmax, meanp)
    #     # weighted = 1 - (weighted / ((v0 / v1) ** 2)) 0.01 25: 8.9% 50: 16.9% 75:53.2%
    #     # 0.01 25: 21.9% 50: 29.32% 75:59.89%
    #     weighted = 1 - (weighted / ))
    return weighted


def calculate_weighted(data, data_op, v0, v1):
    D = 0
    deltai = 0
    sizelist = []
    sizelist = list(map(lambda x: len(x), data))
    # if len(sizelist) != 0:
    #     mediandata = get_list_median(get_list_median(data))
    # D = sum(sizelist) / len(data)
    D = sum(sizelist) / (v0 * v0)
    for j in data:
        di = len(j) / (v1 * v1)
        wdi = di / D

        if wdi > 1:
            wdi = 1
        else:
            wdi = 1 - wdi
        e0 = data_op[0] + data_op[1]
        eidata = get_list_median(j)
        ei = data_op[1] + data_op[0] - (eidata[0] + eidata[1])
        wei = 1 - (ei/e0) * (2 * (v1 / v0))
        deltai += (1 * wdi)
    return deltai


def get_list_median(data):
    result = []
    listsize = len(data)
    if listsize % 2 == 0:
        median = data[listsize//2]
        result = median
    if listsize % 2 == 1:
        median = data[(listsize-1)//2]
        result = median
    return result


def caculate_mp(name, data, realdata, v0, v1, result_dict, result_lock):
    xmin, ymin, _ = data["ops"].min(0)
    xmax, ymax, _ = data["ops"].max(0)
    xidx = np.arange(xmin, xmax, v1)
    xidx = np.r_[xidx, xmax]
    yidx = np.arange(ymin, ymax, v1)
    yidx = np.r_[yidx, ymax]
    weighted = direct_v(data["ops"], data["op"], yidx, xidx, v0, v1)
    Ngap_num = weighted
    op_index = np.where((realdata[:, 0] == data["op"][0]) & (
        realdata[:, 1] == data["op"][1]) & (realdata[:, 2] == data["op"][2]))[0]
    Ngap = {"op_index": op_index, "Ngap_num": Ngap_num}
    with result_lock:
        result_dict[name] = Ngap
    return Ngap


def readlas(lasfile):
    inFile = laspy.read(lasfile)
    x, y, z = inFile.x, inFile.y, inFile.z
    arr = np.c_[x, y, z]
    return arr


def opp(name, ops, op, z_v0, result_dict, result_lock):
    # set op as origin point
    # square_size = z_v0 / z_v1
    # newdata = [[]for row in range(square_size ** 2)]
    # x, y, z = op[0], op[1], op[2]
    # newdata[len(newdata) // 2].append(op)

    halfv0 = z_v0 / 2
    newdata_index = np.where(
        (ops[:, 0] < (op[0] + halfv0)) & (ops[:, 0] >= (op[0] - halfv0)) &
        (ops[:, 1] < (op[1] + halfv0)) & (ops[:, 1] >= (op[1] - halfv0)))[0]

    newdata = {"op": op, "ops": ops[newdata_index]}
    with result_lock:
        result_dict[name] = newdata
    return newdata


def start_cacu_byops(managed_dict, realdata, v0, v1):
    param_dict = []
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    for namex, datas in managed_dict.items():
        param_dict.append(['task{}'.format(namex), datas,
                          realdata, v0, v1])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results2 = [pool.apply_async(caculate_mp, args=(
        name, datas, realdata, v0, v1, managed_dict, managed_locker)) for name, datas, realdata, v0, v1 in param_dict]
    results2 = [p.get() for p in results2]
    result_item = []
    for namex, datas in managed_dict.items():
        result_item.append(datas)
    return result_item


def create_square(ops, z_v0, pool):
    param_dict = []
    for i in range(0, len(ops)):
        param_dict.append(
            ['task{}'.format(i), ops, ops[i], z_v0])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(opp, args=(
        name, ops, op, z_v0, managed_dict, managed_locker)) for name, ops, op, z_v0 in param_dict]
    results = [p.get() for p in results]
    return managed_dict


def init_data(zhigh_percent, realdata, z_v0, pool):
    filterindex = np.where(realdata[:, 2] >= realdata[:, 2].min(
    )+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]
    print(len(filter_xyz))
    manage_dict = create_square(filter_xyz, z_v0, pool)
    return manage_dict


def init_realdata(filename):
    realdata = readlas(filename)
    # x = realdata[:, 0] + realdata[:, 1] * 1j
    # np.unique(x)
    # idx = np.unique(x, return_index=True)[1]
    # realas = realdata[idx]
    return realdata
# if __name__ == "__main__":
#     fdir = os.path.dirname(os.path.abspath(__file__))
#     fname = os.path.basename(os.path.abspath(__file__))[:-3]
#     #root = os.path.dirname(fdir)
#     num_cores = int(mp.cpu_count())
#     pool = mp.Pool(num_cores)
#     filename = r"/home/jack/bck/PAR-fuck/NJ5055_DAF0_N18(1) - 6mm.las"
#     # readdata and rotate(NS)
#     realdata = readlas(filename)
#     zhigh_percent = 50  # 计算的高度层
#     # Todo
#     zmax = (max(realdata[:, 2]) - min(realdata[:, 2])) * \
#         ((100 - zhigh_percent) / 100)
#     vsize = (max(realdata[:, 0]) - min(realdata[:, 0])) * (max(realdata[:, 1]) -
#                                                            min(realdata[:, 1])) * (max(realdata[:, 2]) - min(realdata[:, 2]))
#     meanp = len(realdata) / vsize
#     # inital


#     # direct PAR,
#     v0 = 0.05      # first division, m
#     #gridesize = int(3/v0)
#     v1_range = [0.01]

#     #sec_grid = int(v0/v1)
#     # calculate direct PAR1
#     # filter data above the calculated height layer
#     filterindex = np.where(realdata[:, 2] >= realdata[:, 2].min(
#     )+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
#     filter_xyz = realdata[filterindex]
#     # print(filter_xyz[:,0].max()-filter_xyz[:,0].min())
#     # first division using v0, and calculate gap with a gridsize of v1 in each V0-space

#     for i in range(len(v1_range)):
#         v1 = v1_range[i]            # second division, m
#         # gaps = direct_v(filter_xyz, v0, v1)
#         gaps = create_mp(filter_xyz, v0, v1, zmax, meanp)
#         print('方向孔隙率：{}%'.format(round(gaps, 2)))
