from hashlib import new
import laspy
import numpy as np
import os
import multiprocessing as mp


def direct_v_two(data, v0, v1, dymax, dxmax, dymin, dxmin, flag):
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
    weighted = calculate_weighted(newlist, v0, v1, flag)
    # weighted = 1 - (weighted / ((v0 / v1) ** 2)) 0.01 25: 8.9% 50: 16.9% 75:53.2%
    # 0.01 25: 21.9% 50: 29.32% 75:59.89%
    weighted = (weighted / (len(xidx[:-1]) * len(yidx[:-1])))
    return weighted


def direct_v(name, data, yidx, xidx, v0, v1, y, flag, result_dict, result_lock):
    weighted = 0
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
        print(len(condition_data))
        if len(condition_data) != 0:
            weighted = weighted + \
                direct_v_two(condition_data, v0, v1, dymax,
                             dxmax, dymin, dxmin, flag)
        else:
            weighted = weighted + 1
    with result_lock:
        result_dict[name] = weighted / len(xidx[:-1])
    return weighted / len(xidx[:-1])


def calculate_weighted(data, v0, v1, flag):
    D = 0
    deltai = 0
    sizelist = []
    mediandata = []
    sizelist = list(map(lambda x: len(x), data))
    if flag == 1:
        D = max(sizelist)
    elif flag == 0:
        D = sum(sizelist) / len(data)
    mediandata = get_list_median(get_list_median(data))
    for j in data:
        wdi = len(j) / D
        e0 = mediandata[0] + mediandata[1]
        eidata = get_list_median(j)
        ei = mediandata[1] + mediandata[0] - (eidata[0] + eidata[1])
        wei = 1 - (ei/e0) + ((2 * v1) / v0)
        deltai += (1 * wdi * wei)
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


def create_mp(data, v, v1, w_caculate_flag):
    xmin, ymin, _ = data.min(0)
    xmax, ymax, _ = data.max(0)
    xidx = np.arange(xmin, xmax, v)
    xidx = np.r_[xidx, xmax]
    yidx = np.arange(ymin, ymax, v)
    yidx = np.r_[yidx, ymax]
    param_dict = []
    y = len(yidx[:-1])
    for i in range(0, y):
        param_dict.append(['task{}'.format(i), data, yidx,
                          xidx, v, v1, i, w_caculate_flag])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results2 = [pool.apply_async(direct_v, args=(
        name, data, yidx, xidx, v, v1, y, w_caculate_flag, managed_dict, managed_locker))
        for name, data, yidx, xidx, v, v1, y, w_caculate_flag in param_dict]
    results2 = [p.get() for p in results2]
    Ngap_num = 0
    for namex, datas in managed_dict.items():
        Ngap_num = Ngap_num + datas
    Ngap = Ngap_num / y * 100
    return Ngap


def readlas(lasfile):
    inFile = laspy.read(lasfile)
    x, y, z = inFile.x, inFile.y, inFile.z
    arr = np.c_[x, y, z]
    return arr


if __name__ == "__main__":
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    #root = os.path.dirname(fdir)
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    filename = r"D:\bck\PAR-visual\NJ5055_DAF0_N18(1) - 18mm.las"
    # readdata and rotate(NS)
    realdata = readlas(filename)
    # Todo

    # inital
    zhigh_percent = 75  # 计算的高度层
    # direct PAR,
    v0 = 0.05      # first division, m
    #gridesize = int(3/v0)
    v1_range = [0.01]

    w_caculate_flag = 0  # 权计算D取最大值还是平均数的标识，0代表取平均数 1是最大值
    #sec_grid = int(v0/v1)

    # calculate direct PAR
    # filter data above the calculated height layer
    filterindex = np.where(realdata[:, 2] >= realdata[:, 2].min(
    )+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]
    # print(filter_xyz[:,0].max()-filter_xyz[:,0].min())
    # first division using v0, and calculate gap with a gridsize of v1 in each V0-space

    for i in range(len(v1_range)):
        v1 = v1_range[i]            # second division, m
        # gaps = direct_v(filter_xyz, v0, v1)
        gaps = create_mp(filter_xyz, v0, v1, w_caculate_flag)
        print('方向孔隙率：{}%'.format(round(gaps, 2)))
