import laspy
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import multiprocessing as mp


def diffuse(name, newlist, halfball, v2, r_grid, result_dict, result_lock, arrange_t=0):
    minp = min(halfball[:, 0])
    mint = min(halfball[:, 1])
    minr = min(halfball[:, 2])
    arr_r = max(halfball[:, 2]) / r_grid
    arrange_p = 0
    arrange_r = 0
    for i in range(0, len(newlist)):
        tdata_e = mint + v2 * arrange_t
        tdata = mint + v2 * (arrange_t + 1)
        pdata_e = minp + v2 * arrange_p
        pdata = minp + v2 * (arrange_p + 1)
        rdata_e = minr + arr_r * arrange_r
        rdata = minr + arr_r * (arrange_r + 1)
        condition_data_index = np.where((halfball[:, 1] <= tdata) & (halfball[:, 1] >= tdata_e) & (
            halfball[:, 0] <= pdata) & (halfball[:, 0] >= pdata_e) & (halfball[:, 2] <= rdata) & (halfball[:, 2] >= rdata_e))
        condition_data = halfball[condition_data_index]
        for j in condition_data:
            newlist[i].append(j)
        arrange_p += 1
        if (i + 1) % int(360 / v2) == 0:
            arrange_p = 0
            arrange_r += 1
        if (i + 1) % (r_grid * int(360 / v2)) == 0:
            arrange_p = 0
            arrange_r = 0
        # print(tdata, pdata, rdata)
    with result_lock:
        result_dict[name] = newlist
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


def doball(zhigh_percent, halfball, r_grid, v2, pool):
    param_dict = []
    Ngap_num = 0
    theta_grid = int(90 / v2)
    phi_grid = int(360 / v2)
    for i in range(0, theta_grid):
        param_dict.append(
            ['task{}'.format(i), [[]for row in range(phi_grid * r_grid)], halfball, v2, r_grid, i])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(diffuse, args=(
        name, nlist, halfball, v2, rgrid, managed_dict, managed_locker, arrang_t)) for name, nlist, halfball, v2, rgrid, arrang_t in param_dict]
    results = [p.get() for p in results]
    for namex, datas in managed_dict.items():
        Ngap_num = Ngap_num + datas.count([])
    Ngap_num = Ngap_num / \
        ((theta_grid * phi_grid *
          r_grid))
    print("{}%层 v2:{} r_grid:{} 角度孔隙率：{} %".format(
        zhigh_percent, 90 / theta_grid, r_grid, round(Ngap_num * 100, 2)))


if __name__ == "__main__":
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    #root = os.path.dirname(fdir)
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心，正在加载计算核心和数据..")
    pool = mp.Pool(num_cores)

    filename = r"/home/jack/bck/NJ5055_DAF10_N18(1) - 6mm.las"
    # readdata and rotate(NS)
    realdata = readlas(filename)
    # Todo
    # inital
    zhigh_percent = 75  # 计算的高度层

    # diffuse PAR
    v2 = 1  # degree
   # theta_grid = int(90/v2)
    #phi_grid = int(360/v2)
    r_grid = 1             # 2 blocks

    # calculate direct PAR
    # filter data above the calculated height layer
    filterindex = np.where(realdata[:, 2] >= realdata[:, 2].min(
    )+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]

    # first division using v0, and calculate gap with a gridsize of v1 in each V0-space
    x, y, z = filter_xyz[:, 0], filter_xyz[:, 1], filter_xyz[:, 2]
    #ops_index = np.where(filter_xyz[:,2]==filter_xyz[:,2].min())
    #ops = filter_xyz[ops_index]

    op_x, op_y, op_z = x.mean(), y.mean(), z.min()
    op = np.column_stack([op_x, op_y, op_z])

    rmax = z.max() - op_z
    # set op as origin point
    newdata_x, newdata_y, newdata_z = x-op_x, y-op_y, z-op_z
    new_op = op - op
    newdata = np.c_[newdata_x, newdata_y, newdata_z]

    # filter a  halfball
    distance = list(map(lambda x, y, z: math.sqrt(
        x**2 + y**2 + z**2), newdata_x, newdata_y, newdata_z))
    rindex = np.where(distance < rmax)
    ballxyz = newdata[rindex]
    # descartes to polar coordinates
    phi = list(map(lambda x, y: atan2(x, y)*180 /
               math.pi, ballxyz[:, 0], ballxyz[:, 1]))
    r = list(map(lambda x, y, z: math.sqrt(x**2 + y**2 + z**2),
             ballxyz[:, 0], ballxyz[:, 1], ballxyz[:, 2]))
    theta = list(map(lambda z, r: math.acos(
        z/r)*180/math.pi, ballxyz[:, 2], r))

    halfball = np.c_[phi, theta, r].astype(np.float32)

    doball(zhigh_percent, halfball, r_grid, v2, pool)
