import laspy
import numpy as np

import os
import math

# def caulate():
#


def diffuse_v(data, v):
    tmin, pmin = 0, 0
    tmax, pmax = 90, 360
    tidx = np.arange(tmin, tmax, v)
    tidx = np.r_[tidx, tmax]
    pidx = np.arange(pmin, pmax, v)
    pidx = np.r_[pidx, pmax]

    for t in range(len(tidx[:-1])):
        dtmin, dtmax = tidx[t], tidx[t+1]
        for p in range(len(pidx[:-1])):
            dpmin, dpmax = pidx[p], pidx[p+1]
            if t == len(tidx[:-1])-1:
                condition_data = (data[:, 0] >= dpmin) & (data[:, 0] < dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] <= dtmax)
            elif p == len(pidx[:-1])-1:
                condition_data = (data[:, 0] >= dpmin) & (data[:, 0] <= dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] < dtmax)
            else:
                condition_data = (data[:, 0] >= dpmin) & (data[:, 0] < dpmax) & (
                    data[:, 1] >= dtmin) & (data[:, 1] < dtmax)


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


if __name__ == "__main__":
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    #root = os.path.dirname(fdir)

    filename = r"/home/jack/bck/PAR-fuck/NJ5055_DAF0_N18(1) - 9mm.las"
    # readdata and rotate(NS)
    realdata = readlas(filename)
    # inital
    zhigh_percent = 75  # 计算的高度层
    # diffuse PAR
    v2 = 1  # degree
    #theta_grid = int(90/v2)
    #phi_grid = int(360/v2)
    # filter data above the calculated height layer
    filterindex = np.where(realdata[:, 2] >=
                           realdata[:, 2].min()+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]
    x, y, z = filter_xyz[:, 0], filter_xyz[:, 1], filter_xyz[:, 2]
    ops_index = np.where(filter_xyz[:, 2] == filter_xyz[:, 2].min())
    ops = filter_xyz[ops_index]
    print(len(ops))
    for op in ops:
        # set op as origin point
        new_op = op - op
        newdata_x, newdata_y, newdata_z = x-op[0], y-op[1], z-op[2]
        newdata = np.c_[newdata_x, newdata_y, newdata_z]
        # descartes to polar coordinates
        phi = list(map(lambda x, y: atan2(x, y)*180 /
                   math.pi, newdata[:, 0], newdata[:, 1]))
        r = list(map(lambda x, y, z: math.sqrt(x**2 + y**2 + z**2),
                 newdata[:, 0], newdata[:, 1], newdata[:, 2]))
        theta = list(map(lambda z, r: math.acos(
            z/r)*180/math.pi, newdata[:, 2], r))

        ballxyz = np.c_[phi, theta, r].astype(np.float32)

        diffuse_v(ballxyz, v2)
