import datetime
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


if __name__ == "__main__":
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    #root = os.path.dirname(fdir)
    start_t = datetime.datetime.now()
    filename = r"/home/jack/bck/PAR-fuck/NJ5055_DAF0_N18(1) - 6mm.las"
    # readdata and rotate(NS)
    realdata = readlas(filename)
    # inital
    zhigh_percent = 75  # 计算的高度层
    # diffuse PAR
    v2 = 0.6  # degree
    #theta_grid = int(90/v2)
    #phi_grid = int(360/v2)
    # filter data above the calculated height layer
    filterindex = np.where(realdata[:, 2] >=
                           realdata[:, 2].min()+(realdata[:, 2].max()-realdata[:, 2].min())*zhigh_percent/100)[0]
    filter_xyz = realdata[filterindex]
    x, y, z = filter_xyz[:, 0], filter_xyz[:, 1], filter_xyz[:, 2]
    ops_index = np.where(filter_xyz[:, 2] == filter_xyz[:, 2].min())
    ops = filter_xyz[ops_index]
    # print(len(ops))
    # for opp in ops:
    #     print(opp)
    tmin, pmin = 0, 0
    tmax, pmax = 90, 360
    tidx = np.arange(tmin, tmax, v2)
    tidx = np.r_[tidx, tmax]
    pidx = np.arange(pmin, pmax, v2)
    pidx = np.r_[pidx, pmax]
    ops_x_min, ops_x_max = ops[:, 0].min(), ops[:, 0].max()
    ops_y_min, ops_y_max = ops[:, 1].min(), ops[:, 1].max()
    ops = list(
        filter(lambda x: (x[0] >= ((ops_x_max-ops_x_min) * 0.25) + ops_x_min) and (x[0] <= ((ops_x_max-ops_x_min) * 0.75) + ops_x_min) and (x[1] >= ((ops_y_max-ops_y_min) * 0.25) + ops_y_min) and (x[1] <= ((ops_y_max-ops_y_min) * 0.75) + ops_y_min), ops))
    nn = 0
    print(len(ops))
    for op in ops:
        Ngap_num = 0
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
        newlist = diffuse_v(ballxyz, v2)
        for i in newlist:
            if(len(i) == 0):
                Ngap_num = Ngap_num + 1
        nn = nn + (Ngap_num / ((len(tidx)-1) * (len(pidx)-1)))
    print(nn / len(ops))
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
