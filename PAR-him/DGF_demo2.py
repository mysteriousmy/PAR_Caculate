import laspy
import numpy as np
import os
from numpy.lib.function_base import append, extract, select


def gap(_allProj, xmin, xmax, ymin, ymax, voxelsize):
    # 1* 2**n / 4000. = v1
    N = int(voxelsize*200000)
    numberx = (xmax - xmin)/(1*N/200000.)
    numbery = (ymax - ymin)/(1*N/200000.)
    numberxy = numberx * numbery
    _allProj4000 = (_allProj*200000).astype('i4')
    # >>5        1*2**5 /200000. = 0.00016 m �ķֱ���
    _allProj4000_32 = _allProj4000 // N
    _allProj4000_32 = _allProj4000_32.view('S8').flatten()
    unique = np.unique(_allProj4000_32).view('2i4')
    roinumber = len(unique)
    canopyCover = round(float(roinumber/numberxy), 3)
    return 1-canopyCover


def direct_v(data, v, v1):
    xmin, ymin, _ = data.min(0)
    xmax, ymax, _ = data.max(0)
    xidx = np.arange(xmin, xmax, v)
    xidx = np.r_[xidx, xmax]
    yidx = np.arange(ymin, ymax, v)
    yidx = np.r_[yidx, ymax]
    newlist = []
    for y in range(len(yidx[:-1])):
        dymin, dymax = yidx[y], yidx[y+1]
        for x in range(len(xidx[:-1])):
            dxmin, dxmax = xidx[x], xidx[x+1]
            if y == len(yidx[:-1])-1:
                condition_data = (data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] <= dymax)
            elif x == len(xidx[:-1])-1:
                condition_data = (data[:, 0] >= dxmin) & (data[:, 0] <= dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax)
            else:
                condition_data = (data[:, 0] >= dxmin) & (data[:, 0] < dxmax) & (
                    data[:, 1] >= dymin) & (data[:, 1] < dymax)

            g = gap(data[condition_data][:, :2],
                    dxmin, dxmax, dymin, dymax, v1)

            newlist.append(g)
    newlist = np.asarray(newlist)
    return newlist


def readlas(lasfile):
    inFile = laspy.read(lasfile)
    x, y, z = inFile.x, inFile.y, inFile.z
    arr = np.c_[x, y, z]
    return arr


if __name__ == "__main__":
    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    #root = os.path.dirname(fdir)

    filename = r"/home/jack/bck/PAR-fuck/NJ5055_DAF0_N18(1) - 6mm.las"
    # readdata and rotate(NS)
    realdata = readlas(filename)
    # Todo

    # inital
    zhigh_percent = 75  # 计算的高度层
    # direct PAR,
    v0 = 0.15        # first division, m
    #gridesize = int(3/v0)
    v1_range = [0.006, 0.008, 0.01, 0.012, 0.015]
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
        gaps = direct_v(filter_xyz, v0, v1)
        print('方向孔隙率：{}%'.format(round(gaps.mean()*100, 2)))
