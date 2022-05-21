import numpy as np
from Algorithm.param_utils import nxParameter_ball, atan2
import math
import multiprocessing as mp


class LodaData_Ball(object):
    def __init__(self, theta_grid, phi_grid, r_percent=0, r_grid=0, gridsize=0, zhigh=0, lasfile="", car2ball="t"):
        self.theta_grid = theta_grid
        self.phi_grid = phi_grid
        self.r_percent = r_percent
        self.r_grid = r_grid
        self.car2balldata = car2ball
        self.gridsize = gridsize
        self.zhigh = zhigh
        self.lasfile = lasfile

    def getParamsR(self):
        params_init = object
        if isinstance(self.car2balldata, str):
            balldata = self.car2ball()
        else:
            balldata = self.car2balldata
        sfilter_data = balldata
        params_init = nxParameter_ball(
            sfilter_data[:, 1], sfilter_data[:, 2], self.theta_grid, self.phi_grid, data_r=sfilter_data[:, 0], r_grid=self.r_grid)
        self.car2balldata = sfilter_data
        return params_init

    def car2ball(self):
        newdata_x = np.asarray(self.lasfile[:, 0])
        newdata_y = np.asarray(self.lasfile[:, 1])
        newdata_z = np.asarray(self.lasfile[:, 2])
        phi = list(map(lambda x, y: wd2(x, y), newdata_x, newdata_y))
        r = list(map(lambda x, y, z: math.sqrt(x**2 + y**2 + z**2),
                     newdata_x, newdata_y, newdata_z))
        theta = list(map(lambda z, r: wdw(z, r), newdata_z, r))
        print("最大phi值：{}".format(max(phi)), "最大theta值:{}".format(max(theta)))
        ball = np.column_stack([r, theta, phi]).astype(np.float32)
        self.car2balldata = ball
        return ball


def wd2(x, y):
    re = atan2(x, y) * 180
    while re - 360 > 0:
        re = re - 360
    return re


def wdw(z, r):
    re = math.acos((z / r)) * 180
    while re - 90 > 0:
        re = re - 90
    return re


def caculates_ball_t(name, newlist, params_init, result_dict, result_lock, mint, minp, minr, arrange_t=0):
    arrange_p = 0
    arrange_r = 0
    alldata = np.c_[params_init.data_x, params_init.data_y, params_init.data_r]
    for i in range(0, len(newlist)):
        tdata_e = mint + params_init.arr_t * arrange_t
        tdata = mint + params_init.arr_t * (arrange_t + 1)
        pdata_e = minp + params_init.arr_p * arrange_p
        pdata = minp + params_init.arr_p * (arrange_p + 1)
        rdata_e = minr + params_init.arr_r * arrange_r
        rdata = minr + params_init.arr_r * (arrange_r + 1)
        condition_data_index = np.where((params_init.data_x <= tdata) & (params_init.data_x >= tdata_e) & (
            params_init.data_y <= pdata) & (params_init.data_y >= pdata_e) & (params_init.data_r <= rdata) & (params_init.data_r >= rdata_e))
        condition_data = alldata[condition_data_index]
        for j in condition_data:
            newlist[i].append(j)
        arrange_p += 1
        if (i + 1) % params_init.phi_grid == 0:
            arrange_p = 0
            arrange_r += 1
        if (i + 1) % (params_init.r_grid * params_init.phi_grid) == 0:
            arrange_p = 0
            arrange_r = 0
        # print(tdata, pdata, rdata)
    with result_lock:
        result_dict[name] = newlist
    return newlist


def doball(zhigh_percent, r_grid, newload, pool):
    Ngap_num = 0
    load_datas_ball = newload

    params_init = load_datas_ball.getParamsR()
    param_dict = []
    mint = np.asarray(params_init.data_x).min()
    minp = np.asarray(params_init.data_y).min()
    minr = np.asarray(params_init.data_r).min()
    for i in range(0, params_init.theta_grid):
        param_dict.append(
            ['task{}'.format(i), [[]for row in range(params_init.phi_grid * params_init.r_grid)], params_init, mint, minp, minr, i])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(caculates_ball_t, args=(
        name, nlist, param, managed_dict, managed_locker, mint, minp, minr, arrang_t)) for name, nlist, param, mint, minp, minr, arrang_t in param_dict]
    results = [p.get() for p in results]
    for namex, datas in managed_dict.items():
        Ngap_num = Ngap_num + datas.count([])
    Ngap_num = Ngap_num / \
        ((params_init.theta_grid * params_init.phi_grid *
          params_init.r_grid))
    print("{}%层 v2:{} r_grid:{} 角度孔隙率：{} %".format(
        zhigh_percent, 90 / params_init.theta_grid, r_grid, round(Ngap_num * 100, 2)))
