import numpy as np
from Algorithm.param_utils import nxParameter_xyz
import multiprocessing as mp


class LoadData(object):
    def __init__(self, gridsize=1, secgrid=0, zhigh=0, lasfile=""):
        self.lasfile = lasfile
        self.lasfile_t = self.setZero()
        self.zhigh = round(max(self.lasfile_t[:, 2]) * float(zhigh / 100), 2)
        self.gridsize = gridsize
        self.secgrid = secgrid
        self.filepath = ""

    def getParams(self):
        sfilter_data = []
        newdata = self.lasfile_t
        sdata = np.where(newdata[:, 2] >= self.zhigh)[0]
        sfilter_data = newdata[sdata]
        params_init = nxParameter_xyz(
            sfilter_data[:, 0], sfilter_data[:, 1], gridsize=self.gridsize, secgrid=self.secgrid)
        self.lasfile_t = sfilter_data
        return params_init

    def setZero(self):
        newdata_x = self.lasfile.x - np.asarray(self.lasfile.x).mean()
        newdata_y = self.lasfile.y - np.asarray(self.lasfile.y).mean()
        newdata_z = self.lasfile.z - np.asarray(self.lasfile.z).min()
        real_data = np.column_stack(
            [newdata_x, newdata_y, newdata_z]).astype(np.float32)
        self.lasfile_t = real_data
        print("加载完毕，开始计算")
        return real_data


def caculates(newlist, params_init, minx=0, miny=0):
    arrange_y = 0
    arrange_x = 0
    alldata = np.column_stack(
        [params_init.data_x, params_init.data_y]).astype(np.float32)
    for i in range(0, len(newlist)):
        xdata_e = minx + params_init.arr * arrange_x
        xdata = minx + params_init.arr * (arrange_x + 1)
        ydata_e = miny + params_init.arr_y * arrange_y
        ydata = miny + params_init.arr_y * (arrange_y + 1)
        condition_data_index = np.where((params_init.data_x <= xdata) & (params_init.data_x >= xdata_e) & (
            params_init.data_y <= ydata) & (params_init.data_y >= ydata_e))[0]
        condition_data = alldata[condition_data_index]
        for j in condition_data:
            newlist[i].append(j)
        arrange_x += 1
        if (i + 1) % params_init.gridsize == 0:
            arrange_y += 1
            arrange_x = 0
    return newlist


def caculates_t(name, newlist, params_init, result_dict, result_lock, minx=0, miny=0, arrange_y=0):
    arrange_x = 0
    alldata = np.column_stack(
        [params_init.data_x, params_init.data_y]).astype(np.float32)
    newlists = []
    Ngap = 0
    for i in range(0, len(newlist)):
        xdata_e = minx + params_init.arr * arrange_x
        xdata = minx + params_init.arr * (arrange_x + 1)
        ydata_e = miny + params_init.arr_y * arrange_y
        ydata = miny + params_init.arr_y * (arrange_y + 1)
        condition_data_index = np.where((params_init.data_x <= xdata) & (params_init.data_x >= xdata_e) & (
            params_init.data_y <= ydata) & (params_init.data_y >= ydata_e))
        condition_data = alldata[condition_data_index]
        newlists = [[]for row in range(params_init.secgrid ** 2)]
        if len(condition_data) != 0:
            show_cell_x = list(map(lambda x: x[0], condition_data))
            show_cell_y = list(map(lambda y: y[1], condition_data))
            params_test = nxParameter_xyz(
                show_cell_x, show_cell_y, gridsize=params_init.secgrid)
            newlistw = caculates(newlists, params_test,
                                 min(show_cell_x), min(show_cell_y))
            Ngap = Ngap + \
                round((newlistw.count([]) / (params_test.gridsize ** 2)) * 100, 2)
        else:
            Ngap = Ngap + 100
        arrange_x += 1

    with result_lock:
        result_dict[name] = Ngap
    return Ngap


def docaculate(params_init, zhigh_percent, pool):
    Ngap_num = 0
    # params_init, sfilter_data = load_datas.getParams()
    param_dict = []
    for i in range(0, params_init.gridsize):
        param_dict.append(
            ['task{}'.format(i), [[]for row in range(params_init.gridsize)], params_init, min(params_init.data_x), min(params_init.data_y), i])
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(caculates_t, args=(
        name, nlist, param, managed_dict, managed_locker, minx, miny, arrang_y)) for name, nlist, param, minx, miny, arrang_y in param_dict]
    results = [p.get() for p in results]
    for namex, datas in managed_dict.items():
        Ngap_num = Ngap_num + datas
    Kx = round(Ngap_num / (params_init.gridsize ** 2), 2)
    print("{}%层 gridsize:{} sec_grid:{} 直射孔隙率：{} %".format(
        zhigh_percent, params_init.gridsize, params_init.secgrid, Kx))
