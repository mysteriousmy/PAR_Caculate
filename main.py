from Algorithm import *


def data_init(zhigh_percent):
    read_data = utils.lasfileAction().getLasfile()
    load_data = direct.LoadData(zhigh=zhigh_percent, lasfile=read_data)
    params_init = load_data.getParams()
    return load_data, params_init


if __name__ == "__main__":
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心，正在加载计算核心和数据..")
    pool = mp.Pool(num_cores)

    zhigh_percent = 75  # 计算的高度层
    v0 = [0.05, 0.1, 0.04]               # first division, m
    v1 = [0.009, 0.01, 0.008]                  # second division, m

    newload, params_init = data_init(zhigh_percent)

    for i, j in zip(v0, v1):
        params_init.gridsize = int(3/i)
        params_init.secgrid = int(i/j)
        params_init.newarr()
        direct.docaculate(params_init, zhigh_percent, pool)

    v2 = [0.3, 2, 0.5]  # degree
    r_percent = 100         # percent from cuurent position to top of canopy
    r_grid = [1, 2, 2]            # 2 blocks

    # 散射孔隙率，后两个为球坐标系垂直（90°）和水平（360°）的分割块数
    for i, j in zip(v2, r_grid):
        theta_grid = int(90/i)
        phi_grid = int(360/i)
        scatter.doball(zhigh_percent, r_percent, j,
                       theta_grid, phi_grid, newload, pool)
