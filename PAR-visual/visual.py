from cmath import log
import datetime
import os
import multiprocessing as mp
import open3d as o3d
import AGF_demo_old2 as agf
import DGF_demo_fuck as dgf


# 传入pari数据进去判断染色
def open3dDraw(filename, PARi_list):
    lasdata_xyz = dgf.readlas(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lasdata_xyz)
    # 先展示全灰色
    o3d.geometry.KDTreeFlann(pcd)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])
    # 粗略染色
    for i in PARi_list:
        r, g, b = 0, 0, 0
        if i["PARi"] <= 600 and i["PARi"] > 400:
            tmp = i["PARi"] / 500
            r = tmp
        if i["PARi"] <= 400 and i["PARi"] > 200:
            tmp_g = i["PARi"] / 400
            r = tmp_g
            g = 1-tmp_g
            b = tmp_g

        if i["PARi"] >= 0 and i["PARi"] <= 200:
            tmp_g = i["PARi"] / 200
            r = 1 - tmp_g
            g = tmp_g
        pcd.colors[i["index"]] = [r, g, b]
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(r"D:\bck\PAR-visual\output.pcd", pcd)

# 计算出PARi组


def caculate_PARi(agf_result, dgf_result):
    PAR_djf0 = 1000
    PAR_dir0 = 2000
    sin_thetas = 0.8
    PARi_list = []
    for agf_ngap, dgf_ngap in zip(agf_result, dgf_result):
        PARi_list.append({"index": dgf_ngap["op_index"][0], "PARi": (
            PAR_djf0 * float(agf_ngap["Ngap_num"]) + PAR_dir0 * float(dgf_ngap["Ngap_num"]) * sin_thetas)})
    print(PARi_list)
    return PARi_list

#


def caculate_PARi_dgf(dgf_result):
    PAR_djf0 = 1000
    sin_thetas = 0.5
    PARi_list = []
    for dgf_ngap in dgf_result:
        print(int(PAR_djf0 * float(dgf_ngap["Ngap_num"]) * sin_thetas))
        PARi_list.append({"index": dgf_ngap["op_index"][0], "PARi": (
            PAR_djf0 * float(dgf_ngap["Ngap_num"]) * sin_thetas)})
    return PARi_list


if __name__ == "__main__":
    start_t = datetime.datetime.now()

    fdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(os.path.abspath(__file__))[:-3]
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)

    filename = r"D:\bck\PAR-visual\18mm - Cloud.las"

    zhigh_percent = 1
    # 直射
    dgf_v0 = 0.06  # 每个正方方格的长度
    dgf_v1 = 0.015
    # 初始化切割正方形数据等
    realdatas = dgf.init_realdata(filename)
    dgf_data_dict = dgf.init_data(
        zhigh_percent, realdatas, dgf_v0, pool)

    # 计算出所有点的孔隙率结果集
    dgf_result = dgf.start_cacu_byops(
        dgf_data_dict, realdatas, dgf_v0, dgf_v1)

    # # 散射
    # agf_v2 = [1.8]
    # # 初始化每个点形成的数据级
    # agf_data_dict = agf.init_data(zhigh_percent, realdatas, pool)
    # # 计算散射的孔隙率结果集
    # agf_result = agf.init_caculate(agf_v2, realdatas, agf_data_dict)
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多线程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    # # 计算PARi结果集
    PARi_result = caculate_PARi_dgf(dgf_result)
    # 进行染色
    open3dDraw(filename, PARi_result)
