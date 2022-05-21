import open3d as o3d
pcd = o3d.io.read_point_cloud(r"D:\bck\PAR-visual\output.pcd")
o3d.visualization.draw_geometries([pcd])
