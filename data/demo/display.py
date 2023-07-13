import numpy as np
import open3d as o3d

from geotransformer.utils.open3d import get_color


def merge_ply(ply_files, final_file_name):
    final_ply = o3d.geometry.PointCloud()
    final_points = np.asarray(final_ply.points)
    final_colors = np.asarray(final_ply.colors)
    for ply_file in ply_files:
        points = np.asarray(ply_file.points)
        colors = np.asarray(ply_file.colors)
        final_points = np.concatenate((final_points, points), axis=0)
        final_colors = np.concatenate((final_colors, colors), axis=0)
    final_ply.points = o3d.utility.Vector3dVector(final_points)
    final_ply.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(final_file_name, final_ply)
    return final_ply

if __name__ == '__main__':
    plyfile_1 = o3d.io.read_point_cloud("/Users/jinjoy/Desktop/output/1.ply")
    plyfile_2 = o3d.io.read_point_cloud("/Users/jinjoy/Desktop/output/2.ply")
    plyfile_3 = o3d.io.read_point_cloud("/Users/jinjoy/Desktop/output/3.ply")
    merge_ply([plyfile_1, plyfile_2, plyfile_3], "registration_result.ply")
    plyfile_1 = plyfile_1.paint_uniform_color(get_color("custom_yellow"))
    plyfile_2 = plyfile_2.paint_uniform_color(get_color("custom_blue"))
    plyfile_3 = plyfile_3.paint_uniform_color(get_color("custom_purple"))
    final_ply = merge_ply([plyfile_1, plyfile_2, plyfile_3], "for_analysis.ply")
    o3d.visualization.draw_geometries([final_ply])

    # plyfile_1 = o3d.io.read_point_cloud("1.ply")
    # plyfile_2 = o3d.io.read_point_cloud("2.ply")
    # plyfile_3 = o3d.io.read_point_cloud("3.ply")
    # final_ply = merge_ply([plyfile_1, plyfile_2, plyfile_3], "original.ply")
    # o3d.visualization.draw_geometries([final_ply])


