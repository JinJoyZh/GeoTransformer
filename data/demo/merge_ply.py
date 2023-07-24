import os
import argparse
import numpy as np
import open3d as o3d

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="point cloud files that to be merged")
    parser.add_argument("-o", required=True, help="the path of output file")
    return parser

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
    parser = make_parser()
    args = parser.parse_args()
    plyfiles = []
    for root, dirs, files in os.walk(args.i, topdown=False):
        for name in files:
            if(name.endswith('.ply')):
                file_path = os.path.join(root, name)
                plyfile = o3d.io.read_point_cloud(file_path)
                plyfiles.append(plyfile)
    final_ply = merge_ply(plyfiles, args.o)
    o3d.visualization.draw_geometries([final_ply])


