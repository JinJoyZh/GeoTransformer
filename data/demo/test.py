import numpy as np
import plyfile
import open3d as o3d


def read_ply_coors(path):
    plydata = plyfile.PlyData.read(path)
    # read coordinate data as numpy
    elements_data_0 = plydata.elements[0].data
    coor_x_array = elements_data_0['x']
    coor_y_array = elements_data_0['y']
    coor_z_array = elements_data_0['z']
    coor_array = np.vstack((coor_x_array, coor_y_array, coor_z_array))
    coor_array = np.asarray(coor_array.T)
    return coor_array

def read_npy(path):
    a = np.load(path)
    print(a.shape)
    return a

def read_ply_o3(path):
    print("Testing IO for meshes ...")
    mesh = o3d.io.read_point_cloud(path)
    points = np.asarray(mesh.points)
    colors = np.asarray(mesh.colors)
    return points, colors

def create_ply(ply_path):
    points, colors = read_ply_o3(ply_path)
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points)
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply

def merge_ply(ply_1, ply_2):
    points_1 = np.asarray(ply_1.points)
    colors_1 = np.asarray(ply_1.colors)
    points_2 = np.asarray(ply_2.points)
    colors_2 = np.asarray(ply_2.colors)
    new_points = np.concatenate((points_1, points_2), axis=0)
    new_colors = np.concatenate((colors_1, colors_2), axis=0)
    new_ply = o3d.geometry.PointCloud()
    new_ply.points = o3d.utility.Vector3dVector(new_points)
    new_ply.colors = o3d.utility.Vector3dVector(new_colors)
    o3d.io.write_point_cloud("merge_test.ply", new_ply)
    return new_ply

def read_points(path):
    plydata = o3d.io.read_point_cloud(path)
    plydata = plydata.voxel_down_sample(voxel_size=0.1)
    points = np.asarray(plydata.points)
    print(points.shape)
    colors = np.asarray(plydata.colors)
    return points, colors
def recordPoints(src, ref):
    src_points, src_colors = read_points(src)
    np.save('src_points.npy', src_points)
    np.save('src_colors.npy', src_colors)
    ref_points, ref_colors = read_points(ref)
    np.save('ref_points.npy', ref_points)
    np.save('ref_colors.npy', ref_colors)


if __name__ == '__main__':
    # path = '/Users/jinjoy/resource/3DRestructiom/dataset/inner/fused/fused1.ply'
    # ply = o3d.io.read_point_cloud(path)
    # ply = ply.voxel_down_sample(voxel_size=0.07)
    # print(ply)
    # o3d.visualization.draw_geometries([ply])

    # src = '/Users/jinjoy/resource/3DRestructiom/dataset/inner/fused/fused1.ply'
    # ref = '/Users/jinjoy/resource/3DRestructiom/dataset/inner/fused/fused2.ply'
    # recordPoints(src, ref)

    a = read_npy("src_points.npy")
    b = read_npy("ref.npy")
    print(a)
    print(b)

    # ply_path_1 = 'cloud_1.ply'
    # ply_1 = create_ply(ply_path_1)
    # points_1 = np.asarray(ply_1.points)
    # print("start")
    # print(points_1.shape)
    # ply_path_2 = 'cloud_2.ply'
    # ply_2 = create_ply(ply_path_2)
    # ply_3 = merge_ply(ply_1, ply_2)
    # o3d.visualization.draw_geometries([ply_1])

    # read_npy('ref.npy')