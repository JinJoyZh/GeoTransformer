import argparse
import datetime

import numpy as np
import open3d as o3d
import torch

from config import make_cfg
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color
from geotransformer.utils.torch import to_cuda, release_cuda
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src ply file path")
    parser.add_argument("--ref_file", required=True, help="ref ply file path")
    parser.add_argument("--weights", required=True, help="model weights file path")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    return parser


def read_points(path):
    plydata = o3d.io.read_point_cloud(path)
    plydata = plydata.voxel_down_sample(voxel_size=0.07)
    points = np.asarray(plydata.points)
    colors = np.asarray(plydata.colors)
    return points, colors

def load_data(args):
    src_points, src_colors = read_points(args.src_file)
    ref_points, ref_colors = read_points(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "ref_colors": ref_colors,
        "src_points": src_points.astype(np.float32),
        "src_colors": src_colors,
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),

    }
    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)
    return data_dict

def merge_ply(ply_1, ply_2, file_name):
    points_1 = np.asarray(ply_1.points)
    colors_1 = np.asarray(ply_1.colors)
    points_2 = np.asarray(ply_2.points)
    colors_2 = np.asarray(ply_2.colors)
    new_points = np.concatenate((points_1, points_2), axis=0)
    new_colors = np.concatenate((colors_1, colors_2), axis=0)
    new_ply = o3d.geometry.PointCloud()
    new_ply.points = new_points
    new_ply.colors = new_colors
    o3d.io.write_point_cloud(file_name, new_ply)
    return new_ply

def main():
    start_time = datetime.datetime.now()
    parser = make_parser()
    args = parser.parse_args()
    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    #print prediction results
    print("prediction done! estimated_transform: ")
    print(estimated_transform)
    print("corr shapes")
    print(output_dict['ref_node_corr_knn_points'].shape)
    print(output_dict['src_node_corr_knn_points'].shape)
    end_time = datetime.datetime.now()
    time_consumed = (end_time - start_time).seconds
    print("time cost: ")
    print(time_consumed)

    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    src_pcd = src_pcd.transform(estimated_transform)
    merge_ply(src_pcd, ref_pcd, "for_comparison.ply")
    ref_pcd.colors = o3d.utility.Vector3dVector(data_dict["ref_colors"])
    src_pcd.colors = o3d.utility.Vector3dVector(data_dict["src_colors"])
    merge_ply(src_pcd, ref_pcd, "for_result.ply")
    # compute time cost
    end_time = datetime.datetime.now()
    time_consumed = (end_time - start_time).seconds
    print("all done! Time consumed: ")
    print(time_consumed)


if __name__ == "__main__":
    main()
