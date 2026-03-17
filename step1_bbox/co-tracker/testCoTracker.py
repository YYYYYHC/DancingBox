# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path_images
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
def construct_query_from_segmentations(segmentation_query_frame, frame_idx=0, sample_rate = 0.01, points3D=None):
    query = []
    obj_id_nums = {}
    points3Ds = {}
    for obj_id, mask in segmentation_query_frame.items():
        points2D = np.where(mask > 0)
        #sample the points in 2D space
        points_num = len(points2D[0])
        sample_num = int(points_num * sample_rate)
        points2D_idx = np.random.choice(points_num, sample_num, replace=False)
        points_t = np.zeros((sample_num, 3))
        points_t[:, 0] = frame_idx
        points_t[:, 1] = points2D[2][points2D_idx]
        points_t[:, 2] = points2D[1][points2D_idx]
        
        # get the corresponding point cloud
        points3D_t = points3D[points2D[1][points2D_idx], points2D[2][points2D_idx], :]
        
        query.append((points_t))
        obj_id_nums[obj_id] = sample_num
        points3Ds[obj_id] = points3D_t
    queries = torch.from_numpy(np.concatenate(query, axis=0)).to(DEFAULT_DEVICE).float()

    return queries[None], obj_id_nums, points3Ds
def load_model_and_video(args):
    # load the input video frame by frame
    video = read_video_from_path_images(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
        else:
            if args.offline:
                window_len = 60
            else:
                window_len = 16
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=args.use_v2_model,
                offline=args.offline,
                window_len=window_len,
            )
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)
    return model, video
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="/home/yhc/vggt/working_dir/images_processed",
        help="path to a video, saved in the format of images",
    )

    parser.add_argument(
        "--checkpoint",
        # default="./checkpoints/cotracker.pth",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=5, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pass it if you would like to use the offline model, in case of online don't pass it",
    )

    args = parser.parse_args()
    assert args.grid_query_frame == 0, "grid_query_frame must be 0"
    
    model, video = load_model_and_video(args)
    
    H, W = video.shape[-2], video.shape[-1]
    assert H==W, "H and W must be the same"
    
    working_dir = os.path.dirname(args.video_path)
    # load the point map /home/yhc/vggt/working_dir/point_maps
    point_map_path = os.path.join(working_dir, "point_maps", f"{args.grid_query_frame:05d}.npy")
    point_map = np.load(point_map_path, allow_pickle=True).item()
    points = point_map['world_points'][0][0] # (H, W, 3)
    # load the segmentations
    segmentations_path = os.path.join(working_dir, "video_segments.npy")
    segmentations = np.load(segmentations_path, allow_pickle=True).item()

    segmentation_query_frame = segmentations[args.grid_query_frame]
    
    # sample queries from segmentation, also get the corresponding points from point map
    queries, obj_id_nums, points3Ds = construct_query_from_segmentations(segmentation_query_frame, frame_idx=args.grid_query_frame, sample_rate=0.02, points3D=points)

    pred_tracks, pred_visibility = model(
        video,
        queries=queries,
        backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    print("computed")
    # get tracked 3d points by obj id
    acc_id = 0
    tracked_points = {}
    for obj_id in obj_id_nums.keys():
        obj_id_end = acc_id + obj_id_nums[obj_id]
        pred_tracks_obj = pred_tracks[0,:,acc_id:obj_id_end,:] # t, k, 2
        pred_visibility_obj = pred_visibility[0,:,acc_id:obj_id_end]        
        vis = Visualizer(save_dir=os.path.join(working_dir, f"vis_tracks_{obj_id}"), pad_value=120, linewidth=3)
        vis.visualize(
            video,
            pred_tracks_obj[None],
            pred_visibility_obj[None],
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
        )
        pred_tracks_obj = pred_tracks_obj.cpu().numpy()
        
        pred_visibility_obj = pred_visibility_obj.cpu().numpy()
        # 找出所有时刻都有效的点的索引
        valid_points_mask = np.all(np.all(pred_tracks_obj >= 0, axis=-1) & np.all(pred_tracks_obj < H, axis=-1), axis=0)
        pred_tracks_obj = pred_tracks_obj[:,valid_points_mask,:]
        pred_visibility_obj = pred_visibility_obj[:,valid_points_mask]
        
        tracked_points_obj = []
        for t in range(pred_tracks_obj.shape[0]):
            point_map = np.load(os.path.join(working_dir, "point_maps", f"{t:05d}.npy"), allow_pickle=True).item()
            points3D_all_t = point_map['world_points'][0][0]
            points3D_t = points3D_all_t[pred_tracks_obj[t,:,1].astype(int), pred_tracks_obj[t,:,0].astype(int), :]
            tracked_points_obj.append(points3D_t)
        tracked_points_obj = np.array(tracked_points_obj)
        tracked_points[obj_id] = tracked_points_obj
        acc_id = obj_id_end
    # save the tracked points
    np.save(os.path.join(working_dir, "tracked_points.npy"), tracked_points)
        
    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir=os.path.join(working_dir, "vis_tracks"), pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks[:, :, obj_id_nums[0]:, :],
        pred_visibility[:, :, obj_id_nums[0]:],
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
    # save the pred_tracks and pred_visibility
    tracks = {}
    tracks['queries'] = queries.cpu().numpy()
    tracks['pred_tracks'] = pred_tracks.cpu().numpy()
    tracks['pred_visibility'] = pred_visibility.cpu().numpy()
    tracks['obj_id_nums'] = obj_id_nums
    
    np.save(os.path.join(working_dir, "tracks.npy"), tracks)
