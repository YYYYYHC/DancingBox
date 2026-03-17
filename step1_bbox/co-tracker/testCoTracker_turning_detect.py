# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path_images
from cotracker.predictor import CoTrackerPredictor
VIS_THRESHOLD = 0.5
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def plot_visible_point_count(visibility, save_path, title):
    """
    Plot and save the number of visible points per frame.

    Args:
        visibility: numpy array of shape (T, K) with boolean visibility values
        save_path: path to save the plot
        title: title for the plot
    """
    visible_counts = np.sum(visibility, axis=1)  # (T,) - visibility is already bool
    frames = np.arange(len(visible_counts))

    plt.figure(figsize=(10, 6))
    plt.plot(frames, visible_counts, linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Visible Track Points')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def construct_full_query(segmentations, seed_frame_num=1):
    frame_num = len(segmentations)
    all_frame_idx = [i*frame_num//seed_frame_num for i in range(seed_frame_num)]
    all_frame_idx = [0, 69] #yhc: for revision_rotateOcclu, only use the first and mid frame
    full_queries = {}
    full_obj_id_nums = {}
    for frame_idx in all_frame_idx:
        segmentation_query_frame = segmentations[frame_idx]
        # sample tracking queries from segmentation, also get the corresponding points from point map
        queries, obj_id_nums = construct_query_from_segmentations(segmentation_query_frame, frame_idx=frame_idx, sample_rate=0.01)
        full_queries[frame_idx] = queries
        full_obj_id_nums[frame_idx] = obj_id_nums
    return full_queries, full_obj_id_nums, all_frame_idx
# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
def construct_query_from_segmentations(segmentation_query_frame, frame_idx=0, sample_rate = 0.01, target_obj_id=None):
    query = []
    obj_id_nums = {}
    for obj_id, mask in segmentation_query_frame.items():
        if target_obj_id is not None:
            if obj_id != target_obj_id:
                continue
        points2D = np.where(mask > 0)
        #sample the points in 2D space
        points_num = len(points2D[0])
        sample_num = int(points_num * sample_rate)
        points2D_idx = np.random.choice(points_num, sample_num, replace=False)
        points_t = np.zeros((sample_num, 3))
        points_t[:, 0] = frame_idx
        points_t[:, 1] = points2D[2][points2D_idx]
        points_t[:, 2] = points2D[1][points2D_idx]
        
        query.append((points_t))
        obj_id_nums[obj_id] = sample_num
    queries = torch.from_numpy(np.concatenate(query, axis=0)).to(DEFAULT_DEVICE).float()

    return queries[None], obj_id_nums
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
   
    # sample tracking queries from segmentation, also get the corresponding points from point map
    full_queries, full_obj_id_nums, all_frame_idx = construct_full_query(segmentations, seed_frame_num=5)

    # Initialize merged data structures per object
    merged_tracks = {}      # {obj_id: list of pred_tracks_obj arrays}
    merged_visibility = {}  # {obj_id: list of pred_visibility_obj arrays}
    merged_3d_points = {}   # {obj_id: list of tracked_points_obj arrays}

    # Loop through all seed frames
    for seed_frame_idx in all_frame_idx:
        print(f"Processing seed frame {seed_frame_idx}...")
        pred_tracks, pred_visibility = model(
            video,
            queries=full_queries[seed_frame_idx],
            backward_tracking=args.backward_tracking,
        )
        print(f"Seed frame {seed_frame_idx} computed")

        # get tracked 3d points by obj id
        acc_id = 0
        for obj_id in full_obj_id_nums[seed_frame_idx].keys():
            obj_id_end = acc_id + full_obj_id_nums[seed_frame_idx][obj_id]
            pred_tracks_obj = pred_tracks[0,:,acc_id:obj_id_end,:] # t, k, 2
            pred_visibility_obj = pred_visibility[0,:,acc_id:obj_id_end]

            # Per-seed-frame visualization
            vis = Visualizer(save_dir=os.path.join(working_dir, f"vis_tracks_{obj_id}_seed{seed_frame_idx}"), pad_value=120, linewidth=3)
            vis.visualize(
                video,
                pred_tracks_obj[None],
                pred_visibility_obj[None],
                query_frame=0 if args.backward_tracking else args.grid_query_frame,
            )
            pred_tracks_obj = pred_tracks_obj.cpu().numpy()
            pred_visibility_obj = pred_visibility_obj.cpu().numpy()

            # Plot visible point count for this single track
            plot_save_path = os.path.join(working_dir, f"vis_tracks_{obj_id}_seed{seed_frame_idx}", "visible_point_count.png")
            plot_visible_point_count(
                pred_visibility_obj,
                plot_save_path,
                f"Object {obj_id} - Seed Frame {seed_frame_idx}: Visible Points per Frame"
            )

            # Filter valid points (within image bounds for all frames)
            valid_points_mask = np.all(np.all(pred_tracks_obj >= 0, axis=-1) & np.all(pred_tracks_obj < H, axis=-1), axis=0)
            pred_tracks_obj = pred_tracks_obj[:,valid_points_mask,:]
            pred_visibility_obj = pred_visibility_obj[:,valid_points_mask]

            # Get 3D points for each frame
            tracked_points_obj = []
            for t in range(pred_tracks_obj.shape[0]):
                point_map_t = np.load(os.path.join(working_dir, "point_maps", f"{t:05d}.npy"), allow_pickle=True).item()
                points3D_all_t = point_map_t['world_points'][0][0]
                points3D_t = points3D_all_t[pred_tracks_obj[t,:,1].astype(int), pred_tracks_obj[t,:,0].astype(int), :]
                tracked_points_obj.append(points3D_t)
            tracked_points_obj = np.array(tracked_points_obj)

            # Initialize lists if first seed frame for this object
            if obj_id not in merged_tracks:
                merged_tracks[obj_id] = []
                merged_visibility[obj_id] = []
                merged_3d_points[obj_id] = []

            # Append to merged lists
            merged_tracks[obj_id].append(pred_tracks_obj)
            merged_visibility[obj_id].append(pred_visibility_obj)
            merged_3d_points[obj_id].append(tracked_points_obj)

            acc_id = obj_id_end

    # Merge all results by concatenating along k dimension (axis=1)
    final_tracked_points = {}
    final_tracks_2d = {}
    final_visibility = {}

    for obj_id in merged_3d_points.keys():
        # Concatenate along points dimension (axis=1)
        final_tracked_points[obj_id] = np.concatenate(merged_3d_points[obj_id], axis=1)
        final_tracks_2d[obj_id] = np.concatenate(merged_tracks[obj_id], axis=1)
        final_visibility[obj_id] = np.concatenate(merged_visibility[obj_id], axis=1)
        print(f"Object {obj_id}: merged {len(merged_3d_points[obj_id])} seed frames, total {final_tracked_points[obj_id].shape[1]} points")

    # Generate combined visualization per object
    for obj_id in final_tracks_2d.keys():
        vis = Visualizer(save_dir=os.path.join(working_dir, f"vis_tracks_{obj_id}_merged"), pad_value=120, linewidth=3)
        vis.visualize(
            video,
            torch.from_numpy(final_tracks_2d[obj_id])[None].to(DEFAULT_DEVICE),
            torch.from_numpy(final_visibility[obj_id])[None].to(DEFAULT_DEVICE),
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
        )

        # Plot visible point count for merged track
        plot_save_path = os.path.join(working_dir, f"vis_tracks_{obj_id}_merged", "visible_point_count.png")
        plot_visible_point_count(
            final_visibility[obj_id],
            plot_save_path,
            f"Object {obj_id} - Merged: Visible Points per Frame"
        )

    # Save merged 3D tracked points (main output)
    np.save(os.path.join(working_dir, "tracked_points.npy"), final_tracked_points)

    # Save visibility separately (same shape correspondence: {obj_id: (T, K)})
    np.save(os.path.join(working_dir, "tracked_points_vis.npy"), final_visibility)

    # Save merged tracking data
    merged_tracks_data = {
        'pred_tracks_2d': final_tracks_2d,
        'pred_visibility': final_visibility,
        'tracked_points_3d': final_tracked_points,
        'seed_frames': all_frame_idx,
        'obj_id_nums_per_seed': full_obj_id_nums,
    }
    np.save(os.path.join(working_dir, "tracks_merged.npy"), merged_tracks_data)

    print(f"Saved merged results: tracked_points.npy and tracks_merged.npy")
