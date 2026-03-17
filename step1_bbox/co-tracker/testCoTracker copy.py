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
def construct_query_from_segmentations(segmentation_query_frame, frame_idx=0, sample_rate = 0.01):
    query = []
    obj_id_nums = {}
    for obj_id, mask in segmentation_query_frame.items():
        points = np.where(mask > 0)
        #sample the points
        points_num = len(points[0])
        sample_num = int(points_num * sample_rate)
        points_idx = np.random.choice(points_num, sample_num, replace=False)
        points_t = np.zeros((sample_num, 3))
        points_t[:, 0] = frame_idx
        points_t[:, 1] = points[2][points_idx]
        points_t[:, 2] = points[1][points_idx]
        query.append((points_t))
        obj_id_nums[obj_id] = sample_num
    queries = torch.from_numpy(np.concatenate(query, axis=0)).to(DEFAULT_DEVICE).float()

    return queries[None], obj_id_nums

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
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
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
    working_dir = os.path.dirname(args.video_path)
    
    segmentations_path = os.path.join(working_dir, "video_segments.npy")
    segmentations = np.load(segmentations_path, allow_pickle=True).item()
    segmentation_query_frame = segmentations[args.grid_query_frame]
    queries, obj_id_nums = construct_query_from_segmentations(segmentation_query_frame, frame_idx=args.grid_query_frame, sample_rate=0.01)
    # queries = torch.tensor([[0, 0, 0]]).to(DEFAULT_DEVICE).float()
    # queries = queries.reshape(1, -1, 3)
    # breakpoint()
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

    pred_tracks, pred_visibility = model(
        video,
        queries=queries,
        backward_tracking=args.backward_tracking,
        # segm_mask=segm_mask
    )
    print("computed")
    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir=os.path.join(working_dir, "vis_tracks"), pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0 if args.backward_tracking else args.grid_query_frame,
    )
    # save the pred_tracks and pred_visibility
    tracks = {}
    tracks['queries'] = queries.cpu().numpy()
    tracks['pred_tracks'] = pred_tracks.cpu().numpy()
    tracks['pred_visibility'] = pred_visibility.cpu().numpy()
    tracks['obj_id_nums'] = obj_id_nums
    
    np.save(os.path.join(working_dir, "tracks.npy"), tracks)
