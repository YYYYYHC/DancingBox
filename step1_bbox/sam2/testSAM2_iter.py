import os
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from utils.vis import show_mask, show_points, show_box
import open3d as o3d
device = torch.device("cuda")
# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "/home/yhc/vggt/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


import torch
from collections import deque
import torch
import numpy as np
from scipy import ndimage

def keep_largest_cc(out_mask_logits: torch.Tensor) -> torch.Tensor:
    """
    仅保留 out_mask_logits 中最大的连通分支，其余 logits 设为 0。
    Args:
        out_mask_logits: shape [1, H, W]  (float tensor)
    Returns:
        logits_same_shape: 同形状 tensor
    """
    logits = out_mask_logits.clone()          # 不改动原 tensor
    mask_np = (logits[0] > 0).cpu().numpy()   # -> bool numpy, 去掉 batch 维

    # 8 邻域结构元：3x3 全 1
    labeled, n_cc = ndimage.label(mask_np, structure=np.ones((3, 3)))
    if n_cc == 0:          # 没有正像素，直接返回全 0
        logits.zero_()
        return logits

    # 各连通块像素数，index 0 是背景，先置 0 避免干扰
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_id = sizes.argmax()

    # 构造 bool 掩码，只留最大块
    keep_mask = (labeled == largest_id)
    # 将 torch.Tensor 对应像素保留，其余清 0
    logits[0][~torch.from_numpy(keep_mask)] = 0
    return logits


def annotate_frame(predictor,ann_frame_idx, ann_obj_id, points, labels, save_dir =None):
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    if save_dir is not None:
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        color =show_mask((out_mask_logits[-1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[-1])
        
        # 保存图像到指定目录
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"frame_{ann_frame_idx}_{ann_obj_id}.png")
        plt.savefig(save_path)
        plt.close()  # 关闭图像以释放内存
        return color

if __name__ == "__main__":
    # select the device for computation
    
    args = argparse.ArgumentParser()
    args.add_argument("--work-dir", type=str, required=True)
    args.add_argument("--save_video", type=bool, default=False)
    args = args.parse_args()
    work_dir = args.work_dir
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    video_dir = os.path.join(work_dir, "images_processed")
    point_map_dir = os.path.join(work_dir, "point_maps")
    tmp_dir = os.path.join(work_dir, "tmp")
    video_save_dir = os.path.join(work_dir, "images_segmented")
    point_map_save_dir = os.path.join(work_dir, "point_maps_segmented")
    clicks_file = os.path.join(work_dir, "clicks.json")
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(point_map_save_dir, exist_ok=True)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    

    #add clicks from clicks.json
    with open(clicks_file, 'r') as f:
        clicks = json.load(f)
    all_colors = {}
    for obj_id, clicks_seq in clicks.items():
        points = np.array([[click[0], click[1]] for click in clicks_seq], dtype=np.float32)
        labels = np.array([click[2] for click in clicks_seq], dtype=np.int32)
        if not args.save_video:
            color = np.random.rand(3)
            annotate_frame(predictor, 0, int(obj_id), points, labels, save_dir=None)
        else:
            color = annotate_frame(predictor, 0, int(obj_id), points, labels, save_dir=tmp_dir)
        all_colors[int(obj_id)] = color[:3]
    np.save(os.path.join(work_dir, "all_colors.npy"), all_colors)

    frame_idx = 0
    
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # all_colors = {int(obj_id): np.random.rand(3) for obj_id,_ in clicks.items()}
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

        point_map_path = os.path.join(point_map_dir, frame_names[out_frame_idx].replace(".jpg", ".npy"))
        point_map = np.load(point_map_path, allow_pickle=True).item()
        world_points = point_map['world_points']
        world_points_conf = point_map['world_points_conf']
        assert world_points.shape[0] == 1 and world_points.shape[1] == 1
        world_points = world_points[0]
        world_points_conf = world_points_conf[0]
        ply_file = o3d.geometry.PointCloud()
        points = []
        colors = []
        
        for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
            mask_logits = keep_largest_cc(mask_logits)
            mask_logits = mask_logits.cpu().numpy()
            
            obj_points = world_points[mask_logits > 0.0]
            obj_points_conf = world_points_conf[mask_logits > 0.0]
            to_save = {
                'world_points': obj_points,
                'world_points_conf': obj_points_conf
            }
            np.save(os.path.join(point_map_save_dir, f"{out_frame_idx:06d}_{obj_id}.npy"), to_save, allow_pickle=True)
            # attach the points and colors to the ply file
        
            points.extend(obj_points)
            colors.extend(np.tile(all_colors[obj_id], (len(obj_points), 1)))
        ply_file.points = o3d.utility.Vector3dVector(points)
        ply_file.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(point_map_save_dir, f"{out_frame_idx:06d}.ply"), ply_file)
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    vis_frame_stride = 1
    np.save(os.path.join(os.path.dirname(video_dir), "video_segments.npy"), video_segments)
    if not args.save_video:
        exit()
    plt.close("all")
    # for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
    #     plt.figure(figsize=(6, 4))
    #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         ax = plt.gca()
    #         ax.axis('off')
    #         show_mask(out_mask, ax, obj_id=out_obj_id)
    #      # 保存图像到指定目录
        
    #     save_path = os.path.join(video_save_dir, f"{out_frame_idx:06d}.png")
    #     plt.savefig(save_path,transparent=True,bbox_inches='tight',pad_inches=0)
    #     plt.close()  # 关闭图像以释放内存
    