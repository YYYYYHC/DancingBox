import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from utils.vis import show_mask, show_points, show_box
device = torch.device("cuda")
# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

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
        show_mask((out_mask_logits[-1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[-1])
        
        # 保存图像到指定目录
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"frame_{ann_frame_idx}_{ann_obj_id}.png")
        plt.savefig(save_path)
        plt.close()  # 关闭图像以释放内存

if __name__ == "__main__":
    # select the device for computation
    

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    video_dir = "/home/yhc/vggt/working_dir/images_processed"
    point_map_dir = "/home/yhc/vggt/working_dir/point_maps"
    tmp_dir = "/home/yhc/vggt/working_dir/tmp"
    video_save_dir = "/home/yhc/vggt/working_dir/images_segmented"
    point_map_save_dir = "/home/yhc/vggt/working_dir/point_maps_segmented"
    os.makedirs(video_save_dir, exist_ok=True)
    os.makedirs(point_map_save_dir, exist_ok=True)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    annotate_frame(predictor,
                   0, 
                   1,
                   np.array([[150, 110]], dtype=np.float32),
                   np.array([1], np.int32),
                   save_dir=tmp_dir)
    
    annotate_frame(predictor,
                   0, 
                   2,
                   np.array([[250, 150]], dtype=np.float32),
                   np.array([1], np.int32),
                   save_dir=tmp_dir)
    
    annotate_frame(predictor,
                   0, 
                   3,
                   np.array([[400, 125]], dtype=np.float32),
                   np.array([1], np.int32),
                   save_dir=tmp_dir)
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        point_map_path = os.path.join(point_map_dir, frame_names[out_frame_idx].replace(".jpg", ".npy"))
        point_map = np.load(point_map_path, allow_pickle=True).item()
        world_points = point_map['world_points']
        world_points_conf = point_map['world_points_conf']
        assert world_points.shape[0] == 1 and world_points.shape[1] == 1
        world_points = world_points[0]
        world_points_conf = world_points_conf[0]
        for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
            mask_logits = mask_logits.cpu().numpy()
            obj_points = world_points[mask_logits > 0.0]
            obj_points_conf = world_points_conf[mask_logits > 0.0]
            to_save = {
                'world_points': obj_points,
                'world_points_conf': obj_points_conf
            }
            np.save(os.path.join(point_map_save_dir, f"{out_frame_idx:06d}_{obj_id}.npy"), to_save, allow_pickle=True)
            
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    vis_frame_stride = 1

    plt.close("all")
    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
         # 保存图像到指定目录
        
        save_path = os.path.join(video_save_dir, f"{out_frame_idx:06d}.png")
        plt.savefig(save_path)
        plt.close()  # 关闭图像以释放内存
    