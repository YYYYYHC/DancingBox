import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import numpy as np
import open3d as o3d
from PIL import Image
import os

def save_point_maps(res, saving_path, save_ply=True, imgs=None):
    world_points_all = res['points'][0]
    world_points_conf_all = res['conf'][0].squeeze(-1)

    #for now only single batch and single image
    
    # x-right, y-down, z-forward
    # to
    # x-right, y-up, z-back
    point_map_save_dir = os.path.join(saving_path, 'point_maps')
    os.makedirs(point_map_save_dir, exist_ok=True)
    rot = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]],device=world_points_all.device, dtype=world_points_all.dtype)
    world_points_all = world_points_all @ rot.T
    world_points_all = world_points_all.to(torch.float32)
    
    world_points_all = world_points_all.cpu().numpy()
    world_points_conf_all = world_points_conf_all.cpu().numpy()
    for i in range(world_points_all.shape[0]):
        point_map_save_path = os.path.join(point_map_save_dir, f'{i:05d}.npy')
        world_points = world_points_all[i][np.newaxis, np.newaxis]
        world_points_conf = world_points_conf_all[i][np.newaxis, np.newaxis]
        tosave = {
            'world_points': world_points,
            'world_points_conf': world_points_conf
        }
        np.save(point_map_save_path, tosave, allow_pickle=True)
    
        # save the ply file
        if save_ply:
            write_ply(world_points_all[i], imgs[i].permute( 1, 2, 0), os.path.join(point_map_save_dir, f'{i:05d}.ply'))
def save_images(images, saving_path):
    assert len(images.shape) == 4
    images = images.permute(0, 2, 3, 1)
    images = images.cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(saving_path, f'{i:05d}.jpg'))

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='/home/yhc/vggt/working_dir',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()
    os.makedirs(args.save_path, exist_ok=True)
    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)
    os.makedirs(os.path.join(args.save_path, 'images_processed'), exist_ok=True)
    save_images(imgs, os.path.join(args.save_path, 'images_processed'))
    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    save_point_maps(res, args.save_path, imgs=imgs)