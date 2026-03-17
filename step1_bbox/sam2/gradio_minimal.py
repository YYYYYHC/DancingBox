# -*- coding: utf-8 -*-
"""
SAM-2 Single Image Annotation Demo (Positive/Negative Click Support)
---------------------------------------------------------------------
Two ways to select an image:
1. Upload a local image
2. Enter server image absolute path and click "Load Server Image"

Then click on the image to annotate, with real-time SAM-2 mask preview.
Click type can be switched in the radio button:
- Positive (label = 1)
- Negative (label = 0)

Annotation Logic:
1. Maintain global `positive_pool` - collects all historical positive sample coordinates (deduplicated).
2. When creating a new obj_id (first click):
   - Use all coordinates in `positive_pool` to generate negative samples (label = 0) for this object.
3. When adding a Positive Click:
   - Only add coordinates to `positive_pool`; does NOT write back to existing objects' negative sample lists.
   - This ensures completed old objects are not polluted by subsequent operations.

Save format:
```json
{
  "<obj_id>": [[x, y, label], ...]
}
```
"""

###############################################################################
# Dependencies
###############################################################################
import os
import json
import inspect
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# SAM-2 imports - assumes sam2 and utils are in PYTHONPATH
from sam2.build_sam import build_sam2_video_predictor
from utils.vis import show_mask, show_points_colored, show_points_colored_negative

###############################################################################
# Command Line Arguments
###############################################################################
parser = argparse.ArgumentParser(description='DancingBox Interface')
parser.add_argument('--work-dir', type=str, default='/home/yhc/vggt/working_dir',
                    help='Working directory path where clicks.json will be saved')
parser.add_argument('--port', type=int, default=7860,
                    help='Gradio server port')
parser.add_argument('--share', action='store_true',
                    help='Create public link')
parser.add_argument('--debug', action='store_true',
                    help='Save segmentation images after each click to debug_log folder')
args = parser.parse_args()

###############################################################################
# Configuration
###############################################################################
SAM2_CHECKPOINT = "/home/yhc/vggt/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_DIR = args.work_dir
TMP_DIR = os.path.join(WORK_DIR, "tmp_demo")
TESTING_PATH = os.path.join(WORK_DIR, "images_processed/00000.jpg")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
DEBUG_MODE = args.debug
DEBUG_LOG_DIR = os.path.join(WORK_DIR, "debug_log")
if DEBUG_MODE:
    import shutil
    if os.path.exists(DEBUG_LOG_DIR):
        shutil.rmtree(DEBUG_LOG_DIR)
    os.makedirs(DEBUG_LOG_DIR)

###############################################################################
# Lazy-load SAM-2 predictor (reduce first click latency)
###############################################################################
_predictor = None
_inference_state = None
_cached_img_dir = None
_click_counter = 0

def _lazy_init_predictor(img_dir: str):
    """Load checkpoint and initialize state for single frame if predictor not yet instantiated"""
    global _predictor, _inference_state, _cached_img_dir
    if _predictor is None:
        _predictor = build_sam2_video_predictor(
            MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE
        )
    # Only reinitialize if directory changed
    if _cached_img_dir != img_dir:
        _inference_state = _predictor.init_state(video_path=img_dir)
        _cached_img_dir = img_dir
    _predictor.reset_state(_inference_state)
    return _predictor, _inference_state

###############################################################################
# Core Functions
###############################################################################

def _ensure_obj_initialized(
    obj_id: int,
    clicks: Dict[int, List[List[float]]],
    positive_pool: List[Tuple[float, float]],
):
    """Initialize obj_id if new, AND add any new positive_pool points as negatives
    Click format: [x, y, label, is_manual] where is_manual=False for auto-negatives"""
    if obj_id not in clicks:
        # New object: initialize with all positive_pool as auto-negatives
        clicks[obj_id] = [[x, y, 0, False] for (x, y) in positive_pool]
    else:
        # Existing object: add any NEW positive_pool points as auto-negatives
        existing_coords = set((c[0], c[1]) for c in clicks[obj_id])
        for (x, y) in positive_pool:
            if (x, y) not in existing_coords:
                clicks[obj_id].append([x, y, 0, False])


def _add_click_and_visualise(
    img_path: str,
    xy: Tuple[float, float],
    obj_id: int,
    click_label: int,
    clicks: Dict[int, List[List[float]]],
    positive_pool: List[Tuple[float, float]],
    masks: Dict[int, np.ndarray],
):
    """Process one click and return (updated_clicks, updated_positive_pool, updated_masks, full_seg_img)"""
    predictor, inf_state = _lazy_init_predictor(os.path.dirname(img_path))

    # 1. Ensure new object negative sample initialization
    _ensure_obj_initialized(obj_id, clicks, positive_pool)

    # 2. Prepare points/labels for SAM-2 inference
    history = clicks[obj_id]
    points = np.array([xy] + [[c[0], c[1]] for c in history], dtype=np.float32)
    labels = np.array([click_label] + [c[2] for c in history], dtype=np.int32)

    # 3. Call SAM-2
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inf_state,
        frame_idx=0,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # 4. Update clicks and positive_pool (mark as manual click)
    clicks[obj_id].append([xy[0], xy[1], click_label, True])

    if click_label == 1 and (xy[0], xy[1]) not in positive_pool:
        positive_pool.append((xy[0], xy[1]))
        # Add this new positive as auto-negative to ALL other existing parts
        for other_obj_id in clicks:
            if other_obj_id != obj_id:
                existing_coords = set((c[0], c[1]) for c in clicks[other_obj_id])
                if (xy[0], xy[1]) not in existing_coords:
                    clicks[other_obj_id].append([xy[0], xy[1], 0, False])

    # 5. Store mask for this object
    masks[obj_id] = (out_mask_logits[-1] > 0).cpu().numpy()

    # 6. Generate full segmentation view (re-runs SAM-2 with all prompts)
    full_seg_preview = _generate_full_segmentation(img_path, clicks)

    return clicks, positive_pool, masks, full_seg_preview


def _generate_full_segmentation(img_path, clicks):
    """Generate full segmentation by re-running SAM-2 with all prompts from all objects"""
    if not clicks:
        # No clicks yet, just return the original image
        return Image.open(img_path)

    # Collect ALL positive points from ALL objects
    all_positives = {}  # obj_id -> list of (x, y) positive coords
    for obj_id, obj_clicks in clicks.items():
        all_positives[obj_id] = [(c[0], c[1]) for c in obj_clicks if c[2] == 1]

    # Initialize fresh predictor state for full segmentation
    predictor, full_inf_state = _lazy_init_predictor(os.path.dirname(img_path))

    # Add prompts for ALL objects with COMPLETE cross-object negatives
    for obj_id, obj_clicks in clicks.items():
        if len(obj_clicks) == 0:
            continue

        # Start with this object's own clicks
        points_list = [[c[0], c[1]] for c in obj_clicks]
        labels_list = [c[2] for c in obj_clicks]

        # Add OTHER objects' positive points as negatives (if not already present)
        existing_coords = set((c[0], c[1]) for c in obj_clicks)
        for other_obj_id, other_positives in all_positives.items():
            if other_obj_id == obj_id:
                continue
            for (x, y) in other_positives:
                if (x, y) not in existing_coords:
                    points_list.append([x, y])
                    labels_list.append(0)  # negative
                    existing_coords.add((x, y))

        points = np.array(points_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)

        predictor.add_new_points_or_box(
            inference_state=full_inf_state,
            frame_idx=0,
            obj_id=int(obj_id),
            points=points,
            labels=labels,
        )

    # Get FINAL masks for all objects together (frame 0 only)
    full_masks = {}
    for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(full_inf_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            full_masks[out_obj_id] = (out_mask_logits[i] > 0).cpu().numpy()
        break  # Only need frame 0

    # Visualization
    img = Image.open(img_path)
    w, h = img.size
    plt.figure(figsize=(w / 50, h / 50), dpi=50)
    plt.axis("off")
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    plt.imshow(img)

    cmap = plt.get_cmap("tab10")

    # Draw all masks from unified inference
    for obj_id, mask in full_masks.items():
        show_mask(mask, ax, obj_id=obj_id)

    # Draw only manual clicks (skip auto-generated negatives)
    # Click format: [x, y, label, is_manual]
    for obj_id, obj_clicks in clicks.items():
        color = cmap(int(obj_id))[:3]  # RGB without alpha
        # Filter manual positive clicks (label=1, is_manual=True)
        pos_points = np.array([[c[0], c[1]] for c in obj_clicks if len(c) > 3 and c[3] and c[2] == 1])
        # Filter manual negative clicks (label=0, is_manual=True)
        neg_points = np.array([[c[0], c[1]] for c in obj_clicks if len(c) > 3 and c[3] and c[2] == 0])
        if len(pos_points) > 0:
            show_points_colored(pos_points, ax, color)
        if len(neg_points) > 0:
            show_points_colored_negative(neg_points, ax, color)

    prev = os.path.join(TMP_DIR, "full_seg.jpg")
    plt.savefig(prev, pad_inches=0.0)
    plt.close()

    if DEBUG_MODE:
        global _click_counter
        _click_counter += 1
        debug_path = os.path.join(DEBUG_LOG_DIR, f"click_{_click_counter:04d}.jpg")
        import shutil
        shutil.copy(prev, debug_path)

    return Image.open(prev)


def _save_clicks(clicks: Dict[int, List[List[float]]]):
    """Save annotations to clicks.json in working directory"""
    save_path = os.path.join(WORK_DIR, "clicks.json")

    # Save to working directory
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(clicks, f, ensure_ascii=False, indent=2)

    # Return success message
    num_objects = len(clicks)
    total_clicks = sum(len(obj_clicks) for obj_clicks in clicks.values())

    return f"Saved to: {save_path}\nContains {num_objects} objects, {total_clicks} clicks total"

###############################################################################
# Gradio UI
###############################################################################
with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as demo:
    gr.Markdown("## SAM-2 Single Image Annotation Demo")

    # --- Image Selection (hidden) ---
    with gr.Row():
        image_upload = gr.Image(label="Upload Local Image", type="pil", visible=False)
        server_path = gr.Textbox(label="Server Image Path", value=TESTING_PATH, visible=False)
        load_srv_btn = gr.Button("Start Segmentation")

    # --- Preview & Click ---
    with gr.Row():
        if "tool" in inspect.signature(gr.Image.__init__).parameters:
            image_preview = gr.Image(label="Click to Segment", tool="select", height=480)
        else:
            image_preview = gr.Image(label="Click to Segment", interactive=True, height=480)

    with gr.Row():
        obj_id_box = gr.Number(label="Current obj_id", value=1, precision=0)
        click_type_radio = gr.Radio(["Positive", "Negative"],
                                     value="Positive", label="Click Type")

    # --- Finish Button ---
    with gr.Row():
        finish_btn = gr.Button("Finish Segmentation", variant="primary")

    # Video display
    video_display = gr.Video(label="Result Video")

    # Save status display
    save_status = gr.Textbox(label="Save Status", interactive=False, lines=2, visible=False)

    # --- Hidden State ---
    img_path_state      = gr.State("")
    clicks_state        = gr.State({})
    positive_pool_state = gr.State([])
    masks_state         = gr.State({})

    ###########################################################################
    # Callbacks
    ###########################################################################

    def _load_local_image(img):
        if img is None:
            return gr.update(), "", {}, [], {}
        p = os.path.join(TMP_DIR, "input.png")
        img.save(p)
        _lazy_init_predictor(TMP_DIR)
        return img, p, {}, [], {}

    image_upload.change(_load_local_image,
                        inputs=image_upload,
                        outputs=[image_preview, img_path_state, clicks_state, positive_pool_state, masks_state])

    def _load_server_image(path):
        path = path.strip()
        if not path or not os.path.isfile(path):
            return gr.update(value=None), "", {}, [], {}
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return gr.update(value=None), "", {}, [], {}
        _lazy_init_predictor(os.path.dirname(path))
        return img, path, {}, [], {}

    load_srv_btn.click(_load_server_image,
                       inputs=server_path,
                       outputs=[image_preview, img_path_state, clicks_state, positive_pool_state, masks_state])

    def _on_click(evt: gr.SelectData,
                  img_path,
                  clicks,
                  positive_pool,
                  masks,
                  obj_id,
                  click_type):
        if not img_path:
            return clicks, positive_pool, masks, gr.update()

        # Extract coordinates (compatible with different gradio versions)
        if hasattr(evt, "index"):
            xy = (evt.index[0], evt.index[1])
        elif hasattr(evt, "x"):
            xy = (evt.x, evt.y)
        else:
            return clicks, positive_pool, masks, gr.update()

        label = 1 if "Positive" in click_type else 0
        new_clicks, new_pool, new_masks, full_preview = _add_click_and_visualise(
            img_path, xy, int(obj_id), label, clicks, positive_pool, masks
        )
        return new_clicks, new_pool, new_masks, full_preview

    image_preview.select(_on_click,
                         inputs=[img_path_state, clicks_state, positive_pool_state, masks_state, obj_id_box, click_type_radio],
                         outputs=[clicks_state, positive_pool_state, masks_state, image_preview])

    # Video path to display after finishing
    VIDEO_PATH = "/home/yhc/vggt/demo.mp4"

    # Finish button callback (save and exit)
    def finish_segmentation(clicks):
        """Save clicks and exit app"""
        result = _save_clicks(clicks)
        print(result)
        os._exit(0)

    finish_btn.click(
        finish_segmentation,
        inputs=clicks_state,
        outputs=video_display
    )

if __name__ == "__main__":
    print(f"Working directory: {WORK_DIR}")
    print(f"clicks.json will be saved to: {os.path.join(WORK_DIR, 'clicks.json')}")

    try:
        app = demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            inbrowser=False,
            prevent_thread_lock=False
        )
    except Exception as e:
        print(f"Launch failed: {e}")
        app = demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            inbrowser=False,
            prevent_thread_lock=False
        )