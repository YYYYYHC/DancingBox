# DancingBox: A Lightweight MoCap System for Character Animation from Physical Proxies

**CHI 2026 | Honorable Mention Award**

DancingBox captures motion from everyday physical objects (boxes, cups, etc.) using a single RGB camera (your phone, for example) and generates full-body character animation — no markers, suits, or depth sensors required.

> I'm open for collaboration. You may contact hcyuan3@gmail.com if interested.


<p align="center">
  <img src="resources/tmp.gif" alt="DancingBox teaser" width="600"/>
</p>

## Pipeline Overview

The system runs in two stages:

1. **Step 1 — Bounding Box Extraction** (`step1_bbox/`): Takes an input video and extracts 3D bounding boxes of physical proxy objects using SAM2 segmentation, CoTracker3 point tracking, and Pi3 monocular 3D reconstruction.

2. **Step 2 — Motion Generation** (`step2_motion/`): A conditional diffusion model takes the extracted bounding boxes as spatial hints and generates plausible full-body human motion (SMPL format).

## Installation

The two pipeline steps use separate conda environments.

**Step 1 — Bounding Box Extraction:**

```bash
conda env create -f step1_bbox/environment.yml
conda activate s1bbox
```

**Step 2 — Motion Generation:**

```bash
conda env create -f step2_motion/environment.yml
conda activate s2motion
```

## Model Downloads

All model files below are git-ignored (each > 50 MB). Download them before running the pipeline.

### SAM2 checkpoints

| File | Size | Download |
|---|---|---|
| `step1_bbox/sam2/checkpoints/sam2.1_hiera_tiny.pt` | 149 MB | [Meta official](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) |
| `step1_bbox/sam2/checkpoints/sam2.1_hiera_small.pt` | 176 MB | [Meta official](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| `step1_bbox/sam2/checkpoints/sam2.1_hiera_base_plus.pt` | 309 MB | [Meta official](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| `step1_bbox/sam2/checkpoints/sam2.1_hiera_large.pt` | 858 MB | [Meta official](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

```bash
cd step1_bbox/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### CoTracker3 checkpoints

| File | Size | Download |
|---|---|---|
| `step1_bbox/co-tracker/checkpoints/scaled_offline.pth` | 97 MB | [HuggingFace](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth) |
| `step1_bbox/co-tracker/checkpoints/scaled_online.pth` | 97 MB | [HuggingFace](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth) |

```bash
cd step1_bbox/co-tracker/checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

### Pi3

The Pi3 model auto-downloads from HuggingFace (`yyfz233/Pi3`) at runtime — no manual download needed.

### Motion generation model

| File | Size | Download |
|---|---|---|
| `step2_motion/ckpt/model.pt` | 168 MB | [Google Drive](https://drive.google.com/file/d/1NE5v6S8t4m3MANICYqnPWj2elxtZ2Y35/view?usp=sharing) |

Place the checkpoint at `step2_motion/ckpt/model.pt` before running Step 2.

## Usage

### Step 1: Extract bounding boxes

Place your input video under `step1_bbox/myvideos/`, then run:

```bash
conda activate s1bbox
cd step1_bbox
python run_single_video.py
```

> **Note1:** Use label `0` to indicate the ground plane object in the interactive prompt. Start an X server if you want `save_vis=True` in a headless environment. 

> **Note2:** When clicking, positive clicks will be automatically considered as negative clicks for other parts. Utilizing this feature saves a lot of clicks from my experience.

The output bounding boxes will be saved to `step1_bbox/working_dir_<video_name>/bboxs/bboxs.npy`.

You can optionally inspect the 3D bounding boxes in Blender using `tools/vis_hints.blend` (modify the bbox path in the Blender script, then run).

### Step 2: Generate motion

From `step2_motion/`, run the generation script with your extracted bounding boxes:

```bash
conda activate s2motion
cd step2_motion

python -m sample.custom_generate_sequence \
  --model_path ./ckpt/model.pt \
  --output_dir ../results/ \
  --hint_path ../step1_bbox/working_dir_<video_name>/bboxs/bboxs.npy \
  --text_prompt 'put your prompt here'\
  --target_height 1.3 --y_offset 0.2 \
  --torso_idx 2 \
  --rotation_idx 3 \
  --no_vis_bbox

# an example hint can be tested by 
python -m sample.custom_generate_sequence --model_path ./ckpt/model.pt --num_repetitions 1 --output_dir ../results/ --hint_path ../examples/ASingleBoxJumpping/bboxs.npy --torso_idx 0 --target_height 1.3 --y_offset 0.2  --no_vis_bbox --no_dataset --text_prompt 'jump'
```
- `--text_prompt`: this is optional, sometimes the cubes are enough to express a motion.
- `--target_height` / `--y_offset`: normalization parameters that may need hand-tuning per input (see [Known Limitations](#known-limitations))
- `--rotation_idx`: camera viewpoint index (0–9) if you want multiple generation from a fixed rotation; the script generates 10 rotating poses without this parameter by default
- `--torso_idx`: index of the torso bounding box, if you use label n for the torso, you need to set this as n-1, since the ground object is ignored. You may also use the blender tool to check the torso cube's idx.
- Remove `--no_vis_bbox` to visualize bounding box alignment with the generated motion, this will take longer when saving videos.

You can first run without `--rotation_idx` to find best angle for the input sequence. Then you may re-run with the best idx to sample more motions from that view.

### Step 3: Export to BVH (optional)

Convert the generated motion to BVH format for use in Blender or other 3D software:

```bash
python tools/get_bvh.py ./results/results.npy -o results/
```

## Known Limitations

- **Inference speed.** The full pipeline (reconstruction + diffusion sampling)  takes time. I'm considering pushing the performance towards real-time. 
- **Normalization parameters require hand-tuning.** The `--target_height` and `--y_offset` flags often need manual adjustment per input to get good results. I'm investigating in the root cause (likely the motion generator is over fitting to specific 'facing direction')



## Citation

If you find this work useful, please cite:

```bibtex
ToBeAdded
```

## Training Data and Scripts

Coming soon :)
