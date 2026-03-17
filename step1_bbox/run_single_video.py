import os
import subprocess
import time 
import json

def check_files(video_paths):
    for v in video_paths:
        if not os.path.exists(v):
            print(f"文件不存在: {v}")
            return False
    return True
video_name="008593amraiseArm"
testid="008593amraiseArm"
interval = 1
run_pi3=True
run_sam2=True
run_labeling=True
run_segmentation=True
run_cotracker=True
run_pcb2bbox=True
save_vis=True
video=f"myvideos/{video_name}.mp4"

if not check_files([video]):
    print("文件检查失败")
    exit(1)
print("文件检查通过")
working_dir=f"../results/working_dir_{testid}"

times = {"pi3": 0, "sam2": 0, "cotracker": 0, "pcb2bbox": 0}

# switch to your python path
python_path="/home/yhc/miniconda3/envs/vggt/bin/python"

# step1: run pi3
if run_pi3:
    start_time = time.time()
    subprocess.run(f"{python_path} Pi3/testPi3.py --data_path {video} --save_path {working_dir} --interval {interval}", shell=True)
    times["pi3"] = time.time() - start_time 
# step2: run sam2
if run_sam2 and run_labeling:
    # 2.1 labelling
    print(f"{python_path} sam2/gradio_minimal.py --work-dir {working_dir}")
    subprocess.run(f"{python_path} sam2/gradio_minimal.py --work-dir {working_dir} --debug", shell=True)
    times["sam2"] = time.time() - start_time 
for w in [working_dir]:
    clicks_path = os.path.join(w, "clicks.json")
    if not os.path.exists(clicks_path):
        print(f"clicks.json 不存在: {clicks_path}")
        exit(1)
print("clicks 检查通过")
    # 2.2 segmentation
if run_sam2 and run_segmentation:
    print("running sam2")
    start_time = time.time()
    subprocess.run(f"{python_path} sam2/testSAM2.py --work-dir {working_dir} --save_video {save_vis}", shell=True)
    times["sam2"] = time.time() - start_time 
# step3: run coTracker
if run_cotracker:
    print("running coTracker")
    start_time = time.time()
    video_path = os.path.join(working_dir, "images_processed")
    subprocess.run(f"{python_path} co-tracker/testCoTracker.py --video_path {video_path} --offline", shell=True)
    times["cotracker"] = time.time() - start_time 
for w in [working_dir]:
    video_path = os.path.join(w, "vis_tracks/video.mp4")
    if not os.path.exists(video_path):
        print(f"video.mp4 不存在: {video_path}")
        exit(1)
print("video.mp4 检查通过")

# step4: run pcb2bbox
if run_pcb2bbox:
    start_time = time.time()
    point_map_dir = os.path.join(working_dir, "point_maps_segmented")
    subprocess.run(f"{python_path} pcds2bboxs.py --dir {point_map_dir} --save_vis {save_vis}", shell=True)
    times["pcb2bbox"] = time.time() - start_time 
with open(os.path.join(working_dir, "times.json"), "w") as f:
    json.dump(times, f)


