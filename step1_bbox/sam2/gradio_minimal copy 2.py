# -*- coding: utf-8 -*-
"""
SAM‑2 单图交互标注 Demo （支持正 / 负 Click）
---------------------------------------------------
两种方式选择图片：
1️⃣ 上传本地图片；
2️⃣ 在文本框输入服务器端图片绝对路径并点击 **加载服务器图片**。

随后在图片上交互点击，实时查看 SAM‑2 mask 预览。
点击类型可在「点击类型」单选框里切换：
📌 正样本 (Positive) —— label = 1
❌ 负样本 (Negative) —— label = 0

⚙️ **标注逻辑**
1. 维护全局 `positive_pool` —— 收集所有历史正样本坐标，且去重。
2. **创建新 obj_id**（第一次被点击）时：
   - 用 `positive_pool` 中全部坐标生成该对象的负样本 (`label = 0`)，实现「后加入对象继承负样本」。
3. **添加 Positive Click** 时：
   - 仅把坐标加入 `positive_pool`；**不会**再回写到已有对象的负样本列表。
   - 这样保证：已经完成标注的旧对象不被后续操作污染。

最终保存格式：
```json
{
  "<obj_id>": [[x, y, label], ...]
}
```
"""

###############################################################################
# 依赖
###############################################################################
import os
import json
import inspect
from typing import Dict, List, Tuple

import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# SAM‑2 imports ——假设 sam2 和 utils 已装到 PYTHONPATH
from sam2.build_sam import build_sam2_video_predictor
from utils.vis import show_mask, show_points

###############################################################################
# 配置
###############################################################################
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"  # ← 改成你的路径
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"         # ← 改成你的路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TMP_DIR = "tmp_demo"
TESTING_PATH = "/home/yhc/vggt/working_dir/images_processed/00000.jpg"
os.makedirs(TMP_DIR, exist_ok=True)

###############################################################################
# 懒加载 SAM‑2 predictor（减少首次点击延迟）
###############################################################################
_predictor = None
_inference_state = None

def _lazy_init_predictor(img_dir: str):
    """如果还没实例化 predictor，则加载 checkpoint 并针对单帧初始化 state"""
    global _predictor, _inference_state
    if _predictor is None:
        _predictor = build_sam2_video_predictor(
            MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE
        )
    # 单帧视频初始化
    _inference_state = _predictor.init_state(video_path=img_dir)
    _predictor.reset_state(_inference_state)
    return _predictor, _inference_state

###############################################################################
# 核心函数
###############################################################################

def _ensure_obj_initialized(
    obj_id: int,
    clicks: Dict[int, List[List[float]]],
    positive_pool: List[Tuple[float, float]],
):
    """若 obj_id 首次出现，用 positive_pool 初始化其负样本集合"""
    if obj_id not in clicks:
        clicks[obj_id] = [[x, y, 0] for (x, y) in positive_pool]


def _add_click_and_visualise(
    img_path: str,
    xy: Tuple[float, float],
    obj_id: int,
    click_label: int,
    clicks: Dict[int, List[List[float]]],
    positive_pool: List[Tuple[float, float]],
):
    """处理一次点击并返回 (updated_clicks, updated_positive_pool, preview_img)"""
    predictor, inf_state = _lazy_init_predictor(os.path.dirname(img_path))

    # 1️⃣ 确保新对象负样本初始化
    _ensure_obj_initialized(obj_id, clicks, positive_pool)

    # 2️⃣ SAM‑2 推理所需 points/labels
    history = clicks[obj_id]
    points = np.array([xy] + [[c[0], c[1]] for c in history], dtype=np.float32)
    labels = np.array([click_label] + [c[2] for c in history], dtype=np.int32)

    # 3️⃣ 调 SAM‑2
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inf_state,
        frame_idx=0,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # 4️⃣ 更新 clicks 与 positive_pool
    clicks[obj_id].append([xy[0], xy[1], click_label])

    if click_label == 1 and (xy[0], xy[1]) not in positive_pool:
        positive_pool.append((xy[0], xy[1]))

    # 5️⃣ 可视化
    img = Image.open(img_path)
    w, h = img.size
    plt.figure(figsize=(w / 50, h / 50), dpi=50)
    plt.axis("off")
    ax = plt.gca(); ax.set_axis_off(); ax.set_position([0, 0, 1, 1])
    plt.imshow(img)
    for c in history:
        show_points(np.array([[c[0], c[1]]]), np.array([c[2]]), ax)
    show_points(points, labels, ax)
    show_mask((out_mask_logits[-1] > 0).cpu().numpy(), ax, obj_id=out_obj_ids[-1])
    os.makedirs(TMP_DIR, exist_ok=True)
    prev = os.path.join(TMP_DIR, "preview.jpg")
    plt.savefig(prev, pad_inches=0.0)
    plt.close()

    return clicks, positive_pool, Image.open(prev)


def _save_clicks(clicks: Dict[int, List[List[float]]]):
    """保存标注为 JSON"""
    path = os.path.join(TMP_DIR, "clicks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clicks, f, ensure_ascii=False, indent=2)
    return path

###############################################################################
# Gradio UI
###############################################################################
with gr.Blocks() as demo:
    gr.Markdown("## SAM‑2 单图交互标注 Demo (支持服务器路径 + 正/负 Click)")

    # --- 选择图片 ---
    with gr.Row():
        image_upload = gr.Image(label="上传本地图片", type="pil")
        server_path = gr.Textbox(label="服务器图片绝对路径", value=TESTING_PATH)
        load_srv_btn = gr.Button("加载服务器图片")

    # --- 预览 & 点击 ---
    if "tool" in inspect.signature(gr.Image.__init__).parameters:
        image_preview = gr.Image(label="预览 / 点击加点", tool="select", height=480)
    else:
        image_preview = gr.Image(label="预览 / 点击加点", interactive=True, height=480)

    with gr.Row():
        obj_id_box = gr.Number(label="当前 obj_id", value=1, precision=0)
        click_type_radio = gr.Radio(["正样本 (Positive)", "负样本 (Negative)"],
                                     value="正样本 (Positive)", label="点击类型")

    # --- 保存 ---
    with gr.Row():
        save_btn = gr.Button("保存坐标 JSON")
        json_file = gr.File(label="下载 JSON")

    # --- 隐藏状态 ---
    img_path_state      = gr.State("")
    clicks_state        = gr.State({})
    positive_pool_state = gr.State([])

    ###########################################################################
    # 回调
    ###########################################################################

    def _load_local_image(img):
        if img is None:
            return gr.update(), "", {}, []
        p = os.path.join(TMP_DIR, "input.png")
        img.save(p)
        _lazy_init_predictor(TMP_DIR)
        return img, p, {}, []

    image_upload.change(_load_local_image,
                        inputs=image_upload,
                        outputs=[image_preview, img_path_state, clicks_state, positive_pool_state])

    def _load_server_image(path):
        path = path.strip()
        if not path or not os.path.isfile(path):
            return gr.update(value=None), "", {}, []
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return gr.update(value=None), "", {}, []
        _lazy_init_predictor(os.path.dirname(path))
        return img, path, {}, []

    load_srv_btn.click(_load_server_image,
                       inputs=server_path,
                       outputs=[image_preview, img_path_state, clicks_state, positive_pool_state])

    def _on_click(evt: gr.SelectData,
                  img_path,
                  clicks,
                  positive_pool,
                  obj_id,
                  click_type):
        if not img_path:
            return clicks, positive_pool, gr.update()

        # 提取坐标（兼容不同 gradio 版本）
        if hasattr(evt, "index"):
            xy = (evt.index[0], evt.index[1])
        elif hasattr(evt, "x"):
            xy = (evt.x, evt.y)
        else:
            return clicks, positive_pool, gr.update()

        label = 1 if "正样本" in click_type else 0
        new_clicks, new_pool, preview = _add_click_and_visualise(
            img_path, xy, int(obj_id), label, clicks, positive_pool
        )
        return new_clicks, new_pool, preview

    image_preview.select(_on_click,
                         inputs=[img_path_state, clicks_state, positive_pool_state, obj_id_box, click_type_radio],
                         outputs=[clicks_state, positive_pool_state, image_preview])

    save_btn.click(_save_clicks,
                   inputs=clicks_state,
                   outputs=json_file)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except ValueError:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
