import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_points_colored(coords, ax, color, marker_size=1200):
    """Show points using a specific color (RGB tuple, values 0-1), solid (not transparent)"""
    if len(coords) == 0:
        return
    rgba = (*color[:3], 1.0)
    ax.scatter(coords[:, 0], coords[:, 1], color=rgba, marker='*',
               s=marker_size, edgecolor='black', linewidth=1.25)


def show_points_colored_negative(coords, ax, color, marker_size=1200):
    """Show negative points with red boundary and mask-colored interior"""
    if len(coords) == 0:
        return
    rgba = (*color[:3], 1.0)
    ax.scatter(coords[:, 0], coords[:, 1], color=rgba, marker='*',
               s=marker_size, edgecolor='red', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))