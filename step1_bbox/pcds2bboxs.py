"""
point_cloud_tracking_multi.py — Multi‑part processing & headless visualisation
=============================================================================
This **supersedes** the earlier single‑part script and now adds **robust point‑cloud
pre‑processing** before any further computation.

New features since the last revision
------------------------------------
* **Multi‑part loading** — files laid out as `{frame:06d}_{partID}.npy|ply`.
* **Pre‑processing** (new!) — per‑frame removal of statistical outliers followed by
  optional synthetic densification of the inlier cloud.
* Per‑part bounding‑box motion computed independently with **point‑to‑point ICP**.
* Off‑screen renderer draws **all parts together** each frame, assigns each part
  a distinct colour, **fixed camera** (user‑defined front/lookat/up + distance).

CLI quick‑start (showing the new flags)
---------------------------------------
```bash
python point_cloud_tracking_multi.py \
    --dir   working_dir/point_maps_segmented \
    --parts 1,2,3                 # list of part IDs
    --frames 0,99                 # start,end frame indices (inclusive)
    # ------- new pre‑proc flags ------
    --outlier_nb    24            # neighbours for statistical removal
    --outlier_std   2.0           # std‑ratio threshold
    --densify_k     2             # synthetic pts per input pt (0 = off)
    --densify_sigma 0.005         # Gaussian jitter (m)
    --voxel 0.01                  # (optional) voxel grid after densify
    # ---------------------------------
    --cam_front 0,0,-1 \
    --cam_up    0,-1,0  \
    --cam_dist  3.0               # distance multiplier relative to scene diag
    --out  working_dir/vis_pcds
```

Dependencies: **Open3D 0.18+** for I/O, ICP, off‑screen rendering.
"""

from __future__ import annotations
import os, glob, math, time, itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from scipy.linalg import expm, logm
import json
# headless rendering hint — must appear before the Open3D import
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
import open3d as o3d
os.environ['DISPLAY'] = ':99'

# -----------------------------------------------------------------------------
# Basic structures
# -----------------------------------------------------------------------------
@dataclass
class BoundingBox:
    center: np.ndarray   # (3,)
    extent: np.ndarray   # (3,)
    rotation: np.ndarray # (3,3)
    def corners(self) -> np.ndarray:
        offs = np.array([[sx,sy,sz] for sx in (-0.5,0.5) for sy in (-0.5,0.5) for sz in (-0.5,0.5)])
        return self.center + (offs * self.extent) @ self.rotation.T


def remove_outliers_mahalanobis(X, p=0.99, max_iter=5):
    """
    X: (N, 3)
    p: confidence level; commonly 0.99 (threshold ~11.345), larger = more lenient
    max_iter: maximum number of iterations
    Returns: X_inliers, mask, d2
    """
    X = np.asarray(X)
    # Chi-squared threshold table for df=3 (avoids scipy dependency)
    chi2_thresh_table = {0.95: 7.815, 0.975: 9.348, 0.99: 11.345, 0.997: 14.160}
    thr = chi2_thresh_table.get(p, 11.345)

    mask = np.ones(len(X), dtype=bool)
    for _ in range(max_iter):
        Xc = X[mask]
        mu = Xc.mean(axis=0)
        # rowvar=False -> columns are variables; pinvh is more robust
        S = np.cov(Xc, rowvar=False)
        Sinv = np.linalg.pinv(S)  # may be degenerate, pinv/pinvh is more stable
        d = X - mu
        d2 = np.einsum('ij,jk,ik->i', d, Sinv, d)  # squared Mahalanobis distance
        new_mask = d2 < thr
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    return X[mask], mask, d2
# -----------------------------------------------------------------------------
# Geometry utils — *new*: interpolate transform, geodesic interpolation
# -----------------------------------------------------------------------------
def interpolate_se3_geodesic(T_target, n_frames):
    T_start = np.eye(4)
    
    # Compute the log of the target transform (relative transform)
    T_rel = np.dot(np.linalg.inv(T_start), T_target)
    log_T = logm(T_rel)

    transforms = []
    for alpha in np.linspace(0, 1, n_frames):
        # Interpolate in log space, then map back via matrix exponential
        T_interp = np.dot(T_start, expm(alpha * log_T))
        transforms.append(T_interp.real)  # discard imaginary part (numerical error)
    
    return transforms
# -----------------------------------------------------------------------------
# Geometry utils — *new*: interpolate transform
# -----------------------------------------------------------------------------
def interpolate_transform(T_target, n_frames=10):
    # Decompose into rotation and translation
    R_target = T_target[:3, :3]
    t_target = T_target[:3, 3]

    # Identity transform
    R_start = np.eye(3)
    t_start = np.zeros(3)

    # Interpolate rotation using Rotation objects and Slerp
    rot_start = Rotation.from_matrix(R_start)
    rot_end = Rotation.from_matrix(R_target)
    slerp = Slerp([0, 1], Rotation.concatenate([rot_start, rot_end]))
    times = np.linspace(0, 1, n_frames)
    rot_interp = slerp(times)

    # Interpolate translation
    t_interp = np.linspace(t_start, t_target, n_frames)

    # Assemble back into a sequence of 4x4 affine matrices
    transforms = []
    for R, t in zip(rot_interp.as_matrix(), t_interp):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        transforms.append(T)

    return transforms


# -----------------------------------------------------------------------------
# Geometry utils — *new*: estimate rotation to align normal with y axis
# -----------------------------------------------------------------------------

def estimate_rotation_to_align_normal_with_y(calibration_points):
    # Fit plane z = ax + by + c
    A = np.c_[calibration_points[:, 0], calibration_points[:, 1], np.ones(len(calibration_points))]
    b = calibration_points[:, 2]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, _ = coeffs

    # Plane normal vector (-a, -b, 1)
    normal = np.array([-a, -b_coef, 1.0])
    normal = normal / np.linalg.norm(normal)

    # Target direction: positive y-axis
    target = np.array([0.0, 1.0, 0.0])

    # Compute rotation axis (cross product) and angle (dot product)
    axis = np.cross(normal, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.eye(3)  # already aligned, no rotation needed

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))

    # Rodrigues' formula to construct rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R  # 3x3 rotation matrix

# -----------------------------------------------------------------------------
# Geometry utils — *new*: estimate x rotation to flatten the plane
# -----------------------------------------------------------------------------
def estimate_x_rotation_to_flatten(calibration_points):
    # Build matrix A and vector b such that z = ax + by + c
    A = np.c_[calibration_points[:, 0], calibration_points[:, 1], np.ones(len(calibration_points))]
    b = calibration_points[:, 2]

    # Least-squares fit of plane coefficients
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, c = coeffs

    # Normal vector (plane Ax + By + Cz + D = 0 has normal (-a, -b, 1))
    normal = np.array([-a, -b_coef, 1.0])
    normal = normal / np.linalg.norm(normal)

    # Extract rotation angle around x-axis (angle of normal in the yz-plane)
    ny, nz = normal[1], normal[2]
    theta_rad = np.arctan2(ny, nz)
    theta_deg = np.degrees(theta_rad)

    # Extract rotation matrix to align normal with y-axis


    return theta_deg  # positive means clockwise rotation around x-axis by theta degrees
# -----------------------------------------------------------------------------
# I/O helpers (npy / ply)
# -----------------------------------------------------------------------------

def _load_frame(fp: str) -> np.ndarray:
    """Load a single part frame stored either as .npy (dict with `world_points`) or .ply."""
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".npy":
        data = np.load(fp, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
            data = data.item()
        pts  = np.asarray(data["world_points"],      dtype=np.float32)
        conf = np.asarray(data["world_points_conf"], dtype=np.float32).reshape(-1,1)
        return np.hstack([pts, conf])
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(fp, remove_nan_points=True, remove_infinite_points=True)
        conf = np.ones((len(pcd.points),1), dtype=np.float32)
        if hasattr(pcd, "point") and "confidence" in pcd.point:
            conf = np.asarray(pcd.point["confidence"], dtype=np.float32).reshape(-1,1)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return np.hstack([pts, conf])
    else:
        raise RuntimeError(f"Unsupported file ext {fp}")

# -----------------------------------------------------------------------------
# Geometry utils — *new*: robust cleaning & densification
# -----------------------------------------------------------------------------
import numpy as np
from scipy import ndimage

def keep_largest_voxel_connected_cluster(cloud, voxel_size=0.01):
    """
    Keep only points belonging to the largest connected voxel region.

    Parameters
    ----------
    cloud : (N, 3) or (N, 4) ndarray
        Input point cloud (xyz or xyz+attributes), in meters.
    voxel_size : float
        Voxel size (e.g. 0.01 = 1 cm).

    Returns
    -------
    filtered_cloud : ndarray
        Points from the largest connected cluster (preserves original columns).
    """
    pts = cloud[:, :3]

    # 1. Map spatial coordinates to voxel grid
    min_bound = pts.min(axis=0)
    voxel_indices = np.floor((pts - min_bound) / voxel_size).astype(int)

    # 2. Build dense voxel grid from sparse points
    max_idx = voxel_indices.max(axis=0) + 1
    grid = np.zeros(shape=tuple(max_idx), dtype=bool)
    grid[tuple(voxel_indices.T)] = True  # occupied voxel = True

    # 3. Label connected components (26-connectivity)
    labeled, num_features = ndimage.label(grid, structure=np.ones((3,3,3)))

    # 4. Assign each point its cluster label
    point_labels = labeled[tuple(voxel_indices.T)]

    # 5. Find the largest connected component (label with most points)
    largest_label = np.bincount(point_labels).argmax()

    # 6. Keep only points belonging to the largest cluster
    keep_mask = point_labels == largest_label
    return cloud[keep_mask]


def _remove_outliers_and_densify(cloud: np.ndarray,
                                 nb_neighbors: int    = 20,
                                 std_ratio: float     = 2.0,
                                 densify_k: int       = 0,
                                 densify_sigma: float = 0.005) -> np.ndarray:
    """Remove statistical outliers & optionally densify by jitter‑copying points.

    Args:
        cloud: (N,4) array with XYZ + confidence.
        nb_neighbors / std_ratio: Statistical outlier removal parameters.
        densify_k: synthetic samples per *input* point (0 → off).
        densify_sigma: Gaussian σ [m] applied independently to XYZ when cloning.
    Returns:
        Cleaned (and possibly densified) cloud of shape (N',4).
    """
    if cloud.size == 0:
        return cloud.copy()

    # --- Outlier removal -----------------------------------------------------
    cloud_scale = np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0))
    cloud = keep_largest_voxel_connected_cluster(cloud, voxel_size=cloud_scale * 0.01)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:,:3]))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_pts  = cloud[ind, :3]
    inlier_conf = cloud[ind, 3:4]
    
    # --- Densification --------------------------------------------------------
    if densify_k > 0 and len(inlier_pts):
        jitter = np.random.normal(scale=densify_sigma, size=(len(inlier_pts)*densify_k, 3)).astype(np.float32)
        synth_pts  = np.repeat(inlier_pts, densify_k, axis=0) + jitter
        synth_conf = np.repeat(inlier_conf, densify_k, axis=0) * 0.5  # down‑weight synthetic samples
        inlier_pts  = np.vstack([inlier_pts,  synth_pts])
        inlier_conf = np.vstack([inlier_conf, synth_conf])

    return np.hstack([inlier_pts, inlier_conf])

# -----------------------------------------------------------------------------
# Higher‑level pre‑processing of whole sequences
# -----------------------------------------------------------------------------

def preprocess_sequence(seq_raw: List[np.ndarray],
                        voxel: float            = 0.02,
                        nb_neighbors: int       = 20,
                        std_ratio: float        = 2.0,
                        densify_k: int          = 0,
                        densify_sigma: float    = 0.005) -> List[np.ndarray]:
    """Frame‑wise cleaning → densify → *optional* voxel down‑sample.

    Processing order:
        1. Statistical outlier removal.
        2. Synthetic densification (if densify_k > 0).
        3. Voxel grid down‑sampling (if voxel > 0).
    """
    out = []
    for f in seq_raw:
        # 1‑2. clean + densify
        f_proc = _remove_outliers_and_densify(f, nb_neighbors, std_ratio, densify_k, densify_sigma)

        # 3. voxel grid (optional)
        if voxel > 0 and len(f_proc):
            pts, conf = f_proc[:,:3], f_proc[:,3:4]
            grid = np.floor(pts / voxel).astype(np.int32)
            _, idx = np.unique(grid, axis=0, return_index=True)
            f_proc = np.hstack([pts[idx], conf[idx]])
        out.append(f_proc)
    return out

# -----------------------------------------------------------------------------
# Bounding‑box utilities (unchanged)
# -----------------------------------------------------------------------------

def pca_bbox(cloud: np.ndarray) -> BoundingBox:
    pts   = cloud[:,:3]
    cen   = pts.mean(0)
    _, eigvecs = np.linalg.eigh(np.cov(pts.T))
    R = eigvecs[:, ::-1]  # largest eigen‑vec → x‑axis
    proj   = (pts - cen) @ R
    lo,hi  = proj.min(0), proj.max(0)
    extent = hi - lo
    cen_local = 0.5 * (hi + lo)
    return BoundingBox(cen + cen_local @ R.T, extent, R)

# Point‑to‑point ICP (Open3D wrapper)

def estimate_transform(src: np.ndarray,
                       tgt: np.ndarray,
                       max_iter: int = 60,
                       dist: float   = 0.1) -> np.ndarray:
    src_p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[:,:3]))
    tgt_p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt[:,:3]))
    res   = o3d.pipelines.registration.registration_icp(
        src_p, tgt_p, dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return res.transformation

def estimate_transform_from_tracks(obj_id: int,
                                   tracked_points: Dict[int, np.ndarray],
                                   src_frame: int,
                                   tgt_frame: int) -> np.ndarray:
    src_points = tracked_points[obj_id][src_frame]
    tgt_points = tracked_points[obj_id][tgt_frame]
    _, mask_src,_ = remove_outliers_mahalanobis(src_points)
    _, mask_tgt,_ = remove_outliers_mahalanobis(tgt_points)
    mask_all = mask_src & mask_tgt
    print(f'filtered {mask_all.sum()}/{src_points.shape[0]} points')
    src_points = src_points[mask_all]
    tgt_points = tgt_points[mask_all]
    
    k = src_points.shape[0]
    # 1. Center the point sets
    mu_src = src_points.mean(axis=0)
    mu_tgt = tgt_points.mean(axis=0)
    src_centered = src_points - mu_src
    tgt_centered = tgt_points - mu_tgt

    # 2. Compute cross-covariance matrix
    H = src_centered.T @ tgt_centered / k   # (3,3)

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct reflection if determinant is negative
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T


    # 5. Translation
    t = mu_tgt - R @ mu_src

    # 6. Assemble homogeneous matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
    # # Create two Open3D point cloud objects
    # src_pcd = o3d.geometry.PointCloud()
    # tgt_pcd = o3d.geometry.PointCloud()

    # # Set point cloud coordinates
    # src_pcd.points = o3d.utility.Vector3dVector(src_points)
    # tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

    # # Assign a color to each point
    # src_colors = np.zeros((len(src_points), 3))
    # tgt_colors = np.zeros((len(tgt_points), 3))
    # for i in range(len(src_points)):
    #     color = np.random.rand(3)
    #     src_colors[i] = color
    #     tgt_colors[i] = color

    # src_pcd.colors = o3d.utility.Vector3dVector(src_colors)
    # tgt_pcd.colors = o3d.utility.Vector3dVector(tgt_colors)

    # # Save point clouds
    # o3d.io.write_point_cloud(f"src_frame_{src_frame}.ply", src_pcd)
    # o3d.io.write_point_cloud(f"tgt_frame_{tgt_frame}.ply", tgt_pcd)
    
    # breakpoint()
    # return 

def compute_bbox_sequence(
                            obj_id: int,
                            seq_proc: List[np.ndarray],
                            tracked_points: Dict[int, np.ndarray],
                            ref_frame: int) -> List[BoundingBox]:
    # choose reference frame with most confidence
    if ref_frame is None:
        ref_idx   = int(np.argmax([f[:,3].sum() for f in seq_proc]))
    else:
        ref_idx = ref_frame
    print(f"ref_idx: {ref_idx}")
    ref_cloud = seq_proc[ref_idx]
    bbox_ref  = pca_bbox(ref_cloud)
    # Ts = [np.eye(4) if i == ref_idx else estimate_transform(ref_cloud, f)
    #       for i, f in enumerate(seq_proc)]
    Ts = [np.eye(4) if i == ref_idx else estimate_transform_from_tracks(obj_id, tracked_points, ref_idx, i)
          for i, f in enumerate(seq_proc)]
    boxes = []
    for T in Ts:
        boxes.append(BoundingBox(T[:3,:3] @ bbox_ref.center + T[:3,3],
                                 bbox_ref.extent.copy(),
                                 T[:3,:3] @ bbox_ref.rotation))
    return (boxes, Ts)

# -----------------------------------------------------------------------------
# Multi‑part orchestration
# -----------------------------------------------------------------------------

def load_multi(dir_path: str,
               part_ids: List[int],
               frame_range: Tuple[int,int]) -> Dict[int,List[np.ndarray]]:
    start, end = frame_range
    parts_data = {pid: [] for pid in part_ids}
    for idx in range(start, end + 1):
        for pid in part_ids:
            pattern = os.path.join(dir_path, f"{idx:06d}_{pid}.*")
            files   = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(pattern)
            parts_data[pid].append(_load_frame(files[0]))
    return parts_data

# -----------------------------------------------------------------------------
# Rendering (fixed camera) — unchanged
# -----------------------------------------------------------------------------

def render_multi(parts_clouds: Dict[int,List[np.ndarray]],
                 parts_boxes:  Dict[int,List[BoundingBox]],
                 out_dir:      str,
                 out_bboxs:    str,
                 colours:      Dict[int,Tuple[float,float,float]],
                 cam_front:    np.ndarray,
                 cam_up:       np.ndarray,
                 cam_dist:     float,
                 visibility_per_obj: Optional[Dict[int, np.ndarray]] = None,
                 width:  int = 1280,
                 height: int = 720,
                 bg=(1,1,1,1),
                 save_vis: bool = False):
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(out_bboxs, exist_ok=True)

    n_frames = len(next(iter(parts_clouds.values())))

    all_bboxs = []
    if save_vis:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        scene    = renderer.scene
        scene.set_background(bg)

        # materials
        mat      = o3d.visualization.rendering.MaterialRecord(); mat.shader      = "defaultUnlit"
        line_mat = o3d.visualization.rendering.MaterialRecord(); line_mat.shader = "unlitLine"; line_mat.line_width = 2.0
        lines    = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]

        # determine static camera centre (global bbox of reference frame 0)
        all_pts = np.vstack([parts_clouds[pid][0][:,:3] for pid in parts_clouds])
        center  = all_pts.mean(0)
        diag    = np.linalg.norm(all_pts.ptp(0))
        eye     = center - cam_front/np.linalg.norm(cam_front) * cam_dist * diag
        for k in tqdm(range(n_frames)):
            bboxs = []
            scene.clear_geometry()
            # add xyz axis
            x_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            scene.add_geometry("x_axis", x_axis, mat)
            y_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            scene.add_geometry("y_axis", y_axis, mat)
            z_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            scene.add_geometry("z_axis", z_axis, mat)
            for pid, color in colours.items():
                cloud = parts_clouds[pid][k]
                pcd   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:,:3]))
                pcd.paint_uniform_color(color)
                scene.add_geometry(f"pc{pid}", pcd, mat)

                box   = parts_boxes[pid][k]
                # check visibility: if all tracked points are invisible, set bbox to zeros
                if visibility_per_obj is not None and pid in visibility_per_obj:
                    if not visibility_per_obj[pid][k].any():
                        # all points invisible for this part at this frame
                        bboxs.append(np.zeros((8, 3)))
                        continue
                bboxs.append(box.corners())
                line  = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(box.corners())
                line.lines  = o3d.utility.Vector2iVector(lines)
                line.colors = o3d.utility.Vector3dVector([color] * len(lines))
                scene.add_geometry(f"bb{pid}", line, line_mat)

            scene.camera.look_at(center, eye, cam_up)
            img = renderer.render_to_image()
            o3d.io.write_image(os.path.join(out_dir, f"{k:06d}.png"), img)
            all_bboxs.append(np.array(bboxs))
    else:
        for k in tqdm(range(n_frames)):
            bboxs = []
            for pid, color in colours.items():
                box   = parts_boxes[pid][k]
                # check visibility: if all tracked points are invisible, set bbox to zeros
                if visibility_per_obj is not None and pid in visibility_per_obj:
                    if not visibility_per_obj[pid][k].any():
                        # all points invisible for this part at this frame
                        bboxs.append(np.zeros((8, 3)))
                        continue
                bboxs.append(box.corners())
            all_bboxs.append(np.array(bboxs))
    all_bboxs = np.array(all_bboxs)
    np.save(os.path.join(out_bboxs, "bboxs.npy"), np.array(all_bboxs))
    print("Render complete →", out_dir)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    import argparse, textwrap
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""\
            Compute per‑part bounding‑box sequences and render them headlessly.
            Includes optional statistical outlier removal and synthetic densification.
        """))
    parser.add_argument("--dir",      default="working_dir/point_maps_segmented", help="Directory with {frame}_{part}.npy|ply files")
    parser.add_argument("--frames",   default="0,-1", help="start,end frame indices (inclusive)")
    parser.add_argument("--ref_frame", type=int, default=0, help="Reference frame")
    parser.add_argument("--inter_fn", type=int, default=0, help="interpolation frame number")
    # --- new pre‑processing flags -------------------------------------------
    parser.add_argument("--outlier_nb",    type=int,   default=3,   help="Statistical outlier removal: nb_neighbors")
    parser.add_argument("--outlier_std",   type=float, default=10,  help="Statistical outlier removal: std_ratio")
    parser.add_argument("--densify_k",     type=int,   default=0,    help="Synthetic points per input point (0 = off)")
    parser.add_argument("--densify_sigma", type=float, default=0.0005,help="Gaussian jitter [m] for densify")
    parser.add_argument("--voxel",         type=float, default=0.02,  help="Voxel size [m] after densify (0 = off)")
    # ------------------------------------------------------------------------

    parser.add_argument("--cam_front", default="0,0,-1", help="Camera front vector")
    parser.add_argument("--cam_up",    default="0,1,0",  help="Camera up vector")
    parser.add_argument("--cam_dist",  type=float, default=1.5, help="Distance multiplier vs diag")
    parser.add_argument("--save_vis", type=bool, default=False, help="Save vis images")
    args = parser.parse_args()

    working_dir = os.path.dirname(args.dir)
    with open(os.path.join(working_dir, "clicks.json"), "r") as f:
        clicks = json.load(f)
    part_ids = list(clicks.keys())
    part_ids = [int(x) for x in part_ids]
    out_dir     = os.path.join(working_dir, "vis_bboxs")
    out_bboxs   = os.path.join(working_dir, "bboxs")

    start, end  = [int(x) for x in args.frames.split(',')]
    if end < 0:
        # auto‑detect last frame by scanning first part ID
        files       = glob.glob(os.path.join(args.dir, f"*_{part_ids[0]}.*"))
        frame_nums  = sorted({int(os.path.basename(f).split('_')[0]) for f in files})
        end         = frame_nums[-1]
    frame_range = (start, end)

    # ---------------------------------------------------------------------
    print("[INFO] loading data …")
    parts_raw  = load_multi(args.dir, part_ids, frame_range)
    # load the tracked points
    tracked_points = np.load(os.path.join(working_dir, "tracked_points.npy"), allow_pickle=True).item()

    # load visibility data from tracks.npy
    tracks_data = np.load(os.path.join(working_dir, "tracks.npy"), allow_pickle=True).item()
    pred_visibility = tracks_data['pred_visibility']  # shape: [1, n_frames, n_points]
    obj_id_nums = tracks_data['obj_id_nums']  # dict mapping obj_id -> num_points

    # extract per-object visibility
    visibility_per_obj = {}
    acc_id = 0
    for obj_id in obj_id_nums.keys():
        obj_id_end = acc_id + obj_id_nums[obj_id]
        visibility_per_obj[obj_id] = pred_visibility[0, :, acc_id:obj_id_end]  # shape: [n_frames, n_points_for_obj]
        acc_id = obj_id_end
    
    print("[INFO] preprocessing … (outlier removal + densify + voxel)")
    parts_proc = {pid: preprocess_sequence(seq,
                                           voxel=args.voxel,
                                           nb_neighbors=args.outlier_nb,
                                           std_ratio=args.outlier_std,
                                           densify_k=args.densify_k,
                                           densify_sigma=args.densify_sigma)
                  for pid, seq in parts_raw.items()}
    theta_deg = None
    if 0 in part_ids:
        
        # 0 is the calibration plane, we first calculate a global rotation along 
        # x axis based on part 0, then apply this rotation to all other parts,
        # then remove the part 0 from the parts_proc
        
        calibration_points = parts_proc[0][0][:,:3]
        # calculate the rotation matrix along x axis
        R = estimate_rotation_to_align_normal_with_y(calibration_points)
        
        #apply the rotation to all other parts
        for pid in part_ids:
            for cloud in parts_proc[pid]:
                cloud[:,:3] = cloud[:,:3] @ R.T
        
        # remove the part 0 from the parts_proc
        parts_proc.pop(0)
        part_ids.pop(0)
        tracked_points.pop(0)
        tracked_points = {pid: tracked_points[pid]@R.T for pid in part_ids}
    
    print("[INFO] computing bbox sequences …")
    bboxs_Ts = {pid: compute_bbox_sequence(pid, seq, tracked_points, ref_frame=args.ref_frame) for pid, seq in parts_proc.items()}
    parts_boxes = {pid: bboxs_Ts[pid][0] for pid in part_ids}
    parts_Ts = {pid: bboxs_Ts[pid][1] for pid in part_ids}
    
    # simple colour table
    palette = plt.cm.rainbow(np.linspace(0, 1, len(part_ids)))[:, :3]  # generate distinct colors using matplotlib rainbow colormap
    colours = {pid: palette[i % len(palette)] for i, pid in enumerate(part_ids)}

    print("[INFO] rendering …")
    front = np.array([float(x) for x in args.cam_front.split(',')])
    up    = np.array([float(x) for x in args.cam_up.split(',')])
    

    if args.inter_fn > 0:
        new_parts_proc = parts_proc.copy()
        new_parts_boxes = parts_boxes.copy()
        #interpolate between the bboxs
        for pid in part_ids:
            points_seq = parts_proc[pid].copy()
            bboxs_seq = parts_boxes[pid].copy()
            ts_seq = parts_Ts[pid].copy()
            new_points_seq = []
            new_bboxs_seq = []
            for i in range(len(points_seq)-1):
                points1 = points_seq[i]
                points2 = points_seq[i+1]
                bbox1 = bboxs_seq[i]
                bbox2 = bboxs_seq[i+1]
                ts1 = ts_seq[i]
                ts2 = ts_seq[i+1]
                ts12 = ts2 @ np.linalg.inv(ts1)
                ts_interp = interpolate_se3_geodesic(ts12, args.inter_fn)
                new_points_seq.append(points1)
                new_bboxs_seq.append(bbox1)
                for j in range(args.inter_fn):
                    transform = ts_interp[j]
                    new_points = points1[:,:3] @ transform[:3,:3].T + transform[:3,3]
                    new_points = np.concatenate([new_points, points1[:,3:]], axis=1)
                    new_bbox = BoundingBox(transform[:3,:3] @ bbox1.center + transform[:3,3], 
                                           bbox1.extent, 
                                           transform[:3,:3] @ bbox1.rotation)
                    new_points_seq.append(new_points)
                    new_bboxs_seq.append(new_bbox)
                new_points_seq.append(points2)
                new_bboxs_seq.append(bbox2)
            # breakpoint()
            new_parts_boxes[pid] = new_bboxs_seq
            new_parts_proc[pid] = new_points_seq
        parts_proc = new_parts_proc
        parts_boxes = new_parts_boxes
    render_multi(parts_proc, parts_boxes, out_dir, out_bboxs, colours, front, up, args.cam_dist,
                 visibility_per_obj=visibility_per_obj, save_vis=args.save_vis)          
        # load the keyframing bboxs
        
if __name__ == "__main__":
    main()
