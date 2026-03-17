import numpy as np

def cubes_to_target_aabb(verts,                # (k, 8, 3) float32/64
                         target_center,        # (3,) target box center
                         target_width):        # (3,) target box full width (Wx,Wy,Wz)
    “””
    Map a batch of cube vertices to a target AABB via PCA-based OBB normalization.

    Parameters
    ----------
    verts : ndarray, shape (k,8,3)
        8 vertex coordinates of all cubes in world/unified coordinates.
    target_center : array-like, shape (3,)
        Target box center (cx, cy, cz).
    target_width  : array-like, shape (3,)
        Target box full width along (x, y, z) axes (W = max-min); multiply by 2 if you only have half-widths.

    Returns
    -------
    out : ndarray, shape (k,8,3)
        Aligned, scaled, and translated vertex array, same order as input.
    “””
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)      # (N,3)
    if v.size == 0:
        raise ValueError(“verts must not be empty”)

    # ---------- 1. PCA principal axes ----------
    mu = v.mean(axis=0)                                         # centroid
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)                      # ascending order
    idx = eigvals.argsort()[::-1]                               # descending order
    R = eigvecs[:, idx]                                         # 3x3 orthogonal basis (column vectors)
    if np.linalg.det(R) < 0:                                    # ensure right-handed system
        R[:, 2] *= -1

    # ---------- 2. Compute semi-axis lengths in PCA frame ----------
    local = (v - mu) @ R                                        # (N,3)
    min3, max3 = local.min(0), local.max(0)
    extents_src = (max3 - min3) * 0.5    # (ex,ey,ez) half-widths (>=0)

    # ---------- 3. Compute affine transform ----------
    target_center = np.asarray(target_center, dtype=np.float64)
    target_width  = np.asarray(target_width , dtype=np.float64)
    if np.any(extents_src == 0):
        raise ValueError(“Source data is degenerate (zero extent) along some axis, cannot scale”)

    S = (target_width * 0.5) / extents_src          # per-axis scale factors
    # Homogeneous 4x4 matrix (right-multiply column vectors):
    #   M = T(c_t) · S · Rᵀ · T(-μ)
    M_lin  = (R * S).T                              # 3x3 linear part = diag(S)·Rᵀ
    M_trans = target_center - M_lin @ mu            # translation part

    # ---------- 4. Transform all points ----------
    v_out = (v @ M_lin.T) + M_trans                 # (N,3)
    return v_out.reshape(verts.shape), M_lin, M_trans
