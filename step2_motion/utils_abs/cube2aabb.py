import numpy as np

def cubes_to_target_aabb(verts,                # (k, 8, 3) float32/64
                         target_center,        # (3,) 目标盒中心
                         target_width):        # (3,) 目标盒全宽 (Wx,Wy,Wz)
    """
    将一批立方体的顶点经 PCA 有向包围盒归一化后，映射到给定目标 AABB。
    
    Parameters
    ----------
    verts : ndarray, shape (k,8,3)
        所有立方体的 8 个顶点坐标，世界/统一坐标系。
    target_center : array-like, shape (3,)
        目标盒中心坐标 (cx, cy, cz)。
    target_width  : array-like, shape (3,)
        目标盒沿 (x, y, z) 轴的“全宽” (W = max-min)；若你只有半宽，请乘 2。
    
    Returns
    -------
    out : ndarray, shape (k,8,3)
        经过对齐、缩放、平移后的顶点数组，顺序与输入一致。
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)      # (N,3)
    if v.size == 0:
        raise ValueError("verts 不能为空")
    
    # ---------- 1. PCA 主轴 ----------
    mu = v.mean(axis=0)                                         # 质心 μ
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)                      # 升序
    idx = eigvals.argsort()[::-1]                               # 降序
    R = eigvecs[:, idx]                                         # 3×3 正交基（列向量）
    if np.linalg.det(R) < 0:                                    # 保证右手系
        R[:, 2] *= -1
    
    # ---------- 2. 在 PCA 坐标系下求半轴长 ----------
    local = (v - mu) @ R                                        # (N,3)
    min3, max3 = local.min(0), local.max(0)
    extents_src = (max3 - min3) * 0.5    # (ex,ey,ez) 半宽 (≥0)
    
    # ---------- 3. 计算仿射变换 ----------
    target_center = np.asarray(target_center, dtype=np.float64)
    target_width  = np.asarray(target_width , dtype=np.float64)
    if np.any(extents_src == 0):
        raise ValueError("源数据在某方向上退化到 0，无法缩放")
    
    S = (target_width * 0.5) / extents_src          # 各轴缩放系数
    # 齐次 4×4 矩阵（右乘列向量）：
    #   M = T(c_t) · S · Rᵀ · T(-μ)
    M_lin  = (R * S).T                              # 3×3 线性部 = diag(S)·Rᵀ
    M_trans = target_center - M_lin @ mu            # 平移部
    
    # ---------- 4. 变换所有点 ----------
    v_out = (v @ M_lin.T) + M_trans                 # (N,3)
    return v_out.reshape(verts.shape), M_lin, M_trans
