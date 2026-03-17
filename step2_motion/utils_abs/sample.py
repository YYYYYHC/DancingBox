import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools
import os
import random

def sample_abstraction_all(abstraction_all):
    '''
    given a abstraction_all, sample a combination of abstraction
    
    return: cubes (cube_num, 8, 3) 
    '''
    center_num = len(abstraction_all)
    center_idx = np.random.randint(0, center_num)
    abstraction_center = abstraction_all[center_idx]
    comb_num = len(abstraction_center)
    comb_idx = np.random.randint(0, comb_num)
    cubes = abstraction_center[comb_idx]
    return cubes

def sample_abstraction(abstraction_dict, center_mode=-1):
    '''
    given a abstraction_dict, sample a combination of abstraction
    
    return: cubes (cube_num, 8, 3) and idxs list [[cube1 idxs 1, cube1 idxs 2, ...], [cube2 idxs 1, cube2 idxs 2, ...], ...]
    '''
    assert center_mode in [1, 2, 3, 4]
    all_cube_branches = abstraction_dict[f"center_mode_{center_mode:02d}"]['all_cube_branches']
    all_branches_idxs = abstraction_dict[f"center_mode_{center_mode:02d}"]['all_cube_branches_idxs']
    # sample a centering idx
    center_idx = np.random.randint(0, len(all_cube_branches))
    #cube_branches: (branch_num, sample_comb_num, cube_num, 8, 3)    
    cube_branches = all_cube_branches[center_idx]
    branches_idxs = all_branches_idxs[center_idx]
    final_cubes = []
    final_idxs = []
    for branch_idx, branch in enumerate(cube_branches):
        if len(branch) == 0:
            continue
        # sample a sample_comb_idx
        sample_comb_idx = np.random.randint(0, len(branch))
        selected_cube = branch[sample_comb_idx]
        selected_idx = branches_idxs[branch_idx][sample_comb_idx]
        final_cubes.append(selected_cube)
        final_idxs+=(selected_idx)
    if len(final_cubes) != 0:
        final_cubes = np.concatenate(final_cubes, axis=0)
    return final_cubes, final_idxs

def sample_all_abstraction(abstraction_dict, center_mode=-1, max_num1=None, max_num2=None):
    '''
    given a abstraction_dict, sample a combination of abstraction
    
    return: cubes (cube_num, 8, 3) and idxs list [[cube1 idxs 1, cube1 idxs 2, ...], [cube2 idxs 1, cube2 idxs 2, ...], ...]
    '''
    assert center_mode in [1, 2, 3, 4, 5]
    all_cube_branches = abstraction_dict[f"center_mode_{center_mode:02d}"]['all_cube_branches']
    all_branches_idxs = abstraction_dict[f"center_mode_{center_mode:02d}"]['all_cube_branches_idxs']
    # 随机选择max_num个中心索引
    all_final_cubes = []
    all_final_idxs = []
    center_indices = list(range(len(all_cube_branches)))
    if max_num1 is not None and len(center_indices) > max_num1:
        center_indices = random.sample(center_indices, max_num1)
    for center_idx in center_indices:
    
        #cube_branches: (branch_num, sample_comb_num, cube_num, 8, 3)    
        cube_branches = all_cube_branches[center_idx]
        branches_idxs = all_branches_idxs[center_idx]
        
        branch_comb_num = [range(len(branch)) for branch in cube_branches]
        all_comb = list(itertools.product(*branch_comb_num))
        
        #random sample max_num combinations
        if max_num2 is not None:
            if len(all_comb) > max_num2:
                all_comb = random.sample(all_comb, max_num2)
        for comb in all_comb:
            comb_list = list(comb)
            final_cubes = []
            final_idxs = []
            for branch_idx, branch in enumerate(cube_branches):
                if len(branch) == 0:
                    continue
                sample_comb_idx = comb_list[branch_idx]
                selected_cube = branch[sample_comb_idx]
                selected_idx = branches_idxs[branch_idx][sample_comb_idx]
                final_cubes.append(selected_cube)
                final_idxs+=(selected_idx)
            if len(final_cubes) != 0:
                final_cubes = np.concatenate(final_cubes, axis=0)
            all_final_cubes.append(final_cubes)
            all_final_idxs.append(final_idxs)
    return all_final_cubes, all_final_idxs

def apply_random_transformation_full(data, rotation_strength=np.pi, translation_strength=1.0,
                                    protion_time = 0.2, num_cube = 1):
    '''
    data: (t, n, 8, 3)
    protion_time: 随机选择x%的时间点
    num_cube: 随机选择x个cube
    '''
    t, n, _, _ = data.shape
    time_indices = np.array(np.random.choice(t, size=int(t*protion_time), replace=False))
    cube_nums = [len(data[time_idx]) for time_idx in time_indices]
    cube_indices = np.array([np.random.choice(cube_nums[i], size=num_cube, replace=False)[0] for i in range(len(time_indices))])
    data[time_indices, cube_indices] = apply_random_transformation_single(data[time_indices, cube_indices], rotation_strength, translation_strength)
    return data

def apply_random_global_scaling_translation(data, scaling_strength=1.0, translation_strength=1.0):
    t, n, _, _ = data.shape
    scaling_factors = np.random.uniform(1-scaling_strength, 1+scaling_strength, size=(1,))
    translation_vectors = np.random.randn( 3) * translation_strength
    data = data * scaling_factors + translation_vectors
    return data

def apply_random_transformation_single(data, rotation_strength=np.pi, translation_strength=1.0):
    """
    对形状为 (n, 8, 3) 的物体组，以各自中心为旋转点施加随机旋转与平移。

    参数:
        data: np.ndarray, shape = (n, 8, 3)
        rotation_strength: float, 最大旋转角度（弧度）
        translation_strength: float, 最大平移范围

    返回:
        transformed: np.ndarray, shape = (n, 8, 3)
    """
    n = data.shape[0]

    # Step 1: 计算每个物体的中心
    centers = np.mean(data, axis=1, keepdims=True)  # (n, 1, 3)

    # Step 2: 居中
    centered_data = data - centers  # (n, 8, 3)

    # Step 3: 生成旋转矩阵
    axes = np.random.randn(n, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8
    angles = np.random.uniform(-rotation_strength, rotation_strength, size=(n,))
    rotvecs = axes * angles[:, None]
    rot_mats = R.from_rotvec(rotvecs).as_matrix()  # (n, 3, 3)

    # Step 4: 应用旋转
    rotated = np.einsum('nij,nkj->nki', rot_mats, centered_data)  # (n, 8, 3)

    # Step 5: 加回中心
    recentered = rotated + centers  # (n, 8, 3)

    # Step 6: 添加随机平移
    translations = np.random.uniform(-translation_strength, translation_strength, size=(n, 1, 3))
    transformed = recentered + translations  # (n, 8, 3)

    return transformed

def apply_random_drop(cubes, idxs):
    k = cubes.shape[0]
    # 随机选择要删除的数量，范围为 [1, k//2]
    drop_num = np.random.randint(1, k // 2 + 1) if k > 1 else 0
    
    # 在 [0, k-1] 中随机选择 drop_num 个索引
    drop_indices = np.random.choice(k, size=drop_num, replace=False)
    
    # 生成保留的索引
    keep_indices = np.array([i for i in range(k) if i not in drop_indices])
    
    cubes_kept = cubes[keep_indices]
    idxs_kept = [idxs[i] for i in keep_indices]
    
    return cubes_kept, idxs_kept

def apply_random_drop_full(data):
    '''
    data: (t, n, 8, 3)
    随机
    '''
    k = data.shape[1]
    # 随机选择要删除的数量，范围为 [1, k//6], 对于少于或等于6个cube的，不删除
    drop_num = np.random.randint(1, k // 2 + 1) if k > 2 else 0
    
    # 在 [0, k-1] 中随机选择 drop_num 个索引
    drop_indices = np.random.choice(k, size=drop_num, replace=False)
    
    # 生成保留的索引
    keep_indices = np.array([i for i in range(k) if i not in drop_indices])
    
    data_kept = data[:, keep_indices]
    return data_kept