# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import pdb
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from copy import deepcopy
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_motion_extract_data
import shutil
from data_loaders.tensors import collate, lengths_to_mask
from utils.text_control_example import collate_all
from os.path import join as pjoin

def rotate_x(points: np.ndarray, angle_deg: float) -> np.ndarray:
                """
                全局绕 x 轴旋转。
                
                Parameters
                ----------
                points : np.ndarray
                    形状为 (k, 3) 的点集，每行一个三维坐标 (x, y, z)。
                angle_deg : float
                    旋转角度（度）。
                
                Returns
                -------
                np.ndarray
                    旋转后的点集，形状仍为 (k, 3)。
                """
                
                theta = np.deg2rad(angle_deg)          # 度 → 弧度
                c, s   = np.cos(theta), np.sin(theta)   # 余弦、正弦
                
                # 绕 x 轴的旋转矩阵
                R = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]], dtype=points.dtype)
                
                # (k, 3) @ (3, 3)^T → (k, 3)
                return points @ R.T
            
def rotate_y(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)          # 度 → 弧度
    c, s   = np.cos(theta), np.sin(theta)   # 余弦、正弦
    
    # 绕 y 轴的旋转矩阵
    R = np.array([[c, 0, s],
                 [0, 1, 0],
                 [-s, 0, c]], dtype=points.dtype)
    return points @ R.T

def rotate_z(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)          # 度 → 弧度
    c, s   = np.cos(theta), np.sin(theta)   # 余弦、正弦
    
    # 绕 z 轴的旋转矩阵
    R = np.array([[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]], dtype=points.dtype)
    return points @ R.T


def rotate_hint_y(hint, angle_deg):
    # hint: [t, cube_num, 8, 3]
    # rotate hint around y axis
    t, cube_num, _, _ = hint.shape
    hint = hint.reshape(-1, 3)
    hint = rotate_y(hint, angle_deg)
    hint = hint.reshape(t, cube_num, 8, 3)
    return hint

def normalize_hint_dir(hint, extra_angle=0):
    # normalize hint dir to be z+ axis
    current_hint_start_center = hint[0].reshape(-1, 3)
    current_hint_start_center = np.mean(current_hint_start_center, axis=0)
    
    current_hint_end_center = hint[-1].reshape(-1, 3)
    current_hint_end_center = np.mean(current_hint_end_center, axis=0)
    current_hint_dir = current_hint_end_center - current_hint_start_center
    current_hint_dir = current_hint_dir / np.linalg.norm(current_hint_dir)
    current_hint_dir_angle_deg = np.arctan2(current_hint_dir[2], current_hint_dir[0]) * 180 / np.pi

    angle_to_rotate = current_hint_dir_angle_deg - 90 + extra_angle
    hint = rotate_hint_y(hint, angle_to_rotate)
    return hint


def normalize_hint(hint, torso_idx=3, target_height=1.3, y_offset=0.3,
                   extra_angle=0.0, camera_rot_x=-10.0, align_direction=True):
    """Rotate then center — guarantees torso at xz origin."""
    hint = hint.copy()

    # 1. Direction alignment + variant rotation (Y-axis)
    if align_direction:
        start = hint[0].reshape(-1, 3).mean(axis=0)
        end = hint[-1].reshape(-1, 3).mean(axis=0)
        d = end - start
        align_angle = np.arctan2(d[2], d[0]) * 180 / np.pi - 90
    else:
        align_angle = 0
    hint = rotate_hint_y(hint, align_angle + extra_angle)

    # 2. Camera correction (X-axis)
    if camera_rot_x != 0:
        t, cn, _, _ = hint.shape
        hint = rotate_x(hint.reshape(-1, 3), camera_rot_x).reshape(t, cn, 8, 3)

    # 3. Center torso at xz origin + scale (on ROTATED data)
    center_full, extent_full = get_obb(hint)
    center_torso, _ = get_obb_id(hint, torso_idx)
    translation = np.array([-center_torso[0],
                            extent_full[1]/2 - center_full[1],
                            -center_torso[2]])
    hint = (hint + translation) * (target_height / extent_full[1])
    hint += np.array([0, y_offset, 0])
    return hint


def generate_rotation_variants(hint, num_variants=10, **norm_kwargs):
    variants = []
    for i in range(num_variants):
        angle = i * (360.0 / num_variants)
        v = normalize_hint(hint, extra_angle=angle, **norm_kwargs)
        variants.append(v[np.newaxis])
    return np.concatenate(variants, axis=0)  # [N, T, cube_num, 8, 3]


def get_obb(hint):
    hint0 = hint[0]
    points = hint0.reshape(-1,3)
    center = np.mean(points, axis=0)
    points_centered = points - center
    extent = np.max(points_centered, axis=0) - np.min(points_centered, axis=0)
    return center, extent
def get_obb_id(hint, idx):
    hint0 = hint[0]
    points = hint0[idx]
    center = np.mean(points, axis=0)
    points_centered = points - center
    extent = np.max(points_centered, axis=0) - np.min(points_centered, axis=0)
    return center, extent
def main():
    args = generate_args()
    if args.no_dataset:
        assert args.text_prompt, "--no_dataset requires --text_prompt"
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    # data loader pads hint/mask to max_frames; n_frames must match.
    # When --hint_path is given, n_frames gets overridden by the hint length later.
    n_frames = max_frames
    is_using_data = not any([args.text_prompt])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
    if args.cfg_scale != 1:
        out_path += '_cfg' + str(args.cfg_scale)
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    if args.no_dataset:
        print('Skipping dataset loading (--no_dataset mode)...')
        mean = np.load('./dataset/HumanML3D/Mean.npy')
        std = np.load('./dataset/HumanML3D/Std.npy')
        data = None
    else:
        print('Loading dataset...')
        data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    
    if args.cfg_scale != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    
    if args.no_dataset:
        lengths = torch.full((args.batch_size,), n_frames, dtype=torch.long, device=dist_util.dev())
        mask = lengths_to_mask(lengths, n_frames).unsqueeze(1).unsqueeze(1)
        model_kwargs = {
            'y': {
                'mask': mask,
                'lengths': lengths,
                'text': [args.text_prompt] * args.batch_size,
            }
        }
    else:
        iterator = iter(data)
        gt_motion, model_kwargs = next(iterator)
        for k, v in model_kwargs['y'].items():
            if torch.is_tensor(v):
                model_kwargs['y'][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []
    all_hint_for_vis = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.cfg_scale != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.cfg_scale
        if args.hint_path:
            hint_raw = np.load(args.hint_path)
            if hint_raw.shape[0] > n_frames:
                hint_raw = hint_raw[:n_frames]

            norm_kwargs = dict(
                torso_idx=args.torso_idx,
                target_height=args.target_height,
                y_offset=args.y_offset,
                camera_rot_x=args.camera_rot_x,
                align_direction=args.align_direction,
            )

            if args.rotation_idx >= 0:
                # Fixed-rotation mode: same angle for all samples
                angle = args.rotation_idx * (360.0 / args.num_rotations)
                single = normalize_hint(hint_raw, extra_angle=angle, **norm_kwargs)
                variants = np.repeat(single[np.newaxis], args.batch_size, axis=0)
            else:
                # Sweep mode: each sample gets a different rotation
                variants = generate_rotation_variants(
                    hint_raw, num_variants=args.batch_size, **norm_kwargs)

            n_frames = variants.shape[1]
            hint = variants.reshape(variants.shape[0], variants.shape[1], -1)
            vec_len = hint.shape[2]
            hint = np.pad(hint, ((0, 0), (0, n_frames - hint.shape[1]), (0, 552 - vec_len)),
                          mode='constant', constant_values=0)
            model_kwargs['y']['hint'] = torch.tensor(hint, device='cuda:0', dtype=torch.float)

            if args.text_prompt:
                model_kwargs['y']['text'] = [args.text_prompt] * len(model_kwargs['y']['text'])

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample = sample[:, :263]
        
        # sample = gt_motion
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            if args.no_dataset:
                sample = (sample.cpu().permute(0, 2, 3, 1) * std + mean).float()
            else:
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        
        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]
            
            if 'hint' in model_kwargs['y']:
                hint = model_kwargs['y']['hint']
                hint = hint.reshape(hint.shape[0], hint.shape[1], -1, 8, 3)
                
                all_hint.append(hint.data.cpu().numpy())
                # hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3)
                all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if 'hint' in model_kwargs['y']:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths, "hint": all_hint_for_vis,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            if 'hint' in model_kwargs['y']:
                hint = all_hint_for_vis[rep_i*args.batch_size + sample_i]
            else:
                hint = None
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            vis_hint = hint if args.vis_bbox else None
            plot_3d_motion_extract_data(animation_save_path, skeleton, motion, dataset=args.dataset, title='', fps=fps, hint=hint, export_data_path=os.path.join(out_path, f'motion_data_{sample_i}_{rep_i}.pkl'))
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title='', fps=fps, hint=vis_hint)
            # plot_3d_motion(animation_save_path.replace('.mp4', '_hint.mp4'), skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint, hintOnly=True)
            # plot_3d_motion(animation_save_path.replace('.mp4', '_line.mp4'), skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint, lineOnly=True)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        # sample_files = save_multiple_samples(args, out_path,
        #                                        row_print_template, all_print_template, row_file_template, all_file_template,
        #                                        caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
