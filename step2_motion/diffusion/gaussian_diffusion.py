# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from os.path import join as pjoin
import torch.nn.functional as F

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        dataset='humanml',
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

        if dataset == 'humanml':
            spatial_norm_path = './dataset/humanml_spatial_norm'
            data_root = './dataset/HumanML3D'
        elif dataset == 'kit':
            spatial_norm_path = './dataset/kit_spatial_norm'
            data_root = './dataset/KIT-ML'
        else:
            raise NotImplementedError('Dataset not recognized!!')
        self.raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy')))
        self.raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy')))
        self.mean = torch.from_numpy(np.load(pjoin(data_root, 'Mean.npy'))).float()
        self.std = torch.from_numpy(np.load(pjoin(data_root, 'Std.npy'))).float()

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            # print('model_output', model_output.shape, model_output)
            # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
            # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def gradients(self, x, hint, mask_hint):
        dis_mode = 'closet'
        assert dis_mode in ['center', 'origin', 'gmm', 'closet']
        
        def center_distance(points, hints, mask_hint):
            #points: (batch_size, n_frames, 22, 3)
            #hints: (batch_size, n_frames, 22+1, 8, 3)
            #mask_hint: (batch_size, n_frames, 22+1, 1)
            # calculate the distance between the center of the points and the center of the hints
            
            points_center = points.mean(dim=2) # (batch_size, n_frames, 3)
            hints_all_cube_centers = hints.sum(dim=3) / 8 # (batch_size, n_frames, 22+1, 3)
            
            # choice the hint center that is closest to the points center in xz
            dist = torch.norm(points_center.unsqueeze(2)[..., [0, 2]] - hints_all_cube_centers[..., [0, 2]], dim=-1) # (batch_size, n_frames, 22+1)
            valid_mask = mask_hint.squeeze(-1)
            dist[~valid_mask] = torch.inf
            closest_hint_center_index = torch.argmin(dist, dim=2) # (batch_size, n_frames)
            closest_hint_center_index_expanded = closest_hint_center_index.unsqueeze(-1).unsqueeze(-1)  # [batch_size,  n_frames  , 1, 1]
            closest_hint_center_index_expanded = closest_hint_center_index_expanded.expand(-1, -1, 1, 3) # [batch_size, n_frames, 1, 3]
            closest_hint_center = hints_all_cube_centers.gather(dim=2, index=closest_hint_center_index_expanded) # (batch_size, n_frames, 1, 3)
            closest_hint_center = closest_hint_center.squeeze(2) # (batch_size, n_frames, 3)
            
            distance = torch.norm(points_center[..., [0, 2]] - closest_hint_center[..., [0, 2]], dim=-1) # (batch_size, n_frames)
            return distance
        def center_distance_gmm(points, hints, mask_hint):
            """
            points:     (B, T, 22, 3)        关节位置
            hints:      (B, T, 22+1, 8, 3)   每个 box 的 8 个角点
            mask_hint:  (B, T, 22+1, 1)      有效 box 的掩码(布尔/0-1)
            
            返回:
            distance:   (B, T) 每帧的 GMM 负平均对数似然，值越小表示越匹配
            """
            # ------------- 配置超参（可按需调整，但不改变接口）-------------
            # 用角点到中心的二阶矩构造的对角协方差再乘以缩放；缩放<1 让高斯更“尖锐”
            cov_scale = 1.0/1.0      # 相当于 std ~ half-length/3
            min_var   = 1e-6         # 数值稳定的最小方差
            
            # ------------- 计算 GMM 参数（均值与对角协方差）-------------
            # box 中心: 8 个角点的平均
            centers = hints.mean(dim=3)                          # (B, T, K, 3)
            
            # 用角点的二阶矩估计“半径”-> 协方差（对角）
            # 更平滑且可微，避免 min/max 带来的不可导点
            # corner_var = E[(corner - center)^2] over 8 corners
            corner_offsets = hints - centers.unsqueeze(3)        # (B, T, K, 8, 3)
            corner_var = (corner_offsets ** 2).mean(dim=3)       # (B, T, K, 3)
            
            # 对角协方差：缩放+下界
            diag_var = cov_scale * corner_var + min_var          # (B, T, K, 3)
            
            # ------------- 构造混合权重（对 valid 分量均匀）-------------
            valid_mask = mask_hint.squeeze(-1).bool()            # (B, T, K)
            K_valid = valid_mask.sum(dim=2, keepdim=True).clamp_min(1)  # (B, T, 1)
            # log 权重: 对 valid 分量为 -log(K_valid)，对 invalid 为 -inf
            log_w = -torch.log(K_valid.to(points.dtype))         # (B, T, 1)
            log_wk = torch.where(
                valid_mask, 
                log_w.expand_as(valid_mask).to(points.dtype), 
                torch.tensor(-float('inf'), device=points.device, dtype=points.dtype)
            )                                                    # (B, T, K)
            
            # ------------- 计算每个关节在 GMM 下的 log 概率 -------------
            # 选择使用的维度：如果只想用 xz 平面，可把 dims = [0, 2]
            dims = [0, 1, 2]
            
            x = points[..., dims]                                # (B, T, J, D)
            mu = centers[..., dims]                              # (B, T, K, D)
            var = diag_var[..., dims]                            # (B, T, K, D)
            
            # 广播到 (B, T, J, K, D)
            diff = x.unsqueeze(3) - mu.unsqueeze(2)              # (B, T, J, K, D)
            var  = var.unsqueeze(2)                              # (B, T, 1, K, D)
            
            # 对角高斯的 log 密度: -0.5 * [ sum_d ( (diff^2 / var) + log(2π var) ) ]
            log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=points.dtype, device=points.device))
            log_prob_components = -0.5 * (
                (diff ** 2 / var).sum(dim=-1) +                 # Mahalanobis (对角)
                (log_two_pi + torch.log(var)).sum(dim=-1)       # log det(2πΣ)
            )                                                   # (B, T, J, K)
            
            # 加上 log 权重并对 K 做 logsumexp -> 混合分布的 log 概率
            log_wk = log_wk.unsqueeze(2)                         # (B, T, 1, K)
            log_mix = torch.logsumexp(log_prob_components + log_wk, dim=3)  # (B, T, J)
            
            # ------------- 聚合为每帧距离（负平均 log 似然）-------------
            # 对 22 个关节取平均，再取负，得到距离；值越小越匹配
            distance = -log_mix.mean(dim=2)                      # (B, T)
            
            # 若某帧没有任何 valid 分量，log_mix = -inf -> distance = +inf（合理地表示“不匹配”）
            return distance
        def center_distance_closet(
            points, hints, mask_hint,
            dims_local=[0,1,2],
            cov_scale=1.0/9.0,
            min_var=1e-6,
            tau_center=0.05,
            softplus_beta=1e-2,
            tail_alpha=0.25,
            squash_to_01=True
        ):
            device = points.device
            p32 = points.float()
            h32 = hints.float()

            B, T, K, Ncorn, _ = h32.shape  # Ncorn=8
            J = p32.shape[2]

            # 1) 中心与 diag_var（返回/可视化用）
            centers = h32.mean(dim=3)                           # (B,T,K,3)
            offsets = h32 - centers.unsqueeze(3)                # (B,T,K,8,3)
            corner_var = (offsets**2).mean(dim=3)               # (B,T,K,3)
            diag_var = torch.clamp(cov_scale * corner_var + min_var, min=min_var)

            valid_mask = mask_hint.squeeze(-1).bool()           # (B,T,K)

            # 2) OBB 局部坐标系 U（PCA）
            C = torch.matmul(offsets.transpose(-1, -2), offsets) / float(Ncorn)   # (B,T,K,3,3)
            evals, evecs = torch.linalg.eigh(C)                                   # (B,T,K,3),(B,T,K,3,3)
            evecs = torch.flip(evecs, dims=(-1,))                                  # (B,T,K,3,3) 列向量按大->小排序
            U_T = evecs.transpose(-1, -2)                                          # (B,T,K,3,3)

            # --- 角点投影到局部坐标：u_corners = offsets @ U^T ---
            # 原来用 matmul，这里改为显式 einsum，避免维度歧义
            # offsets: b t k n d,   U_T: b t k d e  ->  u_corners: b t k n e
            u_corners = torch.einsum('btknd,btkde->btkne', offsets, U_T)          # (B,T,K,8,3)
            a_full = torch.clamp(u_corners.abs().amax(dim=3), min=1e-6)           # (B,T,K,3)

            # 3) joints 投到局部坐标：u = (x - mu) @ U^T
            x_world = p32                                                          # (B,T,J,3)
            mu_world = centers                                                     # (B,T,K,3)
            diff_world = x_world.unsqueeze(3) - mu_world.unsqueeze(2)              # (B,T,J,K,3)
            # 使用显式 einsum：diff_world: b t j k d,  U_T: b t k d e  -> u_full: b t j k e
            u_full = torch.einsum('btjkd,btkde->btjk e', diff_world, U_T).contiguous()
            # 上面空格可能在旧版 torch 报错，稳妥写法（无空格）如下：
            # u_full = torch.einsum('btjkd,btkde->btjke', diff_world, U_T)         # (B,T,J,K,3)

            # 选择参与距离的局部轴
            Didx = torch.tensor(dims_local, device=device, dtype=torch.long)
            u = torch.index_select(u_full, -1, Didx)                               # (B,T,J,K,D)
            a = torch.index_select(a_full,  -1, Didx)                              # (B,T,K,D)

            # 4) 盒外惩罚（平滑 ReLU）
            rel = u.abs() - a.unsqueeze(2)                                         # (B,T,J,K,D)
            over = F.softplus(rel / max(softplus_beta, 1e-8)) * softplus_beta
            sigma = torch.clamp(tail_alpha * a.unsqueeze(2), min=1e-6)             # (B,T,1,K,D)
            E_jk = 0.5 * ((over / sigma) ** 2).sum(dim=-1)                         # (B,T,J,K)

            # 5) “最近 joint” 软选择（归一化局部距离）
            r2 = ((u / (a.unsqueeze(2) + 1e-6)) ** 2).sum(dim=-1)                  # (B,T,J,K)
            r2_min = r2.min(dim=2, keepdim=True).values
            w = torch.softmax(-(r2 - r2_min) / max(tau_center, 1e-8), dim=2)       # (B,T,J,K)
            E_hint = (w * E_jk).sum(dim=2)                                         # (B,T,K)

            # 6) 聚合有效 boxes
            E_hint = torch.where(valid_mask, E_hint, torch.zeros_like(E_hint))
            E_sum = E_hint.sum(dim=2)                                              # (B,T)
            s = 10.0   # 或按你的典型 E 量级取
            distance = E_sum / (E_sum + s)
            # distance = 1.0 - torch.exp(-E_sum) if squash_to_01 else E_sum
            distance = torch.nan_to_num(distance, nan=0.0, posinf=1e6, neginf=0.0)

            return distance
        with torch.enable_grad():
            x.requires_grad_(True)
            x_ = x.permute(0, 3, 2, 1).contiguous()
            x_ = x_.squeeze(2)
            x_ = x_ * self.std.to(x.device) + self.mean.to(x.device)
            n_joints = 22 if x_.shape[-1] == 263 else 21
            joint_pos = recover_from_ric(x_, n_joints)

            if dis_mode == 'center':
                distance = center_distance(joint_pos, hint, mask_hint)
                loss = torch.norm(distance, dim=-1)
            elif dis_mode == 'gmm':
                distance = center_distance_gmm(joint_pos, hint, mask_hint)
                loss = torch.norm(distance, dim=-1)
            elif dis_mode == 'closet':
                distance = center_distance_closet(joint_pos, hint, mask_hint)
                loss = torch.norm(distance, dim=-1)
            else:
                loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [x])[0]
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            grad[..., 0] = 0
            x.detach()
        return loss, grad

    def calc_grad_scale(self, mask_hint):
        assert mask_hint.shape[1] == 196
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def guide(self, x, t,model,clip_denoised=True, denoised_fn=None, 
              model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, 
              train=False, min_variance=0.01, pred_xstart=None):
        """
        Spatial guidance
        """
        n_joint = 22 if x.shape[1] == 263 else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance
        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            ratio = 5
            if t[0] < 200 and t[0] > 10:
                n_guide_steps = 100*ratio
            elif t[0] < 10:
                n_guide_steps = 500*ratio
            else:
                n_guide_steps = 10*ratio

        # process hint
        hint = model_kwargs['y']['hint'].clone().detach()

        mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint+1, 8*3).sum(dim=-1, keepdim=True) != 0
        
        
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint+1, 8, 3)
        
        
        if not train:
        
            scale = 2


        for _ in range(n_guide_steps):
            if pred_xstart is not None:
                loss, grad = self.gradients(pred_xstart, hint, mask_hint)
            else:
                loss, grad = self.gradients(x, hint, mask_hint)
            grad = model_variance * grad
            print(loss.sum())
            if t[0] >= t_stopgrad:
                x = x - scale * grad

        return x.detach()
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        

        if const_noise:
            noise = th.randn_like(x[0])
            noise = noise[None].repeat(x.shape[0], 1, 1, 1)
        else:
            noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        pred_xstart = None
        # YHC: test spatial guidance
        # if 'hint' in model_kwargs['y'].keys():
        #     # spatial guidance/classifier guidance
        #     sample = self.guide(sample, t,
        #                              model=model,
        #                              clip_denoised=clip_denoised,
        #                              denoised_fn=denoised_fn,
        #                              model_kwargs=model_kwargs,
        #                              pred_xstart=pred_xstart)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            if const_noise:
                img = th.randn(*shape[1:], device=device)
                img = img[None].repeat(shape[0], 1, 1, 1)
            else:
                img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        # x_t = self.guide(x_t, t, model_kwargs=model_kwargs, train=True)
        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

        terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

        terms["loss"] = terms["rot_mse"]

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
