import copy
import logging
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import fps as pyg_fps, radius as pyg_radius
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter

from ..build import MODELS
from ..layers import (
    create_linearblock, create_act, random_sample
)


# =============================================================================
# CÁC HÀM TOÁN HỌC QUATERNION VÀ LRF (LOCAL REFERENCE FRAME)
# =============================================================================

def rotation_matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Chuyển đổi ma trận quay 3x3 sang Unit Quaternion (w, x, y, z).
    Thuật toán Shepperd tránh chia cho 0, bảo toàn tính liên tục vi phân.
    R: [E, 3, 3] -> q: [E, 4] với w >= 0.
    """
    orig_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    num_matrices = R.size(0)

    if num_matrices == 0:
        return torch.empty((*orig_shape, 4), device=R.device)

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    q0 = torch.stack([t0, m21 - m12, m02 - m20, m10 - m01], dim=-1)
    q1 = torch.stack([m21 - m12, t1, m01 + m10, m02 + m20], dim=-1)
    q2 = torch.stack([m02 - m20, m01 + m10, t2, m12 + m21], dim=-1)
    q3 = torch.stack([m10 - m01, m02 + m20, m12 + m21, t3], dim=-1)

    candidates_t = torch.stack([t0, t1, t2, t3], dim=-1)
    max_idx = torch.argmax(candidates_t, dim=-1)

    candidates_q = torch.stack([q0, q1, q2, q3], dim=1)
    q = candidates_q[torch.arange(num_matrices, device=R.device), max_idx]

    q = q / (torch.norm(q, dim=-1, keepdim=True) + eps)

    # Đảm bảo w >= 0 (Antipodal symmetry)
    w_negative = (q[:, 0] < 0).unsqueeze(-1)
    q = torch.where(w_negative, -q, q)

    return q.reshape(*orig_shape, 4)


def _single_protein_lrf(p: torch.Tensor) -> torch.Tensor:
    """Dựng LRF cho 1 protein: p [N, 3] -> F [N, 3, 3] chứa [u_x, u_y, u_z] dạng cột."""
    N = p.size(0)
    device = p.device

    if N == 1:
        return torch.eye(3, device=device).unsqueeze(0)
    elif N == 2:
        t = p[1:] - p[:-1]
        u_x = t / (torch.norm(t, dim=-1, keepdim=True) + 1e-8)
        arb = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(2, 1)
        u_z = torch.cross(u_x, arb, dim=-1)
        u_z = u_z / (torch.norm(u_z, dim=-1, keepdim=True) + 1e-8)
        u_y = torch.cross(u_z, u_x, dim=-1)
        return torch.stack([u_x, u_y, u_z], dim=-1)

    # 1. Hướng tiếp tuyến mạch chính u_x
    t = torch.empty_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0]    = p[1] - p[0]
    t[-1]   = p[-1] - p[-2]
    u_x = t / (torch.norm(t, dim=-1, keepdim=True) + 1e-8)

    # 2. Hướng uốn cong u_y (mặt trước / side-chain nhô ra)
    n = torch.empty_like(p)
    n[1:-1] = p[2:] - 2.0 * p[1:-1] + p[:-2]
    n[0]    = n[1]
    n[-1]   = n[-2]

    # Gram-Schmidt trực giao hóa
    proj = torch.sum(n * u_x, dim=-1, keepdim=True) * u_x
    u_y_raw = n - proj
    u_y_norm = torch.norm(u_y_raw, dim=-1, keepdim=True)

    straight_line = (u_y_norm < 1e-4).squeeze(-1)
    if straight_line.any():
        arb = torch.zeros_like(u_x)
        arb[:, 1] = 1.0
        collinear = (torch.abs(u_x[:, 1]) > 0.9).unsqueeze(-1)
        arb = torch.where(collinear, torch.tensor([1.0, 0.0, 0.0], device=device), arb)
        proj_arb = torch.sum(arb * u_x, dim=-1, keepdim=True) * u_x
        u_y_raw = torch.where(straight_line.unsqueeze(-1), arb - proj_arb, u_y_raw)
        u_y_norm = torch.norm(u_y_raw, dim=-1, keepdim=True)

    u_y = u_y_raw / (u_y_norm + 1e-8)
    u_z = torch.cross(u_x, u_y, dim=-1)

    # Ma trận F_i = [u_x, u_y, u_z] (dạng cột)
    return torch.stack([u_x, u_y, u_z], dim=-1)


def compute_backbone_lrf(p: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Dựng LRF cho toàn bộ batch protein rời rạc trong PyG."""
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    F_list = []
    for b in range(batch_size):
        mask = (batch == b)
        p_b = p[mask]
        F_list.append(_single_protein_lrf(p_b))
    return torch.cat(F_list, dim=0)


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
    assert reduction in ['sum', 'max', 'mean'], f"Unsupported reduction: {reduction}"
    return lambda x, index, dim_size: scatter(
        x, index, dim=0, dim_size=dim_size, reduce=reduction
    )


class GaussianRBF(nn.Module):
    """Mở rộng đại lượng vô hướng thành vector K-chiều phân bố phi tuyến."""
    def __init__(self, start=0.0, stop=1.0, num_kernels=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_kernels)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / ((stop - start) / (num_kernels - 1)) ** 2

    def forward(self, v):
        return torch.exp(self.coeff * (v - self.offset) ** 2)


class LocalAggregation(nn.Module):
    """
    Gom cụm đặc trưng cục bộ (N -> N).
    BƯỚC 2: Bổ sung tọa độ cục bộ bất biến p_ij^local (3 chiều) và Quaternion q_ij (4 chiều).
    """
    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 6.0, 'nsample': 32},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 num_rbf: int = 16,
                 **kwargs
                 ):
        super().__init__()
        conv_args = conv_args if conv_args is not None else {}
        self.radius = group_args.get('radius', 6.0)
        self.nsample = group_args.get('nsample', 32)
        self.normalize_dp = group_args.get('normalize_dp', True)
        self.num_rbf = num_rbf

        # RBF đo cự ly vật lý thực tế
        self.rbf_dist = GaussianRBF(start=0.0, stop=self.radius, num_kernels=self.num_rbf)

        # KÊNH CẠNH MỚI: RBF(16) + p_local(3) + Quaternion(4) + x_j(channels[0]) = 23 + C
        channels = list(channels)
        in_edge_channels = self.num_rbf + 3 + 4 + channels[0]

        convs = []
        curr_c = in_edge_channels
        for i in range(len(channels) - 1):
            is_last_layer = (i == len(channels) - 2)
            convs.append(
                create_linearblock(
                    curr_c, channels[i + 1],
                    norm_args=norm_args,
                    act_args=None if is_last_layer and not last_act else act_args,
                    **conv_args
                )
            )
            curr_c = channels[i + 1]
        self.convs = nn.Sequential(*convs)

        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)

    def forward(self, pxbf):
        # pxb nhận bộ 4: [p, x, b, F]
        p, x, b, F_mat = pxbf
        Total_N = p.shape[0]

        # 1. Quét lân cận bằng Ball Query
        row, col = pyg_radius(p, p, self.radius, b, b, max_num_neighbors=self.nsample)

        # 2. Vector hiệu toàn cục
        diff = p[col] - p[row]

        # 3. TỌA ĐỘ CỤC BỘ BẤT BIẾN: p_ij^local = F_i^T * (p_j - p_i)
        F_i = F_mat[row]  # [E, 3, 3]
        p_local = torch.bmm(F_i.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)  # [E, 3]
        p_local_norm = p_local / self.radius if self.normalize_dp else p_local

        # 4. GÓC XOAY TƯƠNG ĐỐI VÀ QUATERNION: R_ij = F_i^T * F_j -> q_ij in R^4
        F_j = F_mat[col]  # [E, 3, 3]
        R_ij = torch.bmm(F_i.transpose(-1, -2), F_j)                                 # [E, 3, 3]
        q_ij = rotation_matrix_to_quaternion(R_ij)                                   # [E, 4]

        # 5. Khoảng cách Euclidean d_ij qua RBF
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)        # [E, 1]
        dist_rbf = self.rbf_dist(dist)                                               # [E, 16]

        # 6. Ghép vector cạnh bất biến SO(3) 100%: [RBF(16) || p_local(3) || q_ij(4) || x_j(C)]
        xj = x[col]
        edge_features = torch.cat([dist_rbf, p_local_norm, q_ij, xj], dim=-1)        # [E, 23 + C]

        # 7. MLP và Max-pooling gom tụ về điểm tâm
        edge_features = self.convs(edge_features)
        x_aggr = self.pool(edge_features, row, dim_size=Total_N)

        return x_aggr


class InvResMLP(nn.Module):
    """Khối Inverted Residual Bottleneck trong PointNeXt."""
    def __init__(self,
                 in_channels: int,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion: int = 1,
                 use_res: bool = True,
                 num_posconvs: int = 2,
                 less_act: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        conv_args = conv_args if conv_args is not None else {}
        mid_channels = int(in_channels * expansion)

        self.convs = LocalAggregation(
            channels=[in_channels, in_channels],
            norm_args=norm_args,
            act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args,
            conv_args=conv_args,
            **aggr_args,
            **kwargs
        )

        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]

        pwconv = []
        for i in range(len(channels) - 1):
            is_last_layer = (i == len(channels) - 2)
            pwconv.append(
                create_linearblock(
                    channels[i], channels[i + 1],
                    norm_args=norm_args,
                    act_args=act_args if not is_last_layer and not less_act else None,
                    **conv_args
                )
            )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pxbf):
        p, x, b, F_mat = pxbf
        identity = x

        # Truyền bộ 4 [p, x, b, F] vào LocalAggregation
        x = self.convs([p, x, b, F_mat])
        x = self.pwconv(x)

        if self.use_res and x.shape[-1] == identity.shape[-1]:
            x = x + identity

        x = self.act(x)
        # Giữ nguyên và truyền tiếp bộ 4 [p, x, b, F]
        return [p, x, b, F_mat]


class SetAbstraction(nn.Module):
    """
    Khối hạ mẫu theo tỷ lệ % (FPS) và gom cụm.
    BƯỚC 2: Tích hợp tọa độ cục bộ p_local và Quaternion q_ij.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int = 1,
                 stride: int = 1,
                 group_args: dict = {'NAME': 'ballquery', 'radius': 6.0, 'nsample': 32},
                 norm_args: dict = {'norm': 'bn1d'},
                 act_args: dict = {'act': 'relu'},
                 conv_args=None,
                 sampler: str = 'fps',
                 feature_type: str = 'dp_fj',
                 use_res: bool = False,
                 is_head: bool = False,
                 num_rbf: int = 16,
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.num_rbf = num_rbf if not is_head else 0
        conv_args = conv_args if conv_args is not None else {}

        # KÊNH CẠNH MỚI: RBF(16) + p_local(3) + Quaternion(4) + in_channels = 23 + in_channels
        in_edge_channels = in_channels if is_head else (in_channels + 3 + 4 + self.num_rbf)
        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_edge_channels] + [mid_channel] * (layers - 1) + [out_channels]

        if self.use_res:
            self.skipconv = create_linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None
            ) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        convs = []
        for i in range(len(channels) - 1):
            is_last_layer = (i == len(channels) - 2)
            convs.append(
                create_linearblock(
                    channels[i], channels[i + 1],
                    norm_args=norm_args if not is_head else None,
                    act_args=None if is_last_layer and (self.use_res or is_head) else act_args,
                    **conv_args
                )
            )
        self.convs = nn.Sequential(*convs)

        if not is_head:
            self.radius = group_args.get('radius', 6.0)
            self.nsample = group_args.get('nsample', 32)
            self.normalize_dp = group_args.get('normalize_dp', True)

            self.rbf_dist = GaussianRBF(start=0.0, stop=self.radius, num_kernels=self.num_rbf)

            if sampler.lower() == 'fps':
                self.sample_fn = pyg_fps
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
            else:
                raise NotImplementedError(f"Sampler {sampler} not supported")

    def forward(self, pxbf):
        p, x, b, F_mat = pxbf

        if self.is_head:
            x = self.convs(x)
            return [p, x, b, F_mat]

        # 1. Hạ mẫu bằng FPS chuẩn
        if not self.all_aggr:
            idx = self.sample_fn(p, b, ratio=(1.0 / self.stride))
            new_p = p[idx]
            new_b = b[idx]
            new_F = F_mat[idx]  # CÁC ĐIỂM TÂM MỚI KẾ THỪA LRF TƯƠNG ỨNG!
        else:
            idx = None
            new_p = p
            new_b = b
            new_F = F_mat

        if self.use_res:
            identity = self.skipconv(x[idx]) if idx is not None else self.skipconv(x)

        # 2. Quét lân cận hai tập điểm bằng Ball Query
        row, col = pyg_radius(p, new_p, self.radius, b, new_b, max_num_neighbors=self.nsample)

        # 3. Vector hiệu
        diff = p[col] - new_p[row]

        # 4. TỌA ĐỘ CỤC BỘ BẤT BIẾN: new_F[row]^T * (p_j - new_p_i)
        F_i = new_F[row]  # [E, 3, 3]
        p_local = torch.bmm(F_i.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)  # [E, 3]
        p_local_norm = p_local / self.radius if self.normalize_dp else p_local

        # 5. GÓC XOAY TƯƠNG ĐỐI VÀ QUATERNION: R_ij = new_F[row]^T * F[col]
        F_j = F_mat[col]  # [E, 3, 3]
        R_ij = torch.bmm(F_i.transpose(-1, -2), F_j)                                 # [E, 3, 3]
        q_ij = rotation_matrix_to_quaternion(R_ij)                                   # [E, 4]

        # 6. Khoảng cách Euclidean d_ij qua RBF
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)        # [E, 1]
        dist_rbf = self.rbf_dist(dist)                                               # [E, 16]

        # 7. Ghép đặc trưng cạnh: [RBF(16) || p_local(3) || q_ij(4) || x_j(C)]
        xj = x[col]
        edge_features = torch.cat([dist_rbf, p_local_norm, q_ij, xj], dim=-1)        # [E, 23 + C]

        # 8. MLP và Max-pooling về các điểm tâm mới
        edge_features = self.convs(edge_features)
        new_x = scatter(edge_features, row, dim=0, dim_size=new_p.size(0), reduce='max')

        if self.use_res:
            new_x = self.act(new_x + identity)

        # Trả về bộ 4 để đi tiếp sang Stage tiếp theo
        return [new_p, new_x, new_b, new_F]


# =============================================================================
# STEM VÀ ENCODER CHÍNH
# =============================================================================

class SurfaceGatedBiochemicalStem(nn.Module):
    """Stem đa phương thức: SGBU (Sequence + Surface qua RBF)."""
    def __init__(self, out_channels=32):
        super().__init__()
        self.seq_branch = nn.Sequential(
            nn.Linear(20, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.rbf_sasa = GaussianRBF(start=0.0, stop=300.0, num_kernels=16)
        self.rbf_rsa  = GaussianRBF(start=0.0, stop=1.0, num_kernels=16)
        self.surf_branch = nn.Sequential(
            nn.Linear(32, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.surface_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, p, x):
        seq_onehot = x[:, :20]
        sasa_val   = x[:, 20:21]
        rsa_val    = x[:, 21:22]

        f_seq = self.seq_branch(seq_onehot)
        sasa_rbf = self.rbf_sasa(sasa_val)
        rsa_rbf  = self.rbf_rsa(rsa_val)
        surf_rbf = torch.cat([sasa_rbf, rsa_rbf], dim=-1)
        f_surf   = self.surf_branch(surf_rbf)

        g = self.surface_gate(f_surf)
        f_0 = g * f_seq + (1.0 - g) * f_surf
        return self.fusion_proj(f_0)


@MODELS.register_module()
class PointNextEncoderPyG(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 width: int = 32,
                 blocks: List[int] = [1, 1, 1, 1, 1, 1],
                 strides: List[int] = [1, 2, 2, 2, 2, 1],
                 block: str or Type[nn.Module] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 6.0,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 2,
                 sa_use_res: bool = True,
                 use_multimodal: bool = True,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = globals()[block] if block in globals() else eval(block)

        self.blocks = blocks
        self.strides = strides
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn1d'})
        self.act_args = kwargs.get('act_args', {'act': 'relu'})
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        
        radius_scaling = kwargs.get('radius_scaling', 1.5)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.use_multimodal = use_multimodal
        if self.use_multimodal:
            self.stem = SurfaceGatedBiochemicalStem(out_channels=width)
            self.in_channels = width
        else:
            self.stem = None
            self.in_channels = in_channels

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'[PointNextEncoderPyG] Radii: {self.radii}, Nsample: {self.nsample}')

        channels = []
        curr_w = width
        for stride in strides:
            if stride != 1:
                curr_w *= 2
            channels.append(curr_w)

        encoder = []
        for i in range(len(blocks)):
            stage_group_args = copy.deepcopy(group_args)
            stage_group_args['radius'] = self.radii[i]
            stage_group_args['nsample'] = self.nsample[i]

            encoder.append(self._make_enc(
                block=block,
                channels=channels[i],
                blocks=blocks[i],
                stride=strides[i],
                group_args=stage_group_args,
                is_head=(i == 0 and strides[i] == 1 and not self.use_multimodal)
            ))

        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        param_list = []
        if isinstance(param, List):
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:
            current = param
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([current] * self.blocks[i])
                else:
                    param_list.append(
                        [current] + [current * param_scaling] * (self.blocks[i] - 1)
                    )
                    current *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args['radius']
        nsample = group_args['nsample']
        
        sa_group_args = copy.deepcopy(group_args)
        sa_group_args['radius'] = radii[0]
        sa_group_args['nsample'] = nsample[0]

        layers.append(
            SetAbstraction(
                in_channels=self.in_channels,
                out_channels=channels,
                layers=self.sa_layers if not is_head else 1,
                stride=stride,
                group_args=sa_group_args,
                sampler=self.sampler,
                norm_args=self.norm_args,
                act_args=self.act_args,
                conv_args=self.conv_args,
                is_head=is_head,
                use_res=self.sa_use_res,
                **self.aggr_args
            )
        )
        self.in_channels = channels

        for i in range(1, blocks):
            inv_group_args = copy.deepcopy(group_args)
            inv_group_args['radius'] = radii[i]
            inv_group_args['nsample'] = nsample[i]

            layers.append(
                block(
                    in_channels=self.in_channels,
                    aggr_args=self.aggr_args,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    group_args=inv_group_args,
                    conv_args=self.conv_args,
                    expansion=self.expansion,
                    use_res=self.use_res
                )
            )
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None, batch0=None):
        if hasattr(p0, 'keys'):
            p0, f0, batch0 = p0['pos'], p0.get('x', None), p0.get('batch', None)
        elif hasattr(p0, 'pos'):
            p0, f0, batch0 = p0.pos, getattr(p0, 'x', None), getattr(p0, 'batch', None)

        if f0 is None:
            f0 = p0.clone()

        # 1. DỰNG HỆ TRỤC LRF CỤC BỘ BAN ĐẦU CHO TỪNG AXIT AMIN (Chỉ mất ~0.2ms trên GPU)
        F_mat = compute_backbone_lrf(p0, batch0)

        # 2. Dung hợp đặc trưng Nút bằng SGBU (chuẩn SE(3))
        if self.use_multimodal and self.stem is not None:
            f0 = self.stem(p0, f0)

        # 3. Đóng gói bộ 4 phần tử [p, x, b, F] cho các Stage của PointNeXt
        pxbf = [p0, f0, batch0, F_mat]
        for stage in self.encoder:
            pxbf = stage(pxbf)

        p, x, b, _ = pxbf
        return p, x, b

    def forward(self, p0, f0=None, batch0=None):
        return self.forward_cls_feat(p0, f0, batch0)