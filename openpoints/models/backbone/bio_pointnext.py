import copy
import logging
from typing import List, Type

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import fps as pyg_fps, radius as pyg_radius
from torch_geometric.nn import global_mean_pool, global_max_pool, fps
from torch_geometric.utils import scatter

from ..build import MODELS
from ..layers import (
    create_convblock1d, create_convblock2d, create_linearblock,
    create_act, CHANNEL_MAP, create_grouper, furthest_point_sample,
    random_sample, three_interpolation, get_aggregation_feautres
)

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
    assert reduction in ['sum', 'max', 'mean'], f"Unsupported reduction: {reduction}"
    return lambda x, index, dim_size: scatter(
        x, index, dim=0, dim_size=dim_size, reduce=reduction
    )

class LocalAggregation(nn.Module):
    """
    Gom cụm đặc trưng cục bộ (N -> N).
    GIAI ĐOẠN 2: Bổ sung Gaussian RBF mã hóa khoảng cách Euclidean d_ij (16 chiều).
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
                 num_rbf: int = 16,        # <-- Số tâm Gauss đo cự ly
                 **kwargs
                 ):
        super().__init__()
        conv_args = conv_args if conv_args is not None else {}
        self.radius = group_args.get('radius', 6.0)
        self.nsample = group_args.get('nsample', 32)
        self.normalize_dp = group_args.get('normalize_dp', True)
        self.num_rbf = num_rbf

        # Khởi tạo RBF đo cự ly từ 0.0 Å đến đúng bán kính radius của tầng này
        self.rbf_dist = GaussianRBF(start=0.0, stop=self.radius, num_kernels=self.num_rbf)

        # Kênh cạnh đầu vào = [RBF(d_ij): 16 chiều] + [dp: 3 chiều] + [x_j: channels[0]]
        channels = list(channels)
        in_edge_channels = self.num_rbf + 3 + channels[0]

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

    def forward(self, pxb):
        p, x, b = pxb
        Total_N = p.shape[0]

        # 1. Quét lân cận
        row, col = safe_radius(p, p, self.radius, b, b, max_num_neighbors=self.nsample)

        # 2. Vector tọa độ tương đối dp [E, 3]
        diff = p[col] - p[row]
        dp = diff / self.radius if self.normalize_dp else diff

        # 3. GIAI ĐOẠN 2: Tính khoảng cách Euclidean d_ij và chiếu qua Gaussian RBF
        # Thêm 1e-8 để chống lỗi đạo hàm khi 2 điểm trùng nhau (d = 0)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)  # [E, 1]
        dist_rbf = self.rbf_dist(dist)                                        # [E, 16]

        # 4. Ghép đặc trưng cạnh: [RBF(d_ij) || dp || x_j]
        xj = x[col]
        edge_features = torch.cat([dist_rbf, dp, xj], dim=-1)                 # [E, 16 + 3 + C]

        # 5. MLP và gom tụ về điểm tâm
        edge_features = self.convs(edge_features)
        x_aggr = self.pool(edge_features, row, dim_size=Total_N)

        return x_aggr

class InvResMLP(nn.Module):
    """
    Khối Inverted Residual Bottleneck trong PointNeXt (Phiên bản PyG Sparse Batching).
    Đặc điểm: Giữ nguyên số lượng điểm N và số kênh C (N -> N, C -> C).
    """

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

        # 1. Tầng tương tác không gian cục bộ (LocalAggregation đã viết ở bước trước)
        # channels = [in_channels, in_channels] -> giữ nguyên số kênh sau khi gom cụm
        self.convs = LocalAggregation(
            channels=[in_channels, in_channels],
            norm_args=norm_args,
            act_args=act_args if num_posconvs > 0 else None,
            group_args=group_args,
            conv_args=conv_args,
            **aggr_args,
            **kwargs
        )

        # 2. Xây dựng danh sách kênh cho Pointwise MLP (mở rộng và nén kênh)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            # Chuẩn mực của Inverted Bottleneck: [C, 4C, C]
            channels = [in_channels, mid_channels, in_channels]

        # 3. Tạo chuỗi Pointwise MLP (dùng create_linearblock của OpenPoints)
        # Vì tensor của PyG có shape [Total_N, C], tầng Linear đóng vai trò như Pointwise 1x1 Conv
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

    def forward(self, pxb):
        """
        Nhận vào tuple/list pxb = (p, x, b):
            p: [Total_N, 3]  -> Tọa độ các điểm
            x: [Total_N, C]  -> Đặc trưng của các điểm
            b: [Total_N]     -> Chỉ số batch
        Trả về:
            [p, x, b]: Giữ nguyên tọa độ, batch index và cập nhật đặc trưng x mới
        """
        p, x, b = pxb
        identity = x  # Giữ lại nhánh tắt (Residual Identity)

        # 1. Gom tụ không gian cục bộ (Depthwise-like)
        x = self.convs([p, x, b])

        # 2. Mở rộng và nén kênh đặc trưng (Pointwise C -> 4C -> C)
        x = self.pwconv(x)

        # 3. Cộng nhánh tắt phần dư (Skip connection)
        # Đảm bảo cùng kích thước kênh C trước khi cộng
        if self.use_res and x.shape[-1] == identity.shape[-1]:
            x = x + identity

        # 4. Kích hoạt phi tuyến cuối cùng
        x = self.act(x)

        return [p, x, b]

class SetAbstraction(nn.Module):
    """
    Khối hạ mẫu theo tỷ lệ % (FPS) và gom cụm.
    GIAI ĐOẠN 2: Bổ sung Gaussian RBF mã hóa khoảng cách Euclidean d_ij (16 chiều).
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
                 num_rbf: int = 16,        # <-- Số tâm Gauss đo cự ly
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.num_rbf = num_rbf if not is_head else 0
        conv_args = conv_args if conv_args is not None else {}

        # 1. Kênh cạnh: Nếu không phải is_head, có thêm 16 chiều RBF + 3 chiều dp
        in_edge_channels = in_channels if is_head else (in_channels + 3 + self.num_rbf)
        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_edge_channels] + [mid_channel] * (layers - 1) + [out_channels]

        # 2. Skip connection
        if self.use_res:
            self.skipconv = create_linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None
            ) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        # 3. Chuỗi MLP
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

            # RBF đo cự ly cho tầng SetAbstraction
            self.rbf_dist = GaussianRBF(start=0.0, stop=self.radius, num_kernels=self.num_rbf)

            if sampler.lower() == 'fps':
                self.sample_fn = pyg_fps
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
            else:
                raise NotImplementedError(f"Sampler {sampler} not supported")

    def forward(self, pxb):
        p, x, b = pxb

        if self.is_head:
            x = self.convs(x)
            return p, x, b

        # 1. Hạ mẫu bằng FPS theo tỷ lệ %
        if not self.all_aggr:
            idx = self.sample_fn(p, b, ratio=(1.0 / self.stride))
            new_p = p[idx]
            new_b = b[idx]
        else:
            idx = None
            new_p = p
            new_b = b

        if self.use_res:
            identity = self.skipconv(x[idx])

        # 2. Quét lân cận hai tập điểm
        row, col = safe_radius(p, new_p, self.radius, b, new_b, max_num_neighbors=self.nsample)

        # 3. Vector tọa độ tương đối dp [E, 3]
        diff = p[col] - new_p[row]
        dp = diff / self.radius if self.normalize_dp else diff

        # 4. GIAI ĐOẠN 2: Tính khoảng cách Euclidean d_ij và chiếu qua Gaussian RBF
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)  # [E, 1]
        dist_rbf = self.rbf_dist(dist)                                        # [E, 16]

        # 5. Ghép đặc trưng cạnh: [RBF(d_ij) || dp || x_j]
        xj = x[col]
        edge_features = torch.cat([dist_rbf, dp, xj], dim=-1)

        # 6. MLP và Max-pooling về các điểm tâm mới
        edge_features = self.convs(edge_features)
        new_x = scatter(edge_features, row, dim=0, dim_size=new_p.size(0), reduce='max')

        if self.use_res:
            new_x = self.act(new_x + identity)

        return new_p, new_x, new_b


class GaussianRBF(nn.Module):
    """
    Hàm cơ sở bán kính Gauss (Gaussian Radial Basis Functions).
    Biến đổi một đại lượng vô hướng liên tục (như SASA/RSA) thành vector K-chiều phân bố phi tuyến.
    """
    def __init__(self, start=0.0, stop=1.0, num_kernels=16):
        super().__init__()
        # Tạo K tâm Gauss rải đều từ start đến stop
        offset = torch.linspace(start, stop, num_kernels)
        self.register_buffer('offset', offset)
        # Độ rộng sigma tương ứng với khoảng cách giữa các tâm
        self.coeff = -0.5 / ((stop - start) / (num_kernels - 1)) ** 2

    def forward(self, v):
        # v: [N, 1] -> output: [N, num_kernels]
        return torch.exp(self.coeff * (v - self.offset) ** 2)


class SurfaceGatedBiochemicalStem(nn.Module):
    """
    TẦNG 1: SURFACE-GATED BIOCHEMICAL UNIT (SGBU)
    - Nhánh 1 (Sequence): 20 Axit Amin One-hot -> Linear Embedding + Tương tác sâu
    - Nhánh 2 (Surface): SASA (0-300 Å²) & RSA (0.0-1.0) mở rộng qua Gaussian RBF 16 bins -> 32 chiều
    - Cơ chế Gating: Bề mặt điều hòa hoạt tính hóa sinh (g * Sequence + (1-g) * Surface)
    TUYỆT ĐỐI KHÔNG DÙNG TỌA ĐỘ p Ở ĐÂY -> BẢO TOÀN SE(3) INVARIANCE 100%!
    """
    def __init__(self, out_channels=32):
        super().__init__()
        
        # 1. Nhánh Sequence (One-hot 20 Axit Amin)
        self.seq_branch = nn.Sequential(
            nn.Linear(20, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # 2. Nhánh Surface (SASA & RSA qua Gaussian RBF)
        self.rbf_sasa = GaussianRBF(start=0.0, stop=300.0, num_kernels=16)
        self.rbf_rsa  = GaussianRBF(start=0.0, stop=1.0, num_kernels=16)
        self.surf_branch = nn.Sequential(
            nn.Linear(32, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # 3. Surface Gate: Tính toán hệ số cổng đóng/mở g in (0, 1)
        self.surface_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        # 4. Tầng chuẩn hóa và chiếu đặc trưng dung hợp cuối cùng
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, p, x):
        # Tách dữ liệu từ tensor x [Total_N, 22]
        seq_onehot = x[:, :20]
        sasa_val   = x[:, 20:21]
        rsa_val    = x[:, 21:22]

        # 1. Trích xuất đặc trưng chuỗi và bề mặt độc lập
        f_seq = self.seq_branch(seq_onehot)

        sasa_rbf = self.rbf_sasa(sasa_val)
        rsa_rbf  = self.rbf_rsa(rsa_val)
        surf_rbf = torch.cat([sasa_rbf, rsa_rbf], dim=-1) # [Total_N, 32]
        f_surf   = self.surf_branch(surf_rbf)

        # 2. Cơ chế cổng điều hòa sinh học (Surface-Gated):
        # Mức độ phơi lộ quyết định mức độ kích hoạt của loại axit amin
        g = self.surface_gate(f_surf)
        f_0 = g * f_seq + (1.0 - g) * f_surf

        # 3. Chiếu ra kích thước width của PointNeXt
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

        # 1. KHỞI TẠO STEM MỚI (SGBU)
        self.use_multimodal = use_multimodal
        if self.use_multimodal:
            self.stem = SurfaceGatedBiochemicalStem(out_channels=width)
            self.in_channels = width
        else:
            self.stem = None
            self.in_channels = in_channels

        # 2. Tự động tính toán bán kính nở rộng
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'[PointNextEncoderPyG] Radii: {self.radii}, Nsample: {self.nsample}')

        # 3. Tính toán số kênh cho các Stage
        channels = []
        curr_w = width
        for stride in strides:
            if stride != 1:
                curr_w *= 2
            channels.append(curr_w)

        # 4. Lắp ráp 6 Stage
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

        # DUNG HỢP BẰNG SGBU CHUẨN SE(3)
        if self.use_multimodal and self.stem is not None:
            f0 = self.stem(p0, f0)

        p, x, b = p0, f0, batch0
        for stage in self.encoder:
            p, x, b = stage([p, x, b])

        return p, x, b

    def forward(self, p0, f0=None, batch0=None):
        return self.forward_cls_feat(p0, f0, batch0)