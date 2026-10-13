import copy
import logging
from typing import List, Type

import torch
import torch.nn as nn
from torch_geometric.nn import radius as pyg_radius
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

def pyg_fps(pos, batch, ratio=0.5, random_start=True):
    """
    Thuật toán Furthest Point Sampling thuần PyTorch hỗ trợ Sparse Batching (PyG).
    Không cần cài đặt pyg-lib hay build C++, chạy mượt mà trên GPU Kaggle.
    """
    try:
        from torch_cluster import fps as cluster_fps
        return cluster_fps(pos, batch, ratio=ratio, random_start=random_start)
    except (ImportError, Exception):
        pass

    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    sampled_indices = []

    for b in range(batch_size):
        mask = (batch == b)
        b_indices = torch.nonzero(mask, as_tuple=True)[0]
        n_pts = b_indices.size(0)
        k = max(1, int(n_pts * ratio))

        if k >= n_pts:
            sampled_indices.append(b_indices)
            continue

        p = pos[b_indices]  # [N_b, 3]
        selected = torch.empty(k, dtype=torch.long, device=pos.device)
        dists = torch.full((n_pts,), float('inf'), device=pos.device)

        farthest_idx = torch.randint(0, n_pts, (1,), device=pos.device).item() if random_start else 0

        for i in range(k):
            selected[i] = farthest_idx
            cur_p = p[farthest_idx]
            d = torch.sum((p - cur_p) ** 2, dim=-1)
            dists = torch.min(dists, d)
            farthest_idx = torch.argmax(dists).item()

        sampled_indices.append(b_indices[selected])

    return torch.cat(sampled_indices, dim=0)

class LocalAggregation(nn.Module):
    """Local aggregation layer for a set (PyG Sparse Batching version).
    Set abstraction layer abstracts features from a larger set to a smaller set.
    Local aggregation layer aggregates features from the same set (N -> N).
    """

    def __init__(
        self,
        channels: List[int],
        norm_args={'norm': 'bn1d'},
        act_args={'act': 'relu'},
        group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
        conv_args=None,
        feature_type='dp_fj',
        reduction='max',
        last_act=True,
        **kwargs
    ):
        super().__init__()

        if kwargs:
            logging.warning(
                f"kwargs: {kwargs} are not used in {__class__.__name__}"
            )

        conv_args = conv_args if conv_args is not None else {}

        # 1. Compute the input channel size based on feature_type
        # For example, 'dp_fj' adds 3 coordinate dimensions.
        channels = list(channels)
        channels[0] = CHANNEL_MAP[feature_type](channels[0])

        # 2. Build edge transformation layers using OpenPoints' create_linearblock
        # Since PyG sparse data has shape [E, C], Linear is equivalent to a 1x1 Conv.
        convs = []

        for i in range(len(channels) - 1):
            is_last_layer = (i == len(channels) - 2)
            convs.append(
                create_linearblock(
                    channels[i],
                    channels[i + 1],
                    norm_args=norm_args,
                    act_args=None if is_last_layer and not last_act else act_args,
                    **conv_args
                )
            )

        self.convs = nn.Sequential(*convs)

        # 3. Extract grouping parameters from group_args
        self.radius = group_args.get('radius', 6.0)
        self.nsample = group_args.get('nsample', 32)
        self.normalize_dp = group_args.get('normalize_dp', True)

        # 4. Set up the pooling function
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pxb):
        """
        Input tuple/list pxb = (p, x, b):
            p: [Total_N, 3]  -> Original coordinates of the points
            x: [Total_N, C]  -> Point features
            b: [Total_N]     -> PyG batch indices

        Returns:
            x_aggr: [Total_N, channels[-1]] -> New features at the center points
        """
        p, x, b = pxb
        Total_N = p.shape[0]

        # 1. Find neighbors using PyG: row represents the center (i),
        # while col represents the neighbor (j).
        # Limit the maximum number of points within the radius to nsample.
        row, col = pyg_radius(
            x=p,
            y=p,
            r=self.radius,
            batch_x=b,
            batch_y=b,
            max_num_neighbors=self.nsample
        )

        # 2. Compute the relative coordinate vector delta_p = p_j - p_i
        dp = p[col] - p[row]

        if self.normalize_dp:
            dp = dp / self.radius  # Normalize to [-1, 1] when normalize_dp is enabled.

        # 3. Construct features based on feature_type
        xj = x[col]

        if self.feature_type == 'dp_fj':
            edge_features = torch.cat([dp, xj], dim=-1)
        elif self.feature_type == 'fj':
            edge_features = xj
        elif self.feature_type == 'dp':
            edge_features = dp
        else:
            raise NotImplementedError(
                f"Feature type {self.feature_type} not implemented"
            )

        # 4. Learn edge features through the MLP
        edge_features = self.convs(edge_features)

        # 5. Aggregate features back to the center points
        # Strictly protected by dim_size=Total_N.
        # Points without neighbors still retain their position in the tensor with value 0.
        x_aggr = self.pool(
            edge_features,
            row,
            dim_size=Total_N
        )

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
    """The modified set abstraction module in PointNet++ with residual connection support
    (Phiên bản PyG Sparse Batching hỗ trợ độ dài biến thiên theo ratio).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int = 1,
                 stride: int = 1,
                 group_args: dict = {'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 norm_args: dict = {'norm': 'bn1d'},
                 act_args: dict = {'act': 'relu'},
                 conv_args=None,
                 sampler: str = 'fps',
                 feature_type: str = 'dp_fj',
                 use_res: bool = False,
                 is_head: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        conv_args = conv_args if conv_args is not None else {}

        # 1. Tính toán danh sách số kênh qua các tầng MLP
        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        # 2. Nhánh Skip Connection (Linear chiếu số kênh nếu in_channels != out_channels)
        if self.use_res:
            self.skipconv = create_linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None
            ) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        # 3. Tạo chuỗi MLP (LinearBlock cho đồ thị thưa [E, C])
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

        # 4. Thiết lập bộ lấy mẫu và tham số gom cụm nếu không phải tầng Head
        if not is_head:
            self.radius = group_args.get('radius', 6.0)
            self.nsample = group_args.get('nsample', 32)
            self.normalize_dp = group_args.get('normalize_dp', True)

            # Thuật toán lấy mẫu FPS
            if sampler.lower() == 'fps':
                self.sample_fn = pyg_fps
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
            else:
                raise NotImplementedError(f"Sampler {sampler} not supported")

    def forward(self, pxb):
        """
        Nhận vào tuple/list pxb = (p, x, b):
            p: [Total_N, 3]  -> Tọa độ các điểm cũ
            x: [Total_N, C]  -> Đặc trưng các điểm cũ
            b: [Total_N]     -> Chỉ số batch
        Trả về:
            [new_p, new_x, new_b]: Tọa độ, đặc trưng và batch index của tập điểm sau khi hạ mẫu
        """
        p, x, b = pxb

        # Trường hợp 1: Tầng Head (chỉ biến đổi đặc trưng kênh, không đổi không gian)
        if self.is_head:
            x = self.convs(x)
            return p, x, b

        # Trường hợp 2: Có hạ mẫu không gian (stride > 1) hoặc gom toàn bộ
        if not self.all_aggr:
            # 1. LẤY MẪU THEO TỶ LỆ PHẦN TRĂM (RATIO = 1 / stride)
            # PyG tự động duyệt từng protein và lấy đúng % số điểm của protein đó
            idx = self.sample_fn(p, b, ratio=(1.0 / self.stride))
            new_p = p[idx]
            new_b = b[idx]
        else:
            idx = None
            new_p = p
            new_b = b

        # 2. Nhánh tắt phần dư (Skip connection)
        if self.use_res:
            identity = x[idx]  # Lấy đặc trưng của các điểm tâm được chọn
            identity = self.skipconv(identity)
        else:
            identity = None

        # 3. QUÉT TÌM LÂN CẬN (Bipartite Graph: từ tập cũ p -> tập tâm mới new_p)
        # row: chỉ số điểm tâm (thuộc new_p), col: chỉ số điểm lân cận (thuộc p)
        row, col = pyg_radius(
            x=p, y=new_p,
            r=self.radius,
            batch_x=b, batch_y=new_b,
            max_num_neighbors=self.nsample
        )

        # 4. Tọa độ tương đối chuẩn hóa
        dp = p[col] - new_p[row]
        if self.normalize_dp:
            dp = dp / self.radius

        # 5. Ghép đặc trưng theo feature_type
        xj = x[col]
        if self.feature_type == 'dp_fj':
            edge_features = torch.cat([dp, xj], dim=-1)
        elif self.feature_type == 'fj':
            edge_features = xj
        elif self.feature_type == 'dp':
            edge_features = dp
        else:
            raise NotImplementedError(f"Feature type {self.feature_type} not implemented")

        # 6. Biến đổi đặc trưng cạnh qua MLP
        edge_features = self.convs(edge_features)

        # 7. Max Pooling về các điểm tâm mới
        # CỰC KỲ QUAN TRỌNG: dim_size=new_p.size(0) bảo vệ không bị lệch shape khi có điểm cô lập
        new_x = scatter(edge_features, row, dim=0, dim_size=new_p.size(0), reduce='max')

        # 8. Cộng nhánh tắt phần dư (Residual addition)
        if self.use_res:
            new_x = self.act(new_x + identity)

        return new_p, new_x, new_b

@MODELS.register_module()
class PointNextEncoderPyG(nn.Module):
    r"""The PyG-compatible Encoder for PointNeXt supporting variable-sized point clouds.
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.

    Tương thích 100% với tham số YAML của OpenPoints gốc.
    """

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
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            # Cho phép giải mã tên class 'InvResMLP' truyền từ YAML
            block = globals()[block] if block in globals() else eval(block)

        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
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

        # 1. Tự động tính toán bán kính và nsample nở rộng qua từng Stage
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'[PointNextEncoderPyG] Radii: {self.radii}, Nsample: {self.nsample}')

        # 2. Quy tắc nhân đôi số kênh mỗi khi gặp stride > 1
        channels = []
        curr_w = width
        for stride in strides:
            if stride != 1:
                curr_w *= 2
            channels.append(curr_w)

        # 3. Lắp ráp các Stage tuần tự
        encoder = []
        for i in range(len(blocks)):
            # Deepcopy group_args để không bị ghi đè tham số giữa các stage
            stage_group_args = copy.deepcopy(group_args)
            stage_group_args['radius'] = self.radii[i]
            stage_group_args['nsample'] = self.nsample[i]

            encoder.append(self._make_enc(
                block=block,
                channels=channels[i],
                blocks=blocks[i],
                stride=strides[i],
                group_args=stage_group_args,
                is_head=(i == 0 and strides[i] == 1)
            ))

        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        """Helper tính toán danh sách tham số (radius hoặc nsample) cho từng block trong stage."""
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
        """Xây dựng 1 Stage hoàn chỉnh: 1 SetAbstraction ở đầu + (blocks - 1) InvResMLP phía sau."""
        layers = []
        radii = group_args['radius']
        nsample = group_args['nsample']
        
        # Cấu hình bán kính cho khối SetAbstraction đầu tiên của Stage
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

        # Các khối InvResMLP tiếp theo trong Stage (giữ nguyên số điểm)
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
        """Trích xuất đặc trưng cho bài toán phân loại (Classification)."""
        # 1. Tự động bóc tách nếu đầu vào là dict (từ concat_collate_fn) hoặc PyG Batch
        if hasattr(p0, 'keys'):
            p0, f0, batch0 = p0['pos'], p0.get('x', None), p0.get('batch', None)
        elif hasattr(p0, 'pos'):
            p0, f0, batch0 = p0.pos, getattr(p0, 'x', None), getattr(p0, 'batch', None)

        # Baseline: Nếu f0 không có, lấy chính tọa độ (x, y, z) làm đặc trưng khởi đầu
        if f0 is None:
            f0 = p0.clone()

        # 2. Cho dữ liệu đi qua lần lượt từng Stage
        p, x, b = p0, f0, batch0
        for stage in self.encoder:
            p, x, b = stage([p, x, b])

        # 3. Trả về bộ 3 sau cùng để Classifier Head làm Global Pooling
        return p, x, b

    def forward(self, p0, f0=None, batch0=None):
        # Mặc định gọi hàm trích xuất đặc trưng phân loại
        return self.forward_cls_feat(p0, f0, batch0)