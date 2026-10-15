import torch
import torch.nn as nn
import logging
from typing import List
from torch_scatter import scatter_softmax
from torch_geometric.nn import global_mean_pool, global_max_pool
from ..layers import create_linearblock
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ...loss import build_criterion_from_cfg
from ...utils import load_checkpoint


@MODELS.register_module()
class BaseCls(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)

        if cls_args is not None:
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

    def forward(self, data):
        global_feat = self.encoder.forward_cls_feat(data)
        return self.prediction(global_feat)

    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt.long())

    def get_logits_loss(self, data, gt):
        logits = self.forward(data)
        return logits, self.criterion(logits, gt.long())


@MODELS.register_module()
class DistillCls(BaseCls):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 distill_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__(encoder_args, cls_args, criterion_args)
        self.distill = encoder_args.get('distill', True)
        in_channels = self.encoder.distill_channels
        distill_args.distill_head_args.in_channels = in_channels
        self.dist_head = build_model_from_cfg(distill_args.distill_head_args)
        self.dist_model = build_model_from_cfg(distill_args).cuda()
        load_checkpoint(self.dist_model, distill_args.pretrained_path)
        self.dist_model.eval()

    def forward(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        if self.distill and self.training:
            global_feat, distill_feature = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat), self.dist_head(distill_feature)
        else:
            global_feat = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat)

    def get_loss(self, pred, gt, inputs):
        return self.criterion(inputs, pred, gt.long(), self.dist_model)

    def get_logits_loss(self, data, gt):
        logits, dist_logits = self.forward(data)
        return logits, self.criterion(data, [logits, dist_logits], gt.long(), self.dist_model)


@MODELS.register_module()
class ClsHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 global_feat: str=None,
                 point_dim: int=2,
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            num_classes (int): class num
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature.
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features.
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.global_feat = global_feat.split(',') if global_feat is not None else None
        self.point_dim = point_dim
        in_channels = len(self.global_feat) * in_channels if global_feat is not None else in_channels
        if mlps is not None:
            mlps = [in_channels] + mlps + [num_classes]
        else:
            mlps = [in_channels, num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)


    def forward(self, end_points):
        if self.global_feat is not None:
            global_feats = []
            for preprocess in self.global_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=self.point_dim, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=self.point_dim, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)
        logits = self.head(end_points)
        return logits

@MODELS.register_module()
class ClsHeadPyG(nn.Module):
    """
    TẦNG 3: CLASSIFIER HEAD TÍCH HỢP GATED MULTIMODAL UNIT (GMU)
    - Nhánh 1 (Z_fold): Nếp gấp khung xương 3D toàn thể qua Mean Pooling
    - Nhánh 2 (Z_pocket): Túi phản ứng hoạt tính qua Attentive Pooling + Max Pooling
    - Dung hợp: Cổng động GMU tự học tỷ trọng cân bằng giữa Khung xương và Túi xúc tác.
    """
    def __init__(self,
                 num_classes: int = 7,
                 in_channels: int = 512,
                 mlps: List[int] = [512, 256],
                 norm_args: dict = {'norm': 'bn1d'},
                 act_args: dict = {'act': 'relu'},
                 dropout: float = 0.5,
                 **kwargs
                 ):
        super().__init__()
        
        # 1. Nhánh Pocket: Học trọng số chú ý để tìm túi xúc tác
        self.att_score = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1)
        )
        self.pocket_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Các tầng chiếu độc lập cho 2 nhánh trước khi vào GMU
        self.proj_fold = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_pocket = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 3. Cổng điều hòa GMU (Gated Multimodal Unit)
        self.gmu_gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid()
        )

        # 4. MLP phân loại chính (đưa từ in_channels về num_classes)
        heads = []
        curr_c = in_channels
        for h_dim in mlps:
            heads.append(create_linearblock(curr_c, h_dim, norm_args=norm_args, act_args=act_args))
            if dropout > 0:
                heads.append(nn.Dropout(dropout))
            curr_c = h_dim
            
        heads.append(create_linearblock(curr_c, num_classes, norm_args=None, act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        # 1. Bóc tách bộ ba (p, x, b) trả về từ PointNextEncoderPyG
        if isinstance(end_points, (list, tuple)):
            _, x, b = end_points
        else:
            x, b = end_points, None

        if b is not None:
            # === NHÁNH 1: HÌNH THÁI NẾP GẤP TOÀN THỂ (Z_fold) ===
            z_fold = global_mean_pool(x, b)  # [B, in_channels]

            # === NHÁNH 2: TÚI XÚC TÁC HOẠT TÍNH (Z_pocket) ===
            # Tính điểm chú ý cho từng điểm trong mỗi protein
            scores = self.att_score(x)                           # [Total_N, 1]
            attn_weights = scatter_softmax(scores, b, dim=0)     # Softmax phân tán theo vector batch
            z_attn = global_mean_pool(x * attn_weights, b)       # Điểm nào là túi xúc tác sẽ có trọng số cao
            z_max  = global_max_pool(x, b)                       # Lấy giá trị cực đại nổi bật
            
            # Kết hợp Attention và Max để mô tả toàn diện túi hoạt tính
            z_pocket_raw = torch.cat([z_attn, z_max], dim=-1)   # [B, 2 * in_channels]
            z_pocket = self.pocket_proj(z_pocket_raw)            # [B, in_channels]

            # === NHÁNH 3: DUNG HỢP BẰNG CỔNG GMU ===
            # Học tỷ trọng cân bằng alpha in (0, 1) cho từng protein
            gate_input = torch.cat([z_fold, z_pocket], dim=-1)   # [B, 2 * in_channels]
            alpha = self.gmu_gate(gate_input)                    # [B, in_channels]

            # Dung hợp đặc trưng: Điều hòa thích ứng giữa nếp gấp và túi xúc tác
            z_fused = alpha * self.proj_fold(z_fold) + (1.0 - alpha) * self.proj_pocket(z_pocket)
        else:
            z_fused = x

        # 4. Dự đoán xác suất 7 lớp Enzyme Commission
        logits = self.head(z_fused)
        return logits