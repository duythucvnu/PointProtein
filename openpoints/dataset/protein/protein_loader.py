import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ..build import DATASETS, concat_collate_fn


@DATASETS.register_module()
class ProteinEC(Dataset):
    """
    Dataset loader nạp dữ liệu Protein Enzyme Commission từ các file .h5.
    Hỗ trợ độ dài biến thiên tự nhiên thông qua concat_collate_fn.
    """
    collate_fn = concat_collate_fn  # <-- BẬT TỰ ĐỘNG GOM BATCH BIẾN THIÊN CỦA OPENPOINTS

    def __init__(self,
                 data_dir="",
                 split='train',
                 use_features=False,  # False: Baseline chỉ dùng XYZ; True: Dùng 22 đặc trưng sinh hóa
                 transform=None,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.split = split.lower()
        self.use_features = use_features
        self.transform = transform
        
        self.h5_filepath = os.path.join(data_dir, f"{self.split}.h5")
        if not os.path.exists(self.h5_filepath):
            raise FileNotFoundError(f"Không tìm thấy file dataset: {self.h5_filepath}")

        with h5py.File(self.h5_filepath, "r") as f:
            self.offsets = f["offsets"][:]
            self.targets = f["targets"][:]
            self.length = len(self.targets)
            
        self.f = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_filepath, "r", swmr=True)

        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        coords = torch.from_numpy(self.f["coords"][start:end]).float()
        target = torch.tensor([self.targets[idx]], dtype=torch.long)

        if self.use_features:
            features = torch.from_numpy(self.f["features"][start:end]).float()
        else:
            features = coords.clone()

        data = {
            'pos': coords,     # [N_i, 3]
            'x': features,     # [N_i, 3] (or [N_i, 22])
            'y': target        # [1]
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def num_classes(self):
        return 7