import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import rotate, zoom, gaussian_filter


class To3DTensor:
    """Convert a 4D numpy array to a torch tensor."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if not isinstance(x, np.ndarray):
            raise TypeError(f'Expected numpy.ndarray, got {type(x)}')

        if x.ndim != 4:  # (C, D, H, W)
            raise ValueError(f'Expected 4 dimensions (C, D, H, W), got {x.ndim}')

        x = torch.from_numpy(x)
        return x


class Normalize3D:
    """Standard normalization for 3D medical images."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 由于已经调窗，只需要简单归一化
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x


class RandomRotation3D:
    """Randomly rotate the volume with small angles."""

    def __init__(self, angle_range: float = 10.0, p: float = 0.5):
        self.angle_range = angle_range
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            # 只在横断面(axial plane)上进行小角度旋转
            angle = np.random.uniform(-self.angle_range, self.angle_range)
            x = rotate(x, angle, axes=(2, 3), reshape=False, mode='nearest')
        return x


class GaussianNoise3D:
    """Add small amount of Gaussian noise."""

    def __init__(self, sigma: float = 0.01, p: float = 0.3):
        self.sigma = sigma
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            noise = np.random.normal(0, self.sigma, x.shape)
            x = x + noise
            # 保持值域在[0,1]
            x = np.clip(x, 0, 1)
        return x


class RandomIntensityShift:
    """Randomly shift intensity values slightly."""

    def __init__(self, shift_range: float = 0.1, p: float = 0.3):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            x = x + shift
            x = np.clip(x, 0, 1)  # 保持在有效范围内
        return x


def make_3d_transform(train: bool = True):
    """Create transformation pipeline for 3D medical images."""
    transforms = []

    if train:
        # 训练时的数据增强
        transforms.extend([
            RandomRotation3D(angle_range=10.0, p=0.5),  # 小角度旋转
            GaussianNoise3D(sigma=0.01, p=0.3),  # 添加少量噪声
            RandomIntensityShift(shift_range=0.1, p=0.3)  # 轻微的强度偏移
        ])

    # 基础处理（训练和验证都需要）
    transforms.extend([
        To3DTensor(),
        Normalize3D()
    ])

    return Compose3D(transforms)


class Compose3D:
    """Compose several transforms together."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img