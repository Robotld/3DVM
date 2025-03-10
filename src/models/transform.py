"""
医学图像处理和增强的转换模块
提供基于MONAI的3D医学图像预处理、增强和归一化功能
"""

import numpy as np
from monai.transforms import (
    Compose, RandRotate90d, RandShiftIntensityd,
    RandScaleIntensityd, RandGaussianNoised, LoadImaged,
    EnsureChannelFirstd, ScaleIntensityd, CenterSpatialCropd,
    RandAffined, RandGaussianSmoothd, Orientationd,
    SpatialPadd, ToTensord, NormalizeIntensityd
)


def create_transforms(config, args):
    """
    创建MONAI转换管道，用于3D医学图像的增强、裁剪和归一化

    Args:
        config: 模型配置对象
        args: 命令行参数对象

    Returns:
        tuple: (train_transforms, val_transforms) - 分别用于训练和验证的转换管道
    """
    # 获取模型参数中的图像大小
    image_size = config.model["params"]["image_size"]

    # 解析中心裁剪尺寸
    if args.center_crop:
        try:
            crop_size = tuple(map(int, args.center_crop.split('x')))
            if len(crop_size) != 3:
                print(f"警告: 中心裁剪尺寸格式错误，使用默认值 ({image_size}x{image_size}x{image_size})")
                crop_size = (image_size, image_size, image_size)
        except:
            print(f"警告: 无法解析中心裁剪尺寸，使用默认值 ({image_size}x{image_size}x{image_size})")
            crop_size = (image_size, image_size, image_size)
    else:
        crop_size = (image_size, image_size, image_size)

    print(f"使用裁剪尺寸: {crop_size}")

    # 基本转换 (训练和验证都需要)
    base_transforms = [
        # 确保数据格式为通道优先
        EnsureChannelFirstd(keys=["image"]),
        # 强度归一化
        NormalizeIntensityd(keys=["image"]),
        # 统一方向
        Orientationd(keys=["image"], axcodes="RAS"),
        # 中心裁剪
        CenterSpatialCropd(keys=["image"], roi_size=crop_size),
        # 如果尺寸小于目标尺寸，进行填充
        SpatialPadd(keys=["image"], spatial_size=crop_size),
        # 转为Tensor
        ToTensord(keys=["image", "label"]),
    ]

    # 仅用于训练数据的增强转换
    if args.augment:
        print("启用数据增强...")
        train_transforms = [
            # 随机90度旋转
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
            # 随机仿射变换 (旋转、缩放、平移)
            RandAffined(
                keys=["image"],
                prob=0.5,
                rotate_range=(np.pi / 20, np.pi / 20, np.pi / 20),  # 小角度旋转
                scale_range=(0.1, 0.1, 0.1),  # 小范围缩放
                translate_range=(10, 10, 10),  # 小范围平移
                mode=("bilinear"),  # 插值模式
                padding_mode="zeros"  # 填充模式
            ),
            # 随机高斯滤波
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
            # 随机强度缩放
            RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1),
            # 随机强度偏移
            RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
            # 随机高斯噪声
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
        ]
        # 组合基本转换和增强转换
        train_transforms = Compose(train_transforms + base_transforms)
        val_transforms = Compose(base_transforms)
    else:
        # 不使用增强时，训练和验证使用相同的转换
        train_transforms = Compose(base_transforms)
        val_transforms = Compose(base_transforms)

    return train_transforms, val_transforms