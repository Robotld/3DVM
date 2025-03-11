import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, List


class NoduleDataset(Dataset):
    """
    以字典形式返回路径与标签，供 MONAI 字典式变换使用。
    注意：此版本并不直接加载图像，而是将文件路径作为 'image' 字典键的值交给 MONAI 的 LoadImaged 等变换处理。
    """

    def __init__(self, root_dirs: Union[str, List[str]], transform=None):
        """
        参数:
            root_dirs: 字符串列表，包含所有数据集路径，或者单个字符串路径
            transform: 数据转换操作 (MONAI 字典变换)
        """
        self.transform = transform
        self.samples = []

        # 处理 root_dirs 可能来自 YAML 的情况
        # 确保 root_dirs 是一个正确的 Python 列表或字符串
        if hasattr(root_dirs, '__iter__') and not isinstance(root_dirs, (str, bytes)):
            # 如果是可迭代对象但不是字符串，将其转换为普通 Python 列表
            root_dirs = list(root_dirs)
        elif isinstance(root_dirs, str):
            # 如果是单个字符串路径，放入列表
            root_dirs = [root_dirs]

        print(f"处理数据集路径: {root_dirs}")

        # 遍历每个数据集路径
        for root_dir in root_dirs:
            # 确保 root_dir 是字符串
            root_dir = str(root_dir)

            if not os.path.exists(root_dir):
                print(f"警告: 路径不存在 - {root_dir}")
                continue

            try:
                for class_folder in os.listdir(root_dir):
                    class_path = os.path.join(root_dir, class_folder)
                    if os.path.isdir(class_path):
                        try:
                            class_label = int(class_folder)
                            for file_name in os.listdir(class_path):
                                if file_name.endswith(('.nii', '.nii.gz')):
                                    self.samples.append((
                                        os.path.join(class_path, file_name),
                                        class_label
                                    ))
                        except ValueError:
                            print(f"警告: 类别文件夹名称无法转为整数: {class_folder}")
            except Exception as e:
                print(f"处理目录时出错 {root_dir}: {str(e)}")

        print(f"找到样本总数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 将 0,1,2 类统一为 0，其余为 1
        if label in (0, 1, 2):
            label = 0
        else:
            label = 1

        # 注意：这里仅保存图像路径，而不是加载图像数据
        # 交给 MONAI 的转变管线 (LoadImaged) 等步骤去解析和加载
        data_dict = {
            "image": img_path,
            "label": label
        }

        # 如果定义了 transform，则在 data_dict 上执行
        if self.transform:
            data_dict = self.transform(data_dict)

        # 返回经过变换后的图像和标签
        return data_dict["image"], data_dict["label"]