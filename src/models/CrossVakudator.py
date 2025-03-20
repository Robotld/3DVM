from collections import Counter

import numpy as np
from monai.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, Dataset

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


def create_balanced_sampler(dataset):
    # 获取所有标签
    all_labels = [sample['label'] for sample in dataset.samples]

    # 计算类别权重（反比于频率）
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in all_labels]

    # 创建WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(all_labels),
        replacement=True
    )

    return sampler


class CrossValidator:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config.cross_validation
        self.train_config = config.training

        # 获取所有标签用于分层采样 - 使用缓存优化
        self.all_labels = np.array([label for _, label in dataset])

        # 使用Counter更快统计
        label_counter = Counter(self.all_labels)
        total_samples = len(self.all_labels)

        print("\n------- 数据集类别分布 -------")
        for label, count in sorted(label_counter.items()):
            percentage = count / total_samples * 100
            print(f"类别 {label}: {count} 个样本 ({percentage:.2f}%)")
        print("----------------------------\n")

        # 初始化分层K折交叉验证器
        self.skf = StratifiedKFold(
            n_splits=self.config["n_splits"],
            shuffle=self.config["shuffle"],
            random_state=self.train_config["random_seed"]
        )

    def get_folds(self, train_transforms=None, val_transforms=None):
        """
        生成交叉验证的折，并应用适当的变换
        参数:
            train_transforms: 训练集使用的变换
            val_transforms: 验证集使用的变换
        """
        # 预计算所有fold，避免重复计算
        all_splits = list(self.skf.split(np.zeros(len(self.all_labels)), self.all_labels))

        for fold, (train_idx, val_idx) in enumerate(all_splits):
            # 创建训练集和验证集的子集，但还不应用变换
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            # 创建包含变换的训练集和验证集
            train_dataset_with_transform = TransformDataset(train_subset, transform=train_transforms)
            val_dataset_with_transform = TransformDataset(val_subset, transform=val_transforms)

            # 高效获取标签
            train_labels = self.all_labels[train_idx]
            val_labels = self.all_labels[val_idx]

            # 使用Counter高效统计
            train_counter = Counter(train_labels)
            val_counter = Counter(val_labels)

            total_train = len(train_labels)
            total_val = len(val_labels)

            print(f"\n------- 第 {fold + 1} 折的类别分布 -------")
            print("训练集:")
            for label, count in sorted(train_counter.items()):
                percentage = count / total_train * 100
                print(f"  类别 {label}: {count} 个样本 ({percentage:.2f}%)")

            print("验证集:")
            for label, count in sorted(val_counter.items()):
                percentage = count / total_val * 100
                print(f"  类别 {label}: {count} 个样本 ({percentage:.2f}%)")
            print("-----------------------------------\n")

            # 使用pin_memory和persistent_workers加速数据加载
            train_loader = DataLoader(
                train_dataset_with_transform,  # 使用带变换的训练集
                batch_size=self.train_config["batch_size"],
                shuffle=True,
                num_workers=self.train_config["num_workers"],
                pin_memory=True,
                persistent_workers=True if self.train_config["num_workers"] > 0 else False
            )

            val_loader = DataLoader(
                val_dataset_with_transform,  # 使用带变换的验证集
                batch_size=self.train_config["batch_size"] * 2,  # 验证时可用更大的batch size
                shuffle=False,
                num_workers=self.train_config["num_workers"],
                pin_memory=True,
                persistent_workers=True if self.train_config["num_workers"] > 0 else False
            )

            yield fold, train_loader, val_loader, train_counter


# 添加一个辅助类，用于在Subset上应用变换
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # 复制原始数据集的属性
        if hasattr(dataset, '_data_counter'):
            self._data_counter = dataset._data_counter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从原始数据集中获取数据
        img_path, label = self.dataset[idx]

        # 如果有变换，应用它 (MONAI需要字典格式)
        if self.transform:
            # 需要先创建字典格式
            data_dict = {
                "image": img_path,  # 路径字符串
                "label": label  # 数值标签
            }
            # MONAI的LoadImaged会自动加载图像
            transformed = self.transform(data_dict)
            # 确保返回的是张量，不是字典
            return transformed["image"], transformed["label"]

        # 如果没有变换，返回原始数据
        return img_path, label
