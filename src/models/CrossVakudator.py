from collections import Counter
import numpy as np
from monai.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, Dataset
from .balanced_sampler import BalancedBatchSampler


class TransformDataset(Dataset):
    """应用MONAI转换的数据集包装器"""

    def __init__(self, dataset, transform=None, labels=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = labels  # 预计算的标签数据

        # 复制原数据集属性
        if hasattr(dataset, '_data_counter'):
            self._data_counter = dataset._data_counter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset[idx]
        if not self.transform:
            return img_path, label

        # 应用MONAI转换
        transformed = self.transform({"image": img_path, "label": label})
        return transformed["image"], transformed["label"]


def _print_fold_stats(fold, train_labels, val_labels):
    """打印当前折的类别统计"""
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print(f"\n------- 第 {fold + 1} 折的类别分布 -------")
    print("训练集:")
    for label, count in sorted(train_counts.items()):
        percentage = count / len(train_labels) * 100
        print(f"  类别 {label}: {count} 个样本 ({percentage:.2f}%)")

    print("验证集:")
    for label, count in sorted(val_counts.items()):
        percentage = count / len(val_labels) * 100
        print(f"  类别 {label}: {count} 个样本 ({percentage:.2f}%)")
    print("-----------------------------------\n")


def _print_sampler_stats(sampler):
    """打印采样器配置"""
    print("\n------- Batch采样配置 -------")
    print(f'批次数量: {len(sampler)}')
    print(f"批次大小: {sampler.batch_size}")
    print(f"类别数量: {sampler.n_classes}")
    print(f"每个类别的样本数: {sampler.samples_per_class}")
    for label, indices in sampler.class_indices.items():
        print(f"类别 {label}: 共有 {len(indices)} 个样本")
    print("---------------------------\n")


class CrossValidator:
    """K折交叉验证管理器"""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.cfg_cv = config.cross_validation
        self.cfg_train = config.training

        # 提取并缓存所有标签
        self.all_labels = np.array([label for _, label in dataset])

        # 打印数据集统计
        counts = Counter(self.all_labels)
        total = len(self.all_labels)

        print("\n------- 数据集类别分布 -------")
        for label, count in sorted(counts.items()):
            print(f"类别 {label}: {count}个样本 ({count / total * 100:.2f}%)")
        print("----------------------------\n")

        # 初始化分层K折
        self.skf = StratifiedKFold(
            n_splits = self.cfg_cv["n_splits"],
            shuffle = self.cfg_cv["shuffle"],
            random_state = self.cfg_train["random_seed"]
        )

        # 预计算所有分割，避免重复计算
        self.splits = list(self.skf.split(np.zeros(len(self.all_labels)), self.all_labels))


    def get_folds(self, train_transforms=None, val_transforms=None):
        """
       生成交叉验证的折，并应用适当的变换
       参数:
           train_transforms: 训练集使用的变换
           val_transforms: 验证集使用的变换
        """
        for fold, (train_idx, val_idx) in enumerate(self.splits):
            # 获取当前折的标签数据
            train_labels = self.all_labels[train_idx]
            val_labels = self.all_labels[val_idx]

            # 创建包含变换的训练集和验证集
            train_ds = TransformDataset(Subset(self.dataset, train_idx), train_transforms)
            val_ds = TransformDataset(Subset(self.dataset, val_idx), val_transforms)

            # 打印当前折的类别分布
            _print_fold_stats(fold, train_labels, val_labels)

            # 创建平衡采样器，传递预计算的标签
            sampler = BalancedBatchSampler(
                train_ds,
                self.cfg_train["batch_size"],
                labels = train_labels,
                indices = train_idx
            )

            # 打印批次采样配置
            _print_sampler_stats(sampler)

            # 创建数据加载器
            train_loader = DataLoader(
                train_ds,
                batch_sampler = sampler,
                num_workers = self.cfg_train["num_workers"],
                pin_memory = True,
                persistent_workers = bool(self.cfg_train["num_workers"] > 0)
            )

            val_loader = DataLoader(
                val_ds,
                batch_size = self.cfg_train["batch_size"],
                num_workers = self.cfg_train["num_workers"],
                pin_memory = True,
                persistent_workers = bool(self.cfg_train["num_workers"] > 0)
            )

            # 获取训练集标签分布
            train_counter = Counter(train_labels)

            yield fold, train_loader, val_loader, train_counter

