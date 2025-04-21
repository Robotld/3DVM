import numpy as np
from torch.utils.data import Subset


class BalancedBatchSampler:
    """改进的批次类别平衡采样器，支持灵活采样比例和减少过采样"""

    def __init__(self, dataset, batch_size, labels=None, indices=None,
                 pos_ratio=0.2, max_batches=None):
        """
        初始化采样器
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            labels: 可选的标签数组
            indices: 可选的索引数组
            pos_ratio: 正样本(少数类)在每个批次中的比例，默认0.3
            max_batches: 每个epoch最大批次数，控制过采样程度
        """
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio

        # 处理数据集和索引
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices if indices is None else indices
        else:
            self.dataset = dataset
            self.indices = range(len(dataset)) if indices is None else indices

        # 获取标签
        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = np.array([self.dataset[idx][1] for idx in self.indices])

        # 分析类别
        self.unique_labels = np.unique(self.labels)
        self.n_classes = len(self.unique_labels)

        # 假设二分类问题中标签1为少数类(正例)
        if self.n_classes == 2:
            label_counts = {label: np.sum(self.labels == label) for label in self.unique_labels}
            self.minority_label = min(label_counts, key=label_counts.get)
            self.majority_label = max(label_counts, key=label_counts.get)
        else:
            # 多分类情况使用均衡采样
            self.pos_ratio = 1.0 / self.n_classes

        # 按类别分组索引
        self.class_indices = {
            label: np.where(self.labels == label)[0] for label in self.unique_labels
        }

        # 计算每个批次中各类别的样本数
        if self.n_classes == 2:
            self.samples_per_class = {
                self.minority_label: int(self.batch_size * self.pos_ratio),
                self.majority_label: self.batch_size - int(self.batch_size * self.pos_ratio)
            }
        else:
            self.samples_per_class = {
                label: self.batch_size // self.n_classes for label in self.unique_labels
            }
            # 处理除法余数
            remainder = self.batch_size % self.n_classes
            for i, label in enumerate(self.unique_labels):
                if i < remainder:
                    self.samples_per_class[label] += 1

        # 计算可生成的最大批次数（避免过度重采样）
        min_class_size = min(len(indices) for label, indices in self.class_indices.items())
        max_possible_batches = min_class_size // self.samples_per_class[self.minority_label] if self.n_classes == 2 else \
            min(len(indices) // self.samples_per_class[label] for label, indices in self.class_indices.items())

        # 允许适度重采样，但不过度
        reasonable_batches = int(max_possible_batches * 4)  # 允许少数类最多重复3次

        # 如果指定了最大批次数，使用较小值
        self.batches_per_epoch = reasonable_batches if max_batches is None else min(reasonable_batches, max_batches)

        print(f"每批次包含 {self.samples_per_class} 个样本")
        print(f"每个epoch生成 {self.batches_per_epoch} 个批次")

    def __iter__(self):
        # 为每个类别创建采样器
        indices_by_label = {}
        for label, indices in self.class_indices.items():
            # 复制并打乱索引
            shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices)
            indices_by_label[label] = shuffled_indices

        # 生成批次
        for _ in range(self.batches_per_epoch):
            batch = []

            # 从每个类别选择所需数量的样本
            for label, count in self.samples_per_class.items():
                # 如果当前类别的索引用完，重新打乱
                if len(indices_by_label[label]) < count:
                    new_indices = self.class_indices[label].copy()
                    np.random.shuffle(new_indices)
                    indices_by_label[label] = np.concatenate([indices_by_label[label], new_indices])

                # 添加样本到批次
                batch.extend(indices_by_label[label][:count])
                indices_by_label[label] = indices_by_label[label][count:]

            # 打乱批次内的顺序
            np.random.shuffle(batch)

            # 如果需要，转换回原始索引
            if hasattr(self.dataset, 'indices'):
                batch = [self.dataset.indices[i] for i in batch]

            yield batch

    def __len__(self):
        return self.batches_per_epoch