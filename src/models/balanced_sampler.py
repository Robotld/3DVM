import numpy as np
from torch.utils.data import Subset


class BalancedBatchSampler:
    """批次内类别平衡采样器"""

    def __init__(self, dataset, batch_size, labels=None, indices=None):
        self.batch_size = batch_size

        # 处理数据集和索引
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices if indices is None else indices
        else:
            self.dataset = dataset
            self.indices = range(len(dataset)) if indices is None else indices

        # 获取标签
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.array([self.dataset[idx][1] for idx in self.indices])

        # 分析类别
        self.unique_labels = np.unique(self.labels)
        self.n_classes = len(self.unique_labels)
        self.samples_per_class = batch_size // self.n_classes

        # 按类别分组索引
        self.class_indices = {
            label: np.where(self.labels == label)[0] for label in self.unique_labels
        }

        # 计算最大类别样本数
        self.max_samples = max(len(indices) for indices in self.class_indices.values())

        # 扩充后的总样本数 = 最大类别样本数 × 类别数
        total_balanced_samples = self.max_samples * self.n_classes

        # 批次数 = 扩充后的总样本数 / 批次大小
        self.batches_per_epoch = total_balanced_samples // self.batch_size


    def __iter__(self):
        # 创建扩充后的类别采样器
        samplers = {}
        for label, indices in self.class_indices.items():
            # 如果类别样本不足，则通过重复扩充
            if len(indices) < self.max_samples:
                # 重复采样直到达到最大类别的样本数
                repeats = int(np.ceil(self.max_samples / len(indices)))
                sampler = np.tile(indices, repeats)[:self.max_samples]
                np.random.shuffle(sampler)
            else:
                # 多数类别直接打乱
                sampler = np.array(indices)
                np.random.shuffle(sampler)

            samplers[label] = sampler

        # 生成批次
        for i in range(self.batches_per_epoch):
            batch = []

            # 从每个类别选择样本
            for label, sampler in samplers.items():
                start = i * self.samples_per_class % len(sampler)
                # 如果剩余样本不足
                if start + self.samples_per_class > len(sampler):
                    # 使用剩余部分 + 从头开始的部分
                    remaining = len(sampler) - start
                    batch.extend(sampler[start:])
                    batch.extend(sampler[:self.samples_per_class - remaining])
                else:
                    # 直接选择连续样本
                    batch.extend(sampler[start:start + self.samples_per_class])

            # 打乱批次内顺序
            np.random.shuffle(batch)

            # 如果需要，转换回原始索引
            if isinstance(self.dataset, Subset):
                batch = [self.indices[i] for i in batch]

            yield batch

    def __len__(self):
        return self.batches_per_epoch