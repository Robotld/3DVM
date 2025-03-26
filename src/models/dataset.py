import csv
import glob
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

    def __init__(self, root_dirs: Union[str, List[str]], num_classes, transform=None):
        """
        参数:
            root_dirs: 字符串列表，包含所有数据集路径，或者单个字符串路径
            transform: 数据转换操作 (MONAI 字典变换)
        """
        self.transform = transform
        self.samples = []

        class_map2 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        class_map3 = {0, 1, 2}
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
                            class_label = int(class_folder)  # 默认5分类
                            if num_classes == 2: # 2分类   将 0 1 2 看作一类  3 4 看作一类   0 1 2是浸润性   3 4 是浸润前
                                class_label = class_map2[class_label]
                            elif num_classes == 3:
                                if class_label not in class_map3:   # 不读取这一类
                                    continue

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
        # 注意：这里仅保存图像路径，而不是加载图像数据
        # 交给 MONAI 的转变管线 (LoadImaged) 等步骤去解析和加载
        data_dict = {
            "image": img_path,
            "label": label
        }

        # 如果定义了 transform，则在 data_dict 上执行
        if self.transform:
            data_dict = self.transform(data_dict)

        # 返回经过变换后的图像路径和标签
        return data_dict["image"], data_dict["label"]


import csv
import os
import glob
from typing import Union, List
from torch.utils.data import Dataset


class RecurrenceDataset(Dataset):
    """
    术后复发预测数据集类，支持多个数据根目录
    """

    def __init__(self, csv_path: str, root_dirs: Union[str, List[str]], transform=None):
        """
        参数:
            csv_path: 随访数据CSV文件路径
            root_dirs: 数据集根目录，包含所有CT图像（字符串或字符串列表）
            transform: 数据转换操作
        """
        self.transform = transform
        self.samples = []

        # 读取CSV文件，创建CT号到复发标签的映射
        ct_label_map = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                ct_number = row['CT号']
                recurrence = 1 if row['有无复发'] == '有' else 0
                ct_label_map[ct_number] = recurrence

        # print(ct_label_map)
        # 确保root_dirs是列表
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        # 遍历所有根目录，递归获取所有nii.gz文件
        for root_dir in root_dirs:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 获取CT号
                    if '-' in file:
                        ct_number = file.split('-')[0]
                    else:
                        ct_number = file.split('.')[0]
                    # print(ct_number)
                    # 如果CT号在CSV中存在，则加入样本列表
                    if ct_number in ct_label_map.keys():
                        # print("11111")
                        self.samples.append({
                            'image_path': file_path,
                            'label': ct_label_map[ct_number]
                        })

        # 打印统计信息
        recurrence_count = sum(1 for sample in self.samples if sample['label'] == 1)
        print(f"找到匹配的样本总数: {len(self.samples)}")
        print(f"复发样本数: {recurrence_count}")
        print(f"非复发样本数: {len(self.samples) - recurrence_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        data_dict = {
            "image": sample['image_path'],
            "label": sample['label']
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict["image"], data_dict["label"]

# RecurrenceDataset(r"E:\workplace\3D\datasets\CTreport\协和预后随访复发与否.csv", )
