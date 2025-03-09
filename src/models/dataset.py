import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class NoduleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 遍历文件夹获取所有样本
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                class_label = int(class_folder)
                for file_name in os.listdir(class_path):
                    if file_name.endswith(('.nii', '.nii.gz')):
                        self.samples.append((
                            os.path.join(class_path, file_name),
                            class_label
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if label in (0, 1, 2):
            label = 0
        else:
            label = 1

        # 使用SimpleITK读取图像
        reader = sitk.ImageFileReader()
        reader.SetFileName(img_path)
        image = reader.Execute()

        # 转换为numpy数组
        img_array = sitk.GetArrayFromImage(image)

        # 添加通道维度
        if img_array.ndim == 3:
            img_array = img_array[np.newaxis, ...]

        # 转换为float32并归一化到[0,1]
        img_array = img_array.astype(np.float32)
        if img_array.max() != img_array.min():
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

        # 应用转换
        if self.transform:
            img_array = self.transform(img_array)

        return img_array, label