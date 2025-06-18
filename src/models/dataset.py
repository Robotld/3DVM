import csv

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

        class_map2 = {'0': 1, '1': 1, '2': 1, '3': 0, '4': 0, 'IA':1, 'AIS':0, 'MIA':0}     # 将浸润性作为正类-阳性    浸润前作为负类-阴性
        class_map3 = {'0', '1', '2'}
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
                        class_label = class_folder  # 默认5分类
                        if num_classes == 2: # 2分类   将 0 1 2 看作一类  3 4 看作一类   0 1 2是浸润性   3 4 是浸润前
                            class_label = class_map2[class_label]
                        elif num_classes == 3:
                            if class_label not in class_map3:   # 不读取这一类
                                continue
                        elif num_classes == 5:
                            class_label = int(class_folder)
                        for file_name in os.listdir(class_path):
                            if file_name.endswith(('.nii', '.nii.gz')):
                                self.samples.append((
                                    os.path.join(class_path, file_name),
                                    class_label
                                ))
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


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from collections import defaultdict


# class MultimodalRecurrenceDataset(Dataset):
#     """多模态肺癌复发预测与亚型分类多任务数据集，仅包含有复发标签的样本"""
#
#     def __init__(self, csv_path, root_dirs, transform=None, text_tokenizer=None, max_length=512):
#         """
#         参数:
#             csv_path: 随访数据CSV文件路径
#             root_dirs: 数据集根目录，包含所有CT图像（字符串或字符串列表）
#             transform: 图像转换操作
#             text_tokenizer: 文本分词器
#             max_length: 文本最大长度
#         """
#         self.transform = transform
#         self.text_tokenizer = text_tokenizer
#         self.max_length = max_length
#         self.samples = []
#         self._data_counter = None  # 用于存储交叉验证中的数据计数情况
#
#         # 读取CSV文件
#         df = pd.read_csv(csv_path, encoding='utf-8')
#
#         # 确保root_dirs是列表
#         if isinstance(root_dirs, str):
#             root_dirs = [root_dirs]
#
#         # 处理性别，转为数值
#         df['性别_数值'] = df['性别'].map({'男': 1, '女': 0})
#
#         # 处理年龄，去除"岁"字并转为数值
#         df['年龄_数值'] = df['年龄'].str.replace('岁', '').astype(float)
#
#         # 标准化年龄（0-1）
#         age_min, age_max = df['年龄_数值'].min(), df['年龄_数值'].max()
#         df['年龄_标准化'] = (df['年龄_数值'] - age_min) / (age_max - age_min)
#
#
#         # 处理复发标签
#         df['复发标签'] = df['有无复发'].map({'有': 1, '无': 0})
#
#         # 构建CT号到样本信息的映射
#         ct_info_map = {}
#         # 统计有复发标签的CT号集合
#         ct_with_recurrence_label = set()
#
#         for idx, row in df.iterrows():
#             ct_number = row['CT号']  # 假设列名是"术前最近一次CT号"，请根据实际情况调整
#             if pd.notna(ct_number) and ct_number.strip():  # 确保CT号不为空
#                 recurrence_label = row['复发标签']
#
#                 # 记录所有有复发标签的CT号
#                 ct_with_recurrence_label.add(ct_number)
#
#                 ct_info_map[ct_number] = {
#                     'label': recurrence_label,
#                     # 如果病理报告和CT报告均存在，则用换行符连接；如果某个报告为空，则不影响结果
#                     'report': (row['病理报告'] if pd.notna(row['病理报告']) else "") + "\n" + (
#                         row['CT报告'] if pd.notna(row['CT报告']) else ""),
#                     'gender': row['性别_数值'],
#                     'age': row['年龄_标准化'],
#                 }
#
#         # 用于记录找到CT图像的CT号
#         found_ct_numbers = set()
#         # 根据亚型记录找到的CT图像数量
#         found_by_subtype = defaultdict(int)
#
#         # 遍历所有根目录，收集图像路径和肿瘤亚型标签
#         for root_dir in root_dirs:
#             for subtype in range(5):  # 0-4 分别代表低分化、中分化、高分化、微浸润、原位癌
#                 subtype_dir = os.path.join(root_dir, str(subtype))
#                 if not os.path.exists(subtype_dir):
#                     continue
#
#                 for file in os.listdir(subtype_dir):
#                     if file.endswith('.nii.gz'):
#                         file_path = os.path.join(subtype_dir, file)
#
#                         # 提取CT号
#                         # 尝试从文件名中提取CT号
#                         match = re.search(r'(UCT\d+)', file)
#                         if match:
#                             ct_number = match.group(1)
#                         else:
#                             # 如果没有UCT编号，使用文件名前缀作为标识
#                             ct_number = file.split('.')[0]
#
#                         # 只添加有复发标签的样本
#                         if ct_number in ct_info_map:
#                             info = ct_info_map[ct_number]
#                             self.samples.append({
#                                 'image_path': file_path,
#                                 'report': info['report'],
#                                 'gender': info['gender'],
#                                 'age': info['age'],
#                                 'recurrence_label': info['label'],
#                                 'subtype_label': subtype,  # 肿瘤亚型标签
#                                 'name': info.get('name', ""),
#                                 'ct_number': ct_number
#                             })
#                             found_ct_numbers.add(ct_number)
#                             found_by_subtype[subtype] += 1
#
#         # 找出有复发标签但没有CT图像的CT号
#         missing_ct_numbers = ct_with_recurrence_label - found_ct_numbers
#
#         # 打印统计信息
#         print(f"\n===== 数据集统计信息 =====")
#         print(f"CSV中有复发标签的CT号总数: {len(ct_with_recurrence_label)}")
#         print(f"找到CT图像的样本数: {len(found_ct_numbers)}")
#         print(f"有复发标签但找不到CT图像的CT号数: {len(missing_ct_numbers)}")
#
#         if missing_ct_numbers:
#             print("\n以下CT号有复发标签但找不到对应的CT图像:")
#             for ct_number in sorted(missing_ct_numbers):
#                 print(f"  - {ct_number}")
#
#         # 按复发状态统计
#         recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 1)
#         non_recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 0)
#
#         print(f"\n有效样本总数: {len(self.samples)}")
#         print(f"复发样本数: {recurrence_count}")
#         print(f"非复发样本数: {non_recurrence_count}")
#
#         # 打印亚型分布
#         print("\n各亚型样本分布:")
#         for i in range(5):
#             count = found_by_subtype[i]
#             print(f"  亚型 {i} 样本数: {count}")
#         print("============================\n")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#
#         # 准备文本特征
#         report_text = sample['report']
#         if self.text_tokenizer and report_text:
#             # 使用tokenizer处理文本
#             text_encoding = self.text_tokenizer(
#                 report_text,
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             # 压缩第一维
#             text_inputs = {k: v.squeeze(0) for k, v in text_encoding.items()}
#         else:
#             # 如果没有提供tokenizer或没有文本，创建空的输入
#             if self.text_tokenizer:
#                 text_encoding = self.text_tokenizer(
#                     "",
#                     max_length=self.max_length,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt'
#                 )
#                 text_inputs = {k: v.squeeze(0) for k, v in text_encoding.items()}
#             else:
#                 text_inputs = {}
#
#         # 准备人口学特征
#         demographic = torch.tensor([sample['age'], sample['gender']], dtype=torch.float32)
#
#         data_dict = {
#             'report_text': report_text,
#             "image": sample['image_path'],
#             "text": text_inputs,
#             "demographic": demographic,
#             "recurrence_label": sample['recurrence_label'],
#             "subtype_label": sample['subtype_label'],
#             "ct_number": sample['ct_number']
#         }
#
#         # 应用图像转换
#         if self.transform:
#             # 转换只应用于图像部分
#             image_dict = {"image": data_dict["image"], "label": data_dict["recurrence_label"]}
#             transformed = self.transform(image_dict)
#             data_dict["image"] = transformed["image"]
#
#         return data_dict


import os
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict

# class MultimodalRecurrenceDataset(Dataset):
#     """多模态肺癌复发预测与亚型分类多任务数据集，支持结节大小和多发字段"""
#
#     def __init__(self, csv_path, root_dirs, transform=None, text_tokenizer=None, max_length=512):
#         self.transform = transform
#         self.text_tokenizer = text_tokenizer
#         self.max_length = max_length
#         self.samples = []
#         self._data_counter = None  # 用于存储交叉验证中的数据计数情况
#
#         # 读取CSV文件
#         df = pd.read_csv(csv_path, encoding='utf-8')
#
#         # 确保root_dirs是列表
#         if isinstance(root_dirs, str):
#             root_dirs = [root_dirs]
#
#         # 性别映射
#         df['性别_数值'] = df['性别'].map({'男': 1, '女': 0}).fillna(0)
#         df['年龄_数值'] = df['年龄'].astype(str).str.replace('岁', '', regex=False).astype(float)
#
#         # 结节大小（最大径）归一化（自动处理异常或缺失）
#         df['结节大小'] = pd.to_numeric(df.get('结节大小', None), errors='coerce')
#         nodule_min, nodule_max = df['结节大小'].min(), df['结节大小'].max()
#         df['结节大小_标准化'] = (df['结节大小'] - nodule_min) / (nodule_max - nodule_min)
#         df['结节大小_标准化'] = df['结节大小_标准化'].fillna(0)
#
#         # 是否多发映射
#         df['是否多发_数值'] = df['是否多发'].map({'是': 1, '否': 0}).fillna(0)
#
#         # 年龄归一化
#         age_min, age_max = df['年龄_数值'].min(), df['年龄_数值'].max()
#         df['年龄_标准化'] = (df['年龄_数值'] - age_min) / (age_max - age_min)
#
#         # 复发标签
#         df['复发标签'] = df['有无复发'].map({'有': 1, '无': 0})
#
#         # 构建CT号到样本信息的映射
#         ct_info_map = {}
#         ct_with_recurrence_label = set()
#         for idx, row in df.iterrows():
#             ct_number = row['CT号']
#             if pd.notna(ct_number) and ct_number.strip():
#                 recurrence_label = row['复发标签']
#                 ct_with_recurrence_label.add(ct_number)
#                 # 人口学特征：年龄_标准化、性别_数值、结节大小_标准化、是否多发_数值
#                 ct_info_map[ct_number] = {
#                     'label': recurrence_label,
#                     'report': (row['病理报告'] if pd.notna(row['病理报告']) else "") + "\n" +
#                               (row['CT报告'] if pd.notna(row['CT报告']) else ""),
#                     'gender': row['性别_数值'],
#                     'age': row['年龄_标准化'],
#                     'nodule_size': row['结节大小_标准化'],
#                     'multifocal': row['是否多发_数值'],
#                 }
#
#         found_ct_numbers = set()
#         found_by_subtype = defaultdict(int)
#
#         for root_dir in root_dirs:
#             for subtype in range(5):
#                 subtype_dir = os.path.join(root_dir, str(subtype))
#                 if not os.path.exists(subtype_dir):
#                     continue
#                 for file in os.listdir(subtype_dir):
#                     if file.endswith('.nii.gz'):
#                         file_path = os.path.join(subtype_dir, file)
#                         match = re.search(r'(UCT\d+)', file)
#                         ct_number = match.group(1) if match else file.split('.')[0]
#                         if ct_number in ct_info_map:
#                             info = ct_info_map[ct_number]
#                             self.samples.append({
#                                 'image_path': file_path,
#                                 'report': info['report'],
#                                 # ------------------ 人口学特征：四项 -------------------
#                                 'demographic': [
#                                     info['age'],
#                                     info['gender'],
#                                     info['nodule_size'],
#                                     info['multifocal'],
#                                 ],
#                                 'recurrence_label': info['label'],
#                                 'subtype_label': subtype,
#                                 'ct_number': ct_number,
#                             })
#                             found_ct_numbers.add(ct_number)
#                             found_by_subtype[subtype] += 1
#
#         # 没有CT图像的情况，保留文本/人口学特征，其image设为None
#         missing_ct_numbers = ct_with_recurrence_label - found_ct_numbers
#         for ct_number in missing_ct_numbers:
#             info = ct_info_map[ct_number]
#             self.samples.append({
#                 'image_path': None,
#                 'report': info['report'],
#                 'demographic': [
#                     info['age'],
#                     info['gender'],
#                     info['nodule_size'],
#                     info['multifocal'],
#                 ],
#                 'recurrence_label': info['label'],
#                 'subtype_label': -1,  # 无法判断亚型
#                 'ct_number': ct_number,
#             })
#
#         print(f"\n===== 数据集统计信息 =====")
#         print(f"CSV中有复发标签的CT号总数: {len(ct_with_recurrence_label)}")
#         print(f"找到CT图像的样本数: {len(found_ct_numbers)}")
#         print(f"有复发标签但找不到CT图像的CT号数: {len(missing_ct_numbers)}")
#         if missing_ct_numbers:
#             print("\n以下CT号有复发标签但找不到对应的CT图像:")
#             for ct_number in sorted(missing_ct_numbers):
#                 print(f"  - {ct_number}")
#         recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 1)
#         non_recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 0)
#         print(f"\n有效样本总数: {len(self.samples)}")
#         print(f"复发样本数: {recurrence_count}")
#         print(f"非复发样本数: {non_recurrence_count}")
#         print("\n各亚型样本分布:")
#         for i in range(5):
#             count = found_by_subtype[i]
#             print(f"  亚型 {i} 样本数: {count}")
#         print("============================\n")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         report_text = sample['report']
#         if self.text_tokenizer and report_text:
#             text_encoding = self.text_tokenizer(
#                 report_text,
#                 max_length=self.max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             text_inputs = {k: v.squeeze(0) for k, v in text_encoding.items()}
#         else:
#             if self.text_tokenizer:
#                 text_encoding = self.text_tokenizer(
#                     "",
#                     max_length=self.max_length,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt'
#                 )
#                 text_inputs = {k: v.squeeze(0) for k, v in text_encoding.items()}
#             else:
#                 text_inputs = {}
#
#         demographic = torch.tensor(sample['demographic'], dtype=torch.float32)
#
#         data_dict = {
#             'report_text': report_text,
#             "image": sample['image_path'],
#             "text": text_inputs,
#             "demographic": demographic,
#             "recurrence_label": sample['recurrence_label'],
#             "subtype_label": sample['subtype_label'],
#             "ct_number": sample['ct_number']
#         }
#
#         # 图像存在时才做transform，否则image为None
#         if self.transform and data_dict["image"] is not None:
#             image_dict = {"image": data_dict["image"], "label": data_dict["recurrence_label"]}
#             transformed = self.transform(image_dict)
#             data_dict["image"] = transformed["image"]
#
#         return data_dict

import os
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict


class MultimodalRecurrenceDataset(Dataset):
    """多模态肺癌复发预测与亚型分类多任务数据集，支持预提取的关键词"""

    def __init__(self, csv_path, root_dirs, transform=None, text_tokenizer=None, max_length=512,
                 keyword_fields=None):
        """
        初始化数据集
        参数:
            csv_path: CSV文件路径
            root_dirs: CT图像根目录
            transform: 图像变换
            text_tokenizer: 文本分词器
            max_length: 文本最大长度
            keyword_fields: 要作为关键词提取的CSV列名列表，如果为None，则使用默认字段
        """
        self.transform = transform
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.samples = []
        self._data_counter = None

        # 默认的关键词字段列表
        if keyword_fields is None:
            # self.keyword_fields = [
            #     '性别','年龄', '结节位置', '结节大小', '密度', '结节形态', '分叶征', '毛刺征', '胸膜牵拉',
            #     '胸膜凹陷', '气管支气管征', '血管穿行征', '胸膜侵犯', '脉管侵犯', '神经侵犯',
            #     '气道播散', '结节个数', '是否多发', '肺门淋巴结肿大', '纵隔淋巴结肿大',
            #     '胸腔积液', '纵隔/肺门其他病变', '浸润性腺癌', '生长方式', '胸膜侵犯分级',
            #     '生长模式其他', '切缘状态', '淋巴结转移', 'TTF-1', 'C-MET', 'Ki-67'
            # ]
            self.keyword_fields = ['性别', '年龄', '结节大小', '密度', '结节形态', '分叶征', '毛刺征', '胸膜牵拉',
                                   '胸膜凹陷', '血管穿行征', '结节个数', '是否多发', '肺门淋巴结肿大',
                                   '纵隔淋巴结肿大', '淋巴结转移', 'TTF-1', 'C-MET', 'Ki-67', '浸润性腺癌'
                                  ]
        else:
            self.keyword_fields = keyword_fields

        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8')

        # 确保root_dirs是列表
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        # 性别映射
        df['性别_数值'] = df['性别'].map({'男': 1, '女': 0}).fillna(0)
        df['年龄_数值'] = df['年龄'].astype(str).str.replace('岁', '', regex=False).astype(float)

        # 结节大小（最大径）归一化
        df['结节大小'] = pd.to_numeric(df.get('结节大小', None), errors='coerce')
        nodule_min, nodule_max = df['结节大小'].min(), df['结节大小'].max()
        df['结节大小_标准化'] = (df['结节大小'] - nodule_min) / (nodule_max - nodule_min)
        df['结节大小_标准化'] = df['结节大小_标准化'].fillna(0)

        # 是否多发映射
        df['是否多发_数值'] = df['是否多发'].map({'是': 1, '否': 0}).fillna(0)

        # 年龄归一化
        age_min, age_max = df['年龄_数值'].min(), df['年龄_数值'].max()
        df['年龄_标准化'] = (df['年龄_数值'] - age_min) / (age_max - age_min)

        # 复发标签
        df['复发标签'] = df['是否复发'].map({'有': 1, '无': 0})

        # 构建CT号到样本信息的映射
        ct_info_map = {}
        ct_with_recurrence_label = set()
        for idx, row in df.iterrows():
            ct_number = row['CT号']
            if pd.notna(ct_number) and ct_number.strip():
                recurrence_label = row['复发标签']
                ct_with_recurrence_label.add(ct_number)

                keywords = []
                for field in self.keyword_fields:
                    if field in row:  # 只检查字段是否存在
                        if pd.isna(row[field]) or row[field] is None:
                            # 如果是None或NaN值，标记为"未提及"
                            keywords.append((field, "未提及"))
                        else:
                            # 如果有值，转为字符串并添加
                            keywords.append((field, str(row[field])))
                # 结构化特征
                ct_info_map[ct_number] = {
                    'label': recurrence_label,
                    'report1': row['CT报告'],
                    'report2': row['病理报告'],
                    'gender': row['性别_数值'],
                    'age': row['年龄_标准化'],
                    'nodule_size': row['结节大小_标准化'],
                    'multifocal': row['是否多发_数值'],
                    'keywords': keywords  # 保存关键词-值对
                }

        found_ct_numbers = set()
        found_by_subtype = defaultdict(int)

        for root_dir in root_dirs:
            for subtype in range(5):
                subtype_dir = os.path.join(root_dir, str(subtype))
                if not os.path.exists(subtype_dir):
                    continue
                for file in os.listdir(subtype_dir):
                    if file.endswith('.nii.gz'):
                        file_path = os.path.join(subtype_dir, file)
                        match = re.search(r'(UCT\d+|WWCT\d+)', file)
                        ct_number = match.group(1) if match else file.split('.')[0]
                        if ct_number in ct_info_map:
                            info = ct_info_map[ct_number]
                            self.samples.append({
                                'image_path': file_path,
                                'report1': info['report1'],
                                'report2': info['report2'],
                                'demographic': [
                                    info['age'],
                                    info['gender'],
                                    info['nodule_size'],
                                    info['multifocal'],
                                ],
                                'recurrence_label': info['label'],
                                'subtype_label': subtype,
                                'ct_number': ct_number,
                                'keywords': info['keywords']  # 添加关键词
                            })
                            found_ct_numbers.add(ct_number)
                            found_by_subtype[subtype] += 1

        # 没有CT图像的情况，保留文本/人口学特征，其image设为None
        missing_ct_numbers = ct_with_recurrence_label - found_ct_numbers
        # for ct_number in missing_ct_numbers:
        #     info = ct_info_map[ct_number]
        #     self.samples.append({
        #         'image_path': None,
        #         'report': info['report'],
        #         'demographic': [
        #             info['age'],
        #             info['gender'],
        #             info['nodule_size'],
        #             info['multifocal'],
        #         ],
        #         'recurrence_label': info['label'],
        #         'subtype_label': -1,  # 无法判断亚型
        #         'ct_number': ct_number,
        #         'keywords': info['keywords']  # 添加关键词
        #     })

        print(f"\n===== 数据集统计信息 =====")
        print(f"CSV中有复发标签的CT号总数: {len(ct_with_recurrence_label)}")
        print(f"找到CT图像的样本数: {len(found_ct_numbers)}")
        print(f"有复发标签但找不到CT图像的CT号数: {len(missing_ct_numbers)}")
        recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 1)
        non_recurrence_count = sum(1 for sample in self.samples if sample['recurrence_label'] == 0)
        print(f"\n有效样本总数: {len(self.samples)}")
        print(f"复发样本数: {recurrence_count}")
        print(f"非复发样本数: {non_recurrence_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # if isinstance(idx, list):
        #     return [self.__getitem__(i) for i in idx]
        sample = self.samples[idx]
        report1 = sample['report1']
        text_encoding1 = self.text_tokenizer(
            report1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_inputs1 = {k: v.squeeze(0) for k, v in text_encoding1.items()}

        report2 = sample['report2']
        text_encoding2 = self.text_tokenizer(
            report2,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        text_inputs2 = {k: v.squeeze(0) for k, v in text_encoding2.items()}


        demographic = torch.tensor(sample['demographic'], dtype=torch.float32)

        data_dict = {
            'report1': report1,
            'report2': report2,
            "image": sample['image_path'],
            "text_inputs1": text_inputs1,
            "text_inputs2": text_inputs2,
            "demographic": demographic,
            "recurrence_label": sample['recurrence_label'],
            "subtype_label": sample['subtype_label'],
            "ct_number": sample['ct_number'],
            "keywords": sample['keywords']  # 添加关键词-值对
        }

        # 图像存在时才做transform，否则image为None
        if self.transform and data_dict["image"] is not None:
            image_dict = {"image": data_dict["image"], "label": data_dict["recurrence_label"]}
            transformed = self.transform(image_dict)
            data_dict["image"] = transformed["image"]

        return data_dict