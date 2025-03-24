import os
import re

import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================== 配置参数 ==================
DICOM_ROOT = Path(r"G:\2021年及之前协和CT")
OUTPUT_ROOT = Path(r"G:\datasets3D\test")
TARGET_SPACING = (0.5, 0.5, 0.5)  # 各向同性分辨率
CLASS_MAP = {
    "原位癌": "3",
    "微浸润": "4",
    "低分化": "0",
    "中分化": "1",
    "高分化": "2"
}


# =============================================

def map_path_to_category(dicom_path):
    """智能病理分类映射"""
    path_str = str(dicom_path)
    # print(path_str)
    # 优先匹配精确路径关键词
    for category, keywords in CLASS_MAP.items():
        # print(category, keywords)
        if category in path_str:
            return keywords

def load_dicom_series(dicom_dir):
    """加载无后缀DICOM序列"""
    # reader = sitk.ImageSeriesReader()

    # 获取有效DICOM文件并缓存元数据
    dicom_files = []
    metadata = []
    for f in dicom_dir.glob("*"):
        if f.is_file() and is_valid_dicom(f):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                dicom_files.append(str(f))
                metadata.append((str(f), ds.InstanceNumber))
            except:
                print(f"Skipping file {f}: missing InstanceNumber.")

    # 按InstanceNumber排序
    if metadata:
        dicom_files = [f for f, _ in sorted(metadata, key=lambda x: x[1])]
        # print("DICOM files sorted by InstanceNumber.")
    else:
        print("No valid DICOM files found.")

    return dicom_files


def safe_get_metadata(image, key, default=None):
    """安全获取DICOM元数据"""
    try:
        if image.HasMetaDataKey(key):
            return image.GetMetaData(key)
        return default
    except RuntimeError:
        return default


def apply_hu_conversion(image):
    """
    应用肺窗预设

    参数:
        image: SimpleITK图像对象

    返回:
        应用肺窗后的图像
    """
    # 肺窗设置：窗宽1500，窗位-600
    lung_window_min = -1350  # -600 - 1500/2
    lung_window_max = 150  # -600 + 1500/2

    # 将图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)

    # 应用肺窗
    image_array = np.clip(image_array, lung_window_min, lung_window_max)

    # 归一化为0-1范围
    image_array = (image_array - lung_window_min) / (lung_window_max - lung_window_min)

    # 转回SimpleITK图像
    windowed_image = sitk.GetImageFromArray(image_array)
    windowed_image.SetSpacing(image.GetSpacing())
    windowed_image.SetOrigin(image.GetOrigin())
    windowed_image.SetDirection(image.GetDirection())

    return windowed_image

# def resample_isotropic(image):
#     """优化后的各向同性重采样"""
#     original_spacing = image.GetSpacing()
#     new_size = [int(s * o / t) for s, o, t in zip(image.GetSize(), original_spacing, TARGET_SPACING)]
#
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetOutputSpacing(TARGET_SPACING)
#     resampler.SetSize(new_size)
#     resampler.SetInterpolator(sitk.sitkBSpline)  # B样条插值保留细节
#     return resampler.Execute(image)


def process_patient(dicom_dir):
    """处理单个患者数据"""
    try:
        # 1. 加载DICOM序列
        dicom_files = load_dicom_series(dicom_dir)
        if len(dicom_files) < 10:
            return

        # 2. 读取DICOM数据
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # 3. 预处理流程
        processed = image
        # 3.1 统一坐标系为RAS+
        # processed = sitk.DICOMOrient(processed, "RAS")
        # 3.2 重采样
        # processed = resample_isotropic(processed)
        # 3.3 强度标准化
        processed = apply_hu_conversion(processed)
        # print(dicom_dir)
        # 4. 分类存储
        category = map_path_to_category(dicom_dir)
        output_dir = OUTPUT_ROOT / category
        output_dir.mkdir(parents=True, exist_ok=True)
        # print(category)
        # 生成唯一ID: CT号 + 序列号
        ct_number = ""
        path_parts = str(dicom_dir).split(os.sep)
        for part in path_parts:
            match = re.search(r'(CT\d+)|(UPT\d+)|(MR\d+)', part)
            if match:
                ct_number = part
                break
        # print("ct:", ct_number)
        output_path = output_dir / f"{ct_number}.nii.gz"

        # 5. 保存NIfTI
        sitk.WriteImage(processed, str(output_path))

        return True
    except Exception as e:
        print(f"处理失败 {dicom_dir}: {str(e)}")
        return False

def is_leaf_directory(dir_path):
    """判断是否为叶子目录（没有子目录）"""
    try:
        # 获取目录下所有条目
        entries = list(os.scandir(dir_path))
        # 检查是否存在子目录
        return not any(entry.is_dir() for entry in entries)
    except PermissionError:
        return False


def is_valid_dicom(file_path):
    """验证无后缀文件是否为有效DICOM"""
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
        return True
    except:
        return False


def find_dicom_dirs(root_dir, min_files=10):
    """查找包含足够DICOM文件的叶子目录"""
    valid_dirs = []

    # 使用os.walk进行高效遍历
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc="扫描目录"):
        current_dir = Path(dirpath)

        # 提前剪枝：如果当前目录有子目录，跳过后续检查
        if dirnames:
            continue

        # 检查是否叶子目录
        if not is_leaf_directory(current_dir):
            continue
        valid_dirs.append(current_dir)
        # # 统计有效DICOM文件
        # valid_count = 0
        # for fname in filenames:
        #     file_path = current_dir / fname
        #     if file_path.is_file() and is_valid_dicom(file_path):
        #         valid_count += 1
        #         # 提前终止检查
        #         if valid_count >= min_files:
        #             valid_dirs.append(current_dir)
        #             break
        print(current_dir)
    return valid_dirs


# ================== 主执行流程 ==================
if __name__ == "__main__":

    # 创建输出目录结构
    for category in CLASS_MAP.values():
        (OUTPUT_ROOT / category).mkdir(exist_ok=True)

    # 查找所有患者目录
    print("正在扫描DICOM目录...")
    patient_dirs = find_dicom_dirs(DICOM_ROOT)
    print(f"找到{len(patient_dirs)}个有效病例")

    results = []
    for dir_path in tqdm(patient_dirs):
        results.append(process_patient(dir_path))

    # 打印统计信息
    success_count = sum(results)
    print(f"处理完成! 成功: {success_count}, 失败: {len(results) - success_count}")