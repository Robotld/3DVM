import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pydicom
import SimpleITK as sitk
import numpy as np
from pathlib import Path
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
# 添加默认类别
DEFAULT_CATEGORY = "-1"  # 如果无法确定分类，默认归为"低分化"类别
# 添加并行处理参数
MAX_WORKERS = 8  # 根据CPU核心数调整
# 添加最小DICOM文件数量检查
MIN_DICOM_FILES = 10
# 日志设置
LOG_FILE = OUTPUT_ROOT / "processing.log"
# =============================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def map_path_to_category(dicom_path):
    """智能病理分类映射"""
    path_str = str(dicom_path)

    # 优先匹配精确路径关键词
    for category, label in CLASS_MAP.items():
        if category in path_str:
            return label

    # 如果没有匹配到任何类别，返回默认类别
    return DEFAULT_CATEGORY


def load_dicom_series(dicom_dir):
    """加载无后缀DICOM序列"""
    # 获取有效DICOM文件并缓存元数据
    dicom_files = []
    metadata = []

    try:
        for f in dicom_dir.glob("*"):
            if f.is_file() and is_valid_dicom(f):
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    if hasattr(ds, 'InstanceNumber'):
                        dicom_files.append(str(f))
                        metadata.append((str(f), ds.InstanceNumber))
                    else:
                        logger.warning(f"跳过文件 {f}: 缺少InstanceNumber")
                except Exception as e:
                    logger.warning(f"读取文件 {f} 失败: {str(e)}")

        # 按InstanceNumber排序
        if metadata:
            sorted_metadata = sorted(metadata, key=lambda x: x[1])
            dicom_files = [f for f, _ in sorted_metadata]
            logger.debug(f"DICOM文件已按InstanceNumber排序，共{len(dicom_files)}个文件")
        else:
            logger.warning(f"目录 {dicom_dir} 中没有找到有效DICOM文件")

    except Exception as e:
        logger.error(f"加载DICOM序列 {dicom_dir} 时出错: {str(e)}")

    return dicom_files


def safe_get_metadata(image, key, default=None):
    """安全获取DICOM元数据"""
    try:
        if image.HasMetaDataKey(key):
            return image.GetMetaData(key)
        return default
    except RuntimeError:
        return default


def apply_hu_conversion(image, method='auto', percentile_low=2, percentile_high=98, preset=None):
    """
    应用窗宽窗位转换CT图像

    参数:
        image: SimpleITK图像对象
        method: 窗宽窗位计算方法
        percentile_low: 下限百分位数
        percentile_high: 上限百分位数
        preset: 预设值

    返回:
        处理后的图像
    """
    # 将图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)

    # 预设窗宽窗位字典
    presets = {
        'lung': {'width': 1500, 'level': -600},
        'mediastinum': {'width': 350, 'level': 50},
        'bone': {'width': 2000, 'level': 400},
        'brain': {'width': 80, 'level': 40},
        'liver': {'width': 150, 'level': 30}
    }

    # 确定窗宽窗位
    if method == 'auto':
        # 使用百分位数自动计算窗宽窗位
        p_low = np.percentile(image_array, percentile_low)
        p_high = np.percentile(image_array, percentile_high)
        window_width = p_high - p_low
        window_level = (p_high + p_low) / 2

    elif method == 'otsu':
        # 使用Otsu方法自动计算阈值
        # 首先归一化到0-255范围进行Otsu计算
        min_val = image_array.min()
        max_val = image_array.max()
        range_val = max_val - min_val

        if range_val == 0:  # 防止除零错误
            return image

        img_normalized = ((image_array - min_val) / range_val * 255).astype(np.uint8)

        # 转换为SimpleITK图像以使用OtsuThresholdImageFilter
        sitk_normalized = sitk.GetImageFromArray(img_normalized)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        _ = otsu_filter.Execute(sitk_normalized)  # 执行过滤器
        threshold = otsu_filter.GetThreshold()

        # 将阈值转回原始HU值范围
        threshold_hu = threshold / 255 * range_val + min_val

        # 使用Otsu阈值设置窗宽窗位
        window_width = range_val * 0.7
        window_level = threshold_hu

    elif method == 'preset':
        # 使用预设窗宽窗位
        if isinstance(preset, dict) and 'width' in preset and 'level' in preset:
            window_width = preset['width']
            window_level = preset['level']
        elif preset in presets:
            window_width = presets[preset]['width']
            window_level = presets[preset]['level']
        else:
            # 默认使用肺窗
            logger.info(f"预设 '{preset}' 不存在，使用默认肺窗")
            window_width = 1500
            window_level = -600
    else:
        # 默认使用肺窗
        window_width = 1500
        window_level = -600

    # 计算窗宽窗位的最小值和最大值
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2

    logger.info(f"应用窗宽窗位转换 - 窗宽: {window_width:.1f}, 窗位: {window_level:.1f}")
    logger.debug(f"HU值范围: {window_min:.1f} 到 {window_max:.1f}")

    # 应用窗宽窗位
    image_array = np.clip(image_array, window_min, window_max)

    # 归一化为0-1范围
    image_array = (image_array - window_min) / (window_max - window_min)

    # 转回SimpleITK图像
    windowed_image = sitk.GetImageFromArray(image_array)
    windowed_image.SetSpacing(image.GetSpacing())
    windowed_image.SetOrigin(image.GetOrigin())
    windowed_image.SetDirection(image.GetDirection())

    return windowed_image


def resample_isotropic(image):
    """各向同性重采样"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新尺寸
    new_size = [int(round(s * o / t)) for s, o, t in zip(original_size, original_spacing, TARGET_SPACING)]

    # 创建重采样滤波器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)

    # 使用B样条插值保留细节
    resampler.SetInterpolator(sitk.sitkBSpline)

    # 执行重采样
    try:
        resampled_image = resampler.Execute(image)
        logger.info(f"重采样成功: {original_spacing} -> {TARGET_SPACING}, {original_size} -> {new_size}")
        return resampled_image
    except Exception as e:
        logger.error(f"重采样失败: {str(e)}")
        return image


def process_patient(dicom_dir):
    """处理单个患者数据"""
    try:
        # 1. 加载DICOM序列
        dicom_files = load_dicom_series(dicom_dir)
        if not dicom_files or len(dicom_files) < MIN_DICOM_FILES:
            logger.warning(f"目录 {dicom_dir} 中DICOM文件数量不足: {len(dicom_files)}")
            return False

        # 2. 读取DICOM数据
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # 3. 预处理流程
        # 3.1 统一坐标系为RAS+
        try:
            processed = sitk.DICOMOrient(image, "RAS")
            logger.info("已统一坐标系为RAS+")
        except Exception as e:
            logger.warning(f"统一坐标系失败: {str(e)}")
            processed = image

        # 3.2 重采样
        processed = resample_isotropic(processed)

        # 3.3 强度标准化
        processed = apply_hu_conversion(processed, method='auto')

        # 4. 分类存储
        category = map_path_to_category(dicom_dir)
        if not category:
            logger.warning(f"无法确定 {dicom_dir} 的类别，使用默认类别 {DEFAULT_CATEGORY}")
            category = DEFAULT_CATEGORY

        output_dir = OUTPUT_ROOT / category
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一ID: CT号 + 序列号
        ct_number = ""
        path_parts = str(dicom_dir).split(os.sep)

        # 尝试找到CT号、UPT号或MR号
        for part in path_parts:
            match = re.search(r'(CT\d+)|(UPT\d+)|(MR\d+)', part)
            if match:
                ct_number = match.group(0)
                break

        # 如果没有找到CT号，使用目录名
        if not ct_number:
            ct_number = dicom_dir.name

        output_path = output_dir / f"{ct_number}.nii.gz"

        # 检查是否已存在相同文件
        if output_path.exists():
            logger.info(f"文件 {output_path} 已存在，跳过处理")
            return True

        # 5. 保存NIfTI
        sitk.WriteImage(processed, str(output_path))
        logger.info(f"成功处理并保存: {output_path}")

        return True
    except Exception as e:
        logger.error(f"处理失败 {dicom_dir}: {str(e)}")
        return False


def is_leaf_directory(dir_path):
    """判断是否为叶子目录（没有子目录）"""
    try:
        # 获取目录下所有条目
        entries = list(os.scandir(dir_path))
        # 检查是否存在子目录
        return not any(entry.is_dir() for entry in entries)
    except PermissionError:
        logger.warning(f"无权限访问目录: {dir_path}")
        return False
    except Exception as e:
        logger.warning(f"检查目录 {dir_path} 时出错: {str(e)}")
        return False


def is_valid_dicom(file_path):
    """验证无后缀文件是否为有效DICOM"""
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def find_dicom_dirs(root_dir, min_files=MIN_DICOM_FILES):
    """查找包含足够DICOM文件的叶子目录"""
    valid_dirs = []

    try:
        # 使用os.walk进行高效遍历
        for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc="扫描目录"):
            current_dir = Path(dirpath)

            # 提前剪枝：如果当前目录有子目录，跳过后续检查
            if dirnames:
                continue

            # 检查是否为叶子目录
            if not is_leaf_directory(current_dir):
                continue

            # 快速检查目录中的文件数量
            if len(filenames) < min_files:
                continue

            # 统计有效DICOM文件
            valid_count = 0
            for i, fname in enumerate(filenames):
                # 只检查前20个文件以提高速度
                if i >= 20:
                    break

                file_path = current_dir / fname
                if file_path.is_file() and is_valid_dicom(file_path):
                    valid_count += 1

                # 如果找到足够的DICOM文件，添加此目录
                if valid_count >= min_files:
                    valid_dirs.append(current_dir)
                    logger.info(f"找到有效DICOM目录: {current_dir}")
                    break

    except Exception as e:
        logger.error(f"扫描目录时出错: {str(e)}")

    return valid_dirs


def process_batch(dirs_batch):
    """处理一批目录"""
    results = []
    for dir_path in dirs_batch:
        result = process_patient(dir_path)
        results.append(result)
    return results


# ================== 主执行流程 ==================
if __name__ == "__main__":
    # 确保输出目录存在
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 创建输出目录结构
    for category in CLASS_MAP.values():
        (OUTPUT_ROOT / category).mkdir(exist_ok=True)

    # 查找所有患者目录
    logger.info("正在扫描DICOM目录...")
    patient_dirs = find_dicom_dirs(DICOM_ROOT, MIN_DICOM_FILES)
    logger.info(f"找到{len(patient_dirs)}个有效病例")

    # 使用线程池并行处理
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有处理任务
        future_to_dir = {executor.submit(process_patient, dir_path): dir_path for dir_path in patient_dirs}

        # 使用tqdm跟踪进度
        for future in tqdm(as_completed(future_to_dir), total=len(patient_dirs), desc="处理病例"):
            dir_path = future_to_dir[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logger.error(f"处理 {dir_path} 时发生异常: {str(e)}")
                all_results.append(False)

    # 打印统计信息
    success_count = sum(1 for r in all_results if r)
    logger.info(f"处理完成! 成功: {success_count}, 失败: {len(all_results) - success_count}")

    # 输出每个类别的文件数量
    for category_name, category_id in CLASS_MAP.items():
        category_dir = OUTPUT_ROOT / category_id
        file_count = len(list(category_dir.glob("*.nii.gz")))
        logger.info(f"类别 '{category_name}' (ID: {category_id}): {file_count} 个文件")
