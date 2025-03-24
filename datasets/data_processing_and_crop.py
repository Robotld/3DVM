
"""
    根据yolo 标签。 选取中心点， 裁剪出 96*96*96 大小的结节块
"""



import os
import re
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import statistics

# Configuration parameters
NIFTI_ROOT = Path(r"G:\datasets3D\test")
LABEL_ROOT = Path(r"G:\data2\labels")
OUTPUT_ROOT = Path(r"G:\datasets3D\test_crop")
PATCH_SIZE_MM = 96  # Physical size in mm (before resampling)
TARGET_SPACING = (0.5, 0.5, 2)  # Target isotropic resolution
TARGET_SHAPE = (96, 96, 96)  # Final shape of each patch after resampling
CLASS_NAMES = {
    "0": "低分化",
    "1": "中分化",
    "2": "高分化",
    "3": "原位癌",
    "4": "微浸润"
}


def parse_yolo_annotation(file_path):
    """
    Parse YOLO format annotation file
    Returns: List of [class_id, x_center, y_center, width, height]
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # Convert all to float except class_id (int)
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append([class_id, x_center, y_center, width, height])
    return annotations


def find_nodule_slices(ct_id):
    """
    Find all annotation files for a specific CT scan
    Returns: Dictionary mapping slice numbers to file paths
    """
    slice_files = {}
    pattern = f"{ct_id}_*.txt"
    for file_path in glob.glob(os.path.join(LABEL_ROOT, pattern)):
        # Extract slice number from filename
        match = re.search(r'_(\d+)\.txt$', file_path)
        if match:
            slice_num = int(match.group(1))
            slice_files[slice_num] = file_path
    return slice_files


def find_matching_nifti(ct_id):
    """
    Find matching NIfTI file for a CT ID across all class folders
    """
    for class_folder in os.listdir(NIFTI_ROOT):
        class_path = NIFTI_ROOT / class_folder
        if not class_path.is_dir():
            continue

        # Pattern matching for CT ID
        for nifti_file in class_path.glob(f"*{ct_id}*.nii.gz"):
            return nifti_file, class_folder

    return None, None


def find_longest_contiguous_segment(slice_numbers):
    """找出最长的连续切片段"""
    if not slice_numbers:
        return []

    # 将切片号排序
    sorted_slices = sorted(slice_numbers)

    # 分割连续段
    segments = []
    current_segment = [sorted_slices[0]]

    for num in sorted_slices[1:]:
        if num == current_segment[-1] + 1:
            current_segment.append(num)
        else:
            segments.append(current_segment)
            current_segment = [num]
    segments.append(current_segment)

    # 找出最长段（长度相同时选第一个）
    longest_segment = max(segments, key=lambda x: len(x))
    return longest_segment
def get_central_nodule_slice(slice_files):
    """
    Determine the central slice of a nodule based on available annotations
    """
    if not slice_files:
        return None, None

    # Find the middle slice number
    slice_numbers = sorted(slice_files.keys())
    longest_segment = find_longest_contiguous_segment(slice_numbers)

    central_idx = len(longest_segment) // 2
    central_slice_num = longest_segment[central_idx]

    # Get the annotation file for this slice
    central_file = slice_files[central_slice_num]
    # print(central_file)
    # Parse the annotation to get nodule position
    annotations = parse_yolo_annotation(central_file)
    if not annotations:
        return None, None

    # Use the first annotation (assuming one nodule per file)
    return central_slice_num, annotations[0]


def resample_volume(image, new_spacing, interpolator=sitk.sitkBSpline):
    """
    增强版重采样函数，提供更高质量的图像重采样

    参数:
        image: SimpleITK图像对象
        new_spacing: 目标体素间距，如(0.5, 0.5, 0.5)
        interpolator: 插值器类型，默认使用B样条插值

    返回:
        重采样后的SimpleITK图像
    """
    # 获取原始参数
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()

    # 计算新尺寸
    new_size = [
        int(round(osz * ospacing / nspacing))
        for osz, ospacing, nspacing in zip(original_size, original_spacing, new_spacing)
    ]

    # 判断是放大还是缩小
    is_downsampling = any(ns > os for ns, os in zip(new_spacing, original_spacing))

    # 预处理：如果是降采样，先进行高斯平滑以防止锯齿
    if is_downsampling:
        # 计算平滑参数 - 按照Nyquist采样定理
        sigma = [(ns / os) / 2.0 for ns, os in zip(new_spacing, original_spacing)]
        image = sitk.DiscreteGaussian(image, sigma)

    # 设置重采样过滤器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)  # 对于CT图像，通常使用空气的HU值
    resampler.SetInterpolator(interpolator)

    # 执行重采样
    resampled_image = resampler.Execute(image)

    return resampled_image

# def extract_nodule_patch(nifti_path, central_slice, nodule_info):
#     """
#     Extract a 3D patch centered on the nodule
#
#     Args:
#         nifti_path: Path to the NIfTI file
#         central_slice: Central slice number (1-indexed)
#         nodule_info: YOLO annotation [class_id, x_center, y_center, width, height]
#
#     Returns:
#         3D patch of the nodule
#     """
#     # Load the NIfTI image
#     image = sitk.ReadImage(str(nifti_path))
#     original_spacing = image.GetSpacing()
#     original_size = image.GetSize()
#
#     # First resample to isotropic resolution
#     resampled_image = resample_volume(image, TARGET_SPACING)
#     resampled_size = resampled_image.GetSize()
#
#     # Calculate the new slice index after resampling
#     # Slice numbers start from 1, convert to 0-indexed for calculations
#     slice_idx_0based = central_slice - 1
#
#     # Calculate the ratio of new slice count to old slice count
#     slice_ratio = resampled_size[2] / original_size[2]
#
#     # Calculate the new slice index in the resampled volume
#     resampled_slice_idx = int(round(slice_idx_0based * slice_ratio))
#
#     # Get the nodule center positions (normalized 0-1 in YOLO format)
#     _, nodule_x, nodule_y, _, _ = nodule_info
#
#     # Convert normalized coordinates to actual pixel positions in resampled space
#     center_x = int(round(nodule_x * resampled_size[0]))
#     center_y = int(round(nodule_y * resampled_size[1]))
#     center_z = resampled_slice_idx
#
#     # Calculate the patch half-size in voxels (at 1mm resolution, the size in mm equals the size in voxels)
#     half_size = PATCH_SIZE_MM // 2
#
#     # Calculate extraction boundaries
#     x_min = max(0, center_x - half_size)
#     y_min = max(0, center_y - half_size)
#     z_min = max(0, center_z - half_size)
#
#     x_max = min(resampled_size[0], center_x + half_size)
#     y_max = min(resampled_size[1], center_y + half_size)
#     z_max = min(resampled_size[2], center_z + half_size)
#
#     # Extract region
#     extractor = sitk.RegionOfInterestImageFilter()
#     extractor.SetSize([x_max - x_min, y_max - y_min, z_max - z_min])
#     extractor.SetIndex([x_min, y_min, z_min])
#
#     nodule_patch = extractor.Execute(resampled_image)
#
#     # Pad if necessary to ensure TARGET_SHAPE dimensions
#     padded_patch = sitk.ConstantPad(
#         nodule_patch,
#         [max(0, (TARGET_SHAPE[0] - (x_max - x_min)) // 2),
#          max(0, (TARGET_SHAPE[1] - (y_max - y_min)) // 2),
#          max(0, (TARGET_SHAPE[2] - (z_max - z_min)) // 2)],
#         [max(0, TARGET_SHAPE[0] - (x_max - x_min) - (TARGET_SHAPE[0] - (x_max - x_min)) // 2),
#          max(0, TARGET_SHAPE[1] - (y_max - y_min) - (TARGET_SHAPE[1] - (y_max - y_min)) // 2),
#          max(0, TARGET_SHAPE[2] - (z_max - z_min) - (TARGET_SHAPE[2] - (z_max - z_min)) // 2)]
#     )
#
#     # If patch is too large (shouldn't happen with proper padding), crop to target size
#     if any(padded_patch.GetSize()[i] > TARGET_SHAPE[i] for i in range(3)):
#         final_extractor = sitk.RegionOfInterestImageFilter()
#         final_extractor.SetSize(TARGET_SHAPE)
#         final_extractor.SetIndex([0, 0, 0])
#         final_patch = final_extractor.Execute(padded_patch)
#     else:
#         final_patch = padded_patch
#
#     return final_patch

def extract_nodule_patch(nifti_path, central_slice, nodule_info):
    """
    Extract a 3D patch centered on the nodule using correct physical-to-voxel conversion.

    Args:
        nifti_path: Path to the NIfTI file.
        central_slice: Central slice number (1-indexed).
        nodule_info: YOLO annotation [class_id, x_center, y_center, width, height].

    Returns:
        3D patch of the nodule.
    """
    # Load the NIfTI image (original image)
    image = sitk.ReadImage(str(nifti_path))
    original_spacing = image.GetSpacing()  # (e.g., (0.5, 0.5, 1.0))
    original_size = image.GetSize()

    # First, resample to target isotropic resolution
    resampled_image = resample_volume(image, TARGET_SPACING)
    resampled_size = resampled_image.GetSize()

    # Calculate nodule center in original physical space:
    # Convert normalized YOLO coordinates and slice index to physical coordinates
    _, nodule_x, nodule_y, _, _ = nodule_info
    slice_idx_0based = central_slice - 1
    physical_x = nodule_x * original_size[0] * original_spacing[0]
    physical_y = nodule_y * original_size[1] * original_spacing[1]
    physical_z = slice_idx_0based * original_spacing[2]

    # Convert physical coordinates to voxel coordinates in resampled image
    center_x = int(round(physical_x / TARGET_SPACING[0]))
    center_y = int(round(physical_y / TARGET_SPACING[1]))
    center_z = int(round(physical_z / TARGET_SPACING[2]))

    # Calculate patch half-size in voxels for each dimension,
    # ensuring the extracted patch covers PATCH_SIZE_MM in physical units.
    half_size_x = int(round((PATCH_SIZE_MM / 2) / TARGET_SPACING[0]))
    half_size_y = int(round((PATCH_SIZE_MM / 2) / TARGET_SPACING[1]))
    half_size_z = int(round((PATCH_SIZE_MM / 2) / TARGET_SPACING[2]))

    # Calculate extraction boundaries
    x_min = max(0, center_x - half_size_x)
    y_min = max(0, center_y - half_size_y)
    z_min = max(0, center_z - half_size_z)

    x_max = min(resampled_size[0], center_x + half_size_x)
    y_max = min(resampled_size[1], center_y + half_size_y)
    z_max = min(resampled_size[2], center_z + half_size_z)

    # Extract region from resampled image
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize([x_max - x_min, y_max - y_min, z_max - z_min])
    extractor.SetIndex([x_min, y_min, z_min])
    nodule_patch = extractor.Execute(resampled_image)

    # Pad if necessary to match TARGET_SHAPE dimensions
    padded_patch = sitk.ConstantPad(
        nodule_patch,
        [max(0, (TARGET_SHAPE[0] - (x_max - x_min)) // 2),
         max(0, (TARGET_SHAPE[1] - (y_max - y_min)) // 2),
         max(0, (TARGET_SHAPE[2] - (z_max - z_min)) // 2)],
        [max(0, TARGET_SHAPE[0] - (x_max - x_min) - (TARGET_SHAPE[0] - (x_max - x_min)) // 2),
         max(0, TARGET_SHAPE[1] - (y_max - y_min) - (TARGET_SHAPE[1] - (y_max - y_min)) // 2),
         max(0, TARGET_SHAPE[2] - (z_max - z_min) - (TARGET_SHAPE[2] - (z_max - z_min)) // 2)]
    )

    # Crop if patch exceeds target shape (as safeguard)
    if any(padded_patch.GetSize()[i] > TARGET_SHAPE[i] for i in range(3)):
        final_extractor = sitk.RegionOfInterestImageFilter()
        final_extractor.SetSize(TARGET_SHAPE)
        final_extractor.SetIndex([0, 0, 0])
        final_patch = final_extractor.Execute(padded_patch)
    else:
        final_patch = padded_patch

    return final_patch

def verify_size(patch):
    """
    Verify and fix patch size to match TARGET_SHAPE
    """
    current_size = patch.GetSize()

    # If sizes match exactly, return the patch
    if current_size == TARGET_SHAPE:
        return patch

    # Otherwise, do a resampling to exactly match the target shape
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(patch.GetSpacing())
    resampler.SetSize(TARGET_SHAPE)
    resampler.SetOutputDirection(patch.GetDirection())
    resampler.SetOutputOrigin(patch.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    # Execute the resampling
    fixed_patch = resampler.Execute(patch)
    return fixed_patch


def extract_ct_nodules():
    """
    Main function to process all CTs and extract nodule patches
    """
    # Create output directories
    for class_id in CLASS_NAMES:
        (OUTPUT_ROOT / class_id).mkdir(parents=True, exist_ok=True)

    # Get all unique CT IDs from annotation files
    # ct_pattern = re.compile(r'([^_]+)_[^_]+\.txt')
    ct_ids = set()

    for file_path in os.listdir(LABEL_ROOT):
        ct_id = file_path.split("_")[0]
        ct_ids.add(ct_id)

    print(f"Found {len(ct_ids)} unique CT scans with annotations")

    # return
    # Process each CT scan
    success_count = 0
    for ct_id in tqdm(sorted(ct_ids)):
        # Find all slice annotations for this CT
        slice_files = find_nodule_slices(ct_id)
        # print(slice_files)
        if not slice_files:
            print(f"No annotations found for CT: {ct_id}")
            continue

        # Find the corresponding NIfTI file
        nifti_path, class_id = find_matching_nifti(ct_id)
        if not nifti_path:
            print(f"No matching NIfTI file found for CT: {ct_id}")
            continue

        # Get the central nodule slice and annotation
        central_slice, nodule_info = get_central_nodule_slice(slice_files)
        # if ct_id == "UCT202301140579":
        #     print(slice_files, central_slice, nodule_info)
        if not central_slice or not nodule_info:
            print(f"No valid central nodule found for CT: {ct_id}")
            continue

        try:
            # Extract the nodule patch
            nodule_patch = extract_nodule_patch(nifti_path, central_slice, nodule_info)

            # Verify final size
            final_patch = verify_size(nodule_patch)

            # Save the patch
            output_path = OUTPUT_ROOT / class_id / f"{ct_id}.nii.gz"
            sitk.WriteImage(final_patch, str(output_path))
            success_count += 1

        except Exception as e:
            print(f"Error processing CT {ct_id}: {str(e)}")

    print(f"Processing complete! Successfully extracted {success_count} nodule patches")


if __name__ == "__main__":
    extract_ct_nodules()