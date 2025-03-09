import os
import random
import shutil
from tqdm import tqdm

# 配置参数
source_dir = r"G:\datasets\nodule_patches96_4"
output_dir = r"G:\datasets\nodule_patches96_4_split"
class_folders = ['0', '1', '2', '3', '4']
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}  # 总比例必须为1.0
seed = 42  # 随机种子保证可重复性


def create_directory_structure():
    """创建输出目录结构"""
    for split in split_ratios.keys():
        for cls in class_folders:
            path = os.path.join(output_dir, split, cls)
            os.makedirs(path, exist_ok=True)


def split_dataset():
    """执行数据集分割"""
    random.seed(seed)

    # 统计总文件数
    total_files = 0
    for cls in class_folders:
        cls_path = os.path.join(source_dir, cls)
        total_files += len(os.listdir(cls_path))

    print(f"开始处理数据集，总文件数：{total_files}")

    # 处理每个类别
    for cls in tqdm(class_folders, desc="处理类别"):
        src_cls_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(src_cls_dir) if f.endswith('.nii.gz')]
        random.shuffle(files)

        # 计算分割点
        n_total = len(files)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])

        # 分割文件列表
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # 复制文件
        for split, file_list in zip(
                ['train', 'val', 'test'],
                [train_files, val_files, test_files]
        ):
            for fname in tqdm(file_list, desc=f"复制 {cls} 到 {split}", leave=False):
                src_path = os.path.join(src_cls_dir, fname)
                dst_path = os.path.join(output_dir, split, cls, fname)
                shutil.copy2(src_path, dst_path)

        # 打印分割信息
        print(f"\n类别 {cls} 分割结果:")
        print(f"训练集: {len(train_files)} 文件")
        print(f"验证集: {len(val_files)} 文件")
        print(f"测试集: {len(test_files)} 文件")


def validate_split():
    """验证分割结果"""
    print("\n验证分割结果:")
    for split in split_ratios.keys():
        total = 0
        for cls in class_folders:
            path = os.path.join(output_dir, split, cls)
            count = len(os.listdir(path))
            total += count
            print(f"{split}/{cls}: {count} 文件")
        print(f"{split} 总计: {total} 文件\n")


if __name__ == "__main__":
    # 检查比例总和是否为1
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-9, "分割比例总和必须为1"

    # 创建目录结构
    print("正在创建目录结构...")
    create_directory_structure()

    # 执行分割
    print("开始分割数据集...")
    split_dataset()

    # 验证结果
    validate_split()

    print("数据集分割完成！输出目录：", output_dir)