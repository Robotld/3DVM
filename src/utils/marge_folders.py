# 合并两个具有相同结构的文件夹

import os
import shutil
from pathlib import Path
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('folder_merge.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def merge_folders(source_folders, target_folder):
    logger = setup_logging()

    # 创建目标文件夹
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)

    # 用于记录处理的文件数量
    file_count = 0

    try:
        # 遍历每个源文件夹
        for source_folder in source_folders:
            source_path = Path(source_folder)

            # 确保源文件夹存在
            if not source_path.exists():
                logger.error(f"源文件夹不存在: {source_folder}")
                continue

            # 遍历子文件夹（0, 1, ...）
            for subfolder in source_path.iterdir():
                if subfolder.is_dir():
                    # 创建目标子文件夹
                    target_subfolder = target_path / subfolder.name
                    target_subfolder.mkdir(exist_ok=True)

                    # 复制文件
                    for file in subfolder.glob('*'):
                        if file.is_file():
                            # 构建目标文件路径
                            target_file = target_subfolder / file.name

                            # 如果目标文件已存在，添加后缀以避免覆盖
                            if target_file.exists():
                                base_name = target_file.stem
                                extension = target_file.suffix
                                counter = 1
                                while target_file.exists():
                                    target_file = target_subfolder / f"{base_name}_{counter}{extension}"
                                    counter += 1

                            # 复制文件
                            shutil.move(file, target_file)
                            file_count += 1

                            if file_count % 100 == 0:
                                logger.info(f"已处理 {file_count} 个文件...")

    except Exception as e:
        logger.error(f"合并过程中出现错误: {str(e)}")
        raise

    logger.info(f"合并完成，共处理 {file_count} 个文件")
    return file_count


def main():
    # 源文件夹列表
    source_folders = [
        r"G:\datasets3D\data3d\nodule_patches96_1",
        r"G:\datasets3D\data3d\nodule_patches96_4"
    ]

    # 目标文件夹
    target_folder = r"G:\datasets3D\data3d\nodule_patches96_all"

    try:
        total_files = merge_folders(source_folders, target_folder)
        print(f"成功完成文件夹合并！共处理 {total_files} 个文件")
    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()