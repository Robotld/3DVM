import os
import re
import shutil
from collections import defaultdict
import statistics


class CTAnnotationProcessor:
    def __init__(self, input_dir, output_dir):
        """初始化处理器"""
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def extract_ct_id_and_slice(self, filename):
        """从文件名中提取CT ID和切片编号"""
        # 匹配常见格式如CT660668_26或UCT202101051695-1.5_248等
        print(filename)
        if "CT" in filename or "UPT" in filename or "MR" in filename:
            CT, num = filename.split("_")[0], int(filename.split("_")[1])
            return CT, num
        return None, None

    def read_yolo_annotation(self, filepath):
        """读取YOLO格式标注文件"""
        annotations = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {e}")

        return annotations

    def write_yolo_annotation(self, filepath, annotations):
        """写入YOLO格式标注文件"""
        try:
            with open(filepath, 'w') as f:
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            return True
        except Exception as e:
            print(f"写入文件 {filepath} 时出错: {e}")
            return False

    def find_nodule_groups(self, slice_files):
        """找出连续的切片组（每个组代表一个可能的结节）"""
        if not slice_files:
            return []

        # 按切片编号排序
        slice_files.sort(key=lambda x: x[0])

        groups = []
        current_group = [slice_files[0]]

        for i in range(1, len(slice_files)):
            if slice_files[i][0] == slice_files[i - 1][0] + 1:
                current_group.append(slice_files[i])
            else:
                groups.append(current_group)
                current_group = [slice_files[i]]

        groups.append(current_group)
        return groups

    def calculate_bbox_area(self, annotation):
        """计算单个边界框面积"""
        _, _, _, width, height = annotation
        return width * height

    def calculate_average_bbox_area(self, files):
        """计算平均边界框面积"""
        total_area = 0
        count = 0

        for _, file in files:
            file_path = os.path.join(self.input_dir, file)
            annotations = self.read_yolo_annotation(file_path)

            for ann in annotations:
                area = self.calculate_bbox_area(ann)
                total_area += area
                count += 1

        if count == 0:
            return 0
        return total_area / count

    def calculate_best_annotation(self, files):
        """计算最佳标注（中心点和锚框大小）"""
        all_annotations = []

        for _, file in files:
            file_path = os.path.join(self.input_dir, file)
            annotations = self.read_yolo_annotation(file_path)
            all_annotations.extend(annotations)

        if not all_annotations:
            return None

        # 计算平均标注
        avg_annotation = [0, 0, 0, 0, 0]  # [class_id, x_center, y_center, width, height]
        for ann in all_annotations:
            for i in range(5):
                avg_annotation[i] += ann[i]

        for i in range(5):
            avg_annotation[i] /= len(all_annotations)

        # 对类别ID取整
        avg_annotation[0] = int(round(avg_annotation[0]))

        return tuple(avg_annotation)

    def find_center_slice(self, nodule_group):
        """找出中心切片"""
        slice_numbers = [slice_num for slice_num, _ in nodule_group]
        center_slice = statistics.median_low(slice_numbers)

        # 找到对应的文件
        for slice_num, file in nodule_group:
            if slice_num == center_slice:
                return center_slice, file

        # 如果没有找到准确匹配，则返回最接近的
        closest_slice = min(nodule_group, key=lambda x: abs(x[0] - center_slice))
        return closest_slice

    def generate_three_slices(self, ct_id, center_slice, best_annotation):
        """生成3张标注文件"""
        center_slice_num, center_file = center_slice
        result_files = []

        # 生成左中右三张切片的编号
        slice_numbers = [center_slice_num - 1, center_slice_num, center_slice_num + 1]

        for slice_num in slice_numbers:
            # 构建输出文件名
            output_filename = f"{ct_id}_{slice_num}.txt"
            output_path = os.path.join(self.output_dir, output_filename)

            # 写入标注
            success = self.write_yolo_annotation(output_path, [best_annotation])
            if success:
                result_files.append(output_filename)

        return result_files

    def process_all(self):
        """处理所有标签文件"""
        # 收集所有标签文件
        label_files = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]

        # 按CT ID分组
        ct_groups = defaultdict(list)
        for file in label_files:
            ct_id, slice_num = self.extract_ct_id_and_slice(file[:-4])
            if ct_id and slice_num is not None:
                ct_groups[ct_id].append((slice_num, file))

        print(f"总共发现 {len(ct_groups)} 个不同的CT号")

        # 处理计数
        patients_processed = 0
        patients_with_multiple_nodules = 0
        total_output_files = 0

        # 处理每个CT ID（每个患者）
        for ct_id, files in ct_groups.items():
            print(f"\n处理患者 {ct_id}，共有 {len(files)} 个标注文件")

            # 找出连续的切片组（不同结节）
            nodule_groups = self.find_nodule_groups(files)

            if len(nodule_groups) > 1:
                patients_with_multiple_nodules += 1
                print(f"  发现 {len(nodule_groups)} 个可能的结节组")

            # 如果没有连续的切片组，跳过
            if not nodule_groups:
                print(f"  未发现有效的结节组，跳过 {ct_id}")
                continue

            # 找出最大的结节（平均边界框面积最大）
            largest_nodule = None
            max_avg_area = 0

            for i, group in enumerate(nodule_groups):
                # 如果组内切片少于1张，跳过
                if len(group) < 1:
                    continue

                # 计算平均边界框面积
                avg_area = self.calculate_average_bbox_area(group)

                print(f"  结节组 #{i + 1}: 切片数: {len(group)}, 平均面积: {avg_area:.6f}")

                # 如果当前结节比已找到的最大结节更大，更新
                if avg_area > max_avg_area:
                    max_avg_area = avg_area
                    largest_nodule = group

            # 如果没有找到有效结节，跳过
            if not largest_nodule:
                print(f"  未找到有效结节，跳过 {ct_id}")
                continue

            # 计算最佳标注（从所有切片中）
            best_annotation = self.calculate_best_annotation(largest_nodule)

            # 如果没有有效标注，跳过
            if not best_annotation:
                print(f"  未找到有效标注，跳过 {ct_id}")
                continue

            # 找出中心切片
            center_slice = self.find_center_slice(largest_nodule)
            print(f"  找到中心切片: {center_slice[0]}")

            # 生成3张标注
            result_files = self.generate_three_slices(ct_id, center_slice, best_annotation)

            if result_files:
                patients_processed += 1
                total_output_files += len(result_files)
                print(f"  成功生成 {len(result_files)} 个标注文件: {', '.join(result_files)}")

        # 输出统计信息
        print("\n============ 处理统计 ============")
        print(f"总共处理了 {patients_processed} 个患者")
        print(f"有 {patients_with_multiple_nodules} 个患者存在多个结节")
        print(f"总共生成了 {total_output_files} 个输出文件")
        print(
            f"平均每个患者的输出文件数: {total_output_files / patients_processed if patients_processed > 0 else 0:.2f}")


def main():
    """主函数"""
    input_dir = "G:/data/labels"  # 输入目录
    output_dir = "G:/data/new_labels"  # 输出目录

    print("=" * 50)
    print("CT标注处理程序")
    print("=" * 50)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)

    # 备份输出目录(如果存在)
    if os.path.exists(output_dir):
        backup_dir = output_dir + "_backup"
        print(f"输出目录已存在，备份到 {backup_dir}")
        try:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(output_dir, backup_dir)
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        except Exception as e:
            print(f"备份失败: {e}")

    # 创建处理器并运行
    processor = CTAnnotationProcessor(input_dir, output_dir)
    processor.process_all()

    print("\n处理完成!")


if __name__ == "__main__":
    main()