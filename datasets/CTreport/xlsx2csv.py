import pandas as pd
import os

# 设置数据目录
data_dir = r"E:\workplace\3D\datasets\CTreport"

# 获取所有Excel文件
excel_files = [r"E:\workplace\3D\datasets\CTreport\协和预后随访复发与否.xlsx",]

# 逐个转换
for excel_file in excel_files:
    # 构建完整路径
    excel_path = os.path.join(data_dir, excel_file)
    # 构建输出CSV路径
    csv_path = os.path.join(data_dir, excel_file.replace('.xlsx', '.csv'))
    try:
        # 读取Excel
        df = pd.read_excel(excel_path)

        # 保存为CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"成功将 {excel_file} 转换为 {os.path.basename(csv_path)}")
    except Exception as e:
        print(f"转换 {excel_file} 时出错: {str(e)}")

print("全部转换完成!")