import pandas as pd
import glob
import re

file_paths = glob.glob(r"E:\workplace\3D\datasets\CTreport/*.csv")  # 修改为你的 CSV 目录
df_list = []

columns_mapping = {
    "姓名": "name", "性别": "gender", "年龄": "age", "住院号": "hospital_id",
    "CT号": "ct_id", "CT报告": "ct_report", "标本名称": "sample_name",
    "结节部位（1左上，2左中，3左下，4右上，5右中，6右下）": "nodule_location",
    "大小(cm)": "size_mm", "大小(mm)": "size_mm",
    "分类": "category", "分级": "grade"
}

for file in file_paths:
    df = pd.read_csv(file, encoding="utf-8")
    df.rename(columns=lambda x: columns_mapping.get(x, x), inplace=True)

    def parse_size_cm(value):
        if isinstance(value, str):
            numbers = re.findall(r"\d+\.\d+|\d+", value)  # 提取数字
            if "cm" in value:
                return max(map(float, numbers)) * 10  # 转换为 mm
            return max(map(float, numbers))
        return None

    df["size_mm"] = df["size_mm"].apply(parse_size_cm)

    df["size_mm"] = pd.to_numeric(df["size_mm"], errors="coerce")
    df = df.dropna(subset=["size_mm"])  # 删除 NaN 数据
    if "size_mm" not in df.columns or df["size_mm"].isna().all():
        print(f"警告: 文件 {file} 没有有效的 size_mm 数据，跳过该文件")
        continue

    df = df[["size_mm", "category", "grade", "ct_id"]].dropna()

    # 如果 category 是 "4"，则根据 grade 赋值
    df["category"] = df["category"].astype(str)
    df.loc[df["category"] == "4", "category"] = df.loc[df["category"] == "4", "grade"].apply(
        lambda x: f"4-{int(x)}" if pd.notna(x) else "4-1"
    )

    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)

# 确保 size_mm 为数值类型
df_all["size_mm"] = pd.to_numeric(df_all["size_mm"], errors="coerce")
df_all = df_all.dropna(subset=["size_mm"])  # 删除 NaN 数据

# 统计各分类的大小信息
stats = df_all.groupby("category")["size_mm"].describe()
print("结节大小统计信息：")
print(stats)

# 统计直径 ≤5mm 的结节数量
small_nodules = df_all[df_all["size_mm"] < 5].groupby("category")["size_mm"].count()
print("\n直径 ≤5mm 的结节数量：")
print(small_nodules)

# 统计直径 ≤9mm 的结节数量
num = df_all[df_all["size_mm"] >= 26].groupby("category")["size_mm"].count()
print("\n直径 >14mm 的结节数量：")
print(num)

num = df_all[df_all["size_mm"] >= 26]
count_2_3 = num[num["category"].isin(["2", "3"])]
# print("\n直径为2.2mm的结节的CT号：")
print(num.shape[0], count_2_3.shape[0], count_2_3.shape[0] / num.shape[0])

# # 输出直径为2.2mm的结节的CT号
# nodules = df_all[(df_all["size_mm"] > 14) & (df_all["category"].str.contains("4"))]["ct_id"]



# 计算当结节大小为多少时，2 3类的占比低于5%
total_count = len(df_all)
threshold = 0
ratio = 0.1
for size in sorted(df_all["size_mm"].unique()):
    filtered_df = df_all[df_all["size_mm"] >= size]
    count_2_3 = filtered_df[filtered_df["category"].isin(["2", "3"])].shape[0]
    count_4 = filtered_df[filtered_df["category"].str.contains("4")].shape[0]
    proportion_2_3 = count_2_3 / filtered_df.shape[0] if filtered_df.shape[0] > 0 else 0
    if proportion_2_3 <= ratio:
        threshold = size
        print(f"\n结节大小为 {threshold} mm 时，类别 2 和 3 的占比低于{ratio}。")
        print(f"4类占比{count_4}\n")
        break

