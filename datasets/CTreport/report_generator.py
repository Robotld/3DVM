#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
报告生成模块
"""

import os


def generate_report(df, results, output_dir):
    """
    生成统计分析报告

    Args:
        df: 处理后的DataFrame
        results: 统计分析结果
        output_dir: 输出目录路径
    """
    if df is None or not results:
        print("无法生成报告：数据未加载或分析结果为空")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 定义结节特征列表
    nodule_features = ['磨玻璃结节', '实性结节', '半实性结节', '分叶状', '毛刺',
                       '胸膜凹陷', '空泡', '钙化', '血管集束征']

    with open(f"{output_dir}/statistical_report.md", "w", encoding="utf-8") as f:
        f.write("# 肺结节数据统计分析报告\n\n")

        # 1. 基本统计信息
        f.write("## 1. 基本统计信息\n\n")
        f.write(f"- 总样本数: {results['基本统计']['样本数']}\n\n")

        # 性别分布
        if results['基本统计']['性别分布']:
            f.write("### 1.1 性别分布\n\n")
            total = sum(results['基本统计']['性别分布'].values())
            f.write("| 性别 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            for sex, count in results['基本统计']['性别分布'].items():
                f.write(f"| {sex} | {count} | {count / total * 100:.1f}% |\n")
            f.write("\n")

        # 年龄分布
        if results['基本统计']['年龄统计']:
            f.write("### 1.2 年龄分布\n\n")
            stats = results['基本统计']['年龄统计']
            f.write(f"- 平均年龄: {stats['mean']:.1f}岁\n")
            f.write(f"- 中位年龄: {stats['50%']:.1f}岁\n")
            f.write(f"- 标准差: {stats['std']:.1f}岁\n")
            f.write(f"- 最小年龄: {stats['min']:.1f}岁\n")
            f.write(f"- 最大年龄: {stats['max']:.1f}岁\n\n")

        # 结节大小
        if results['基本统计']['结节大小统计']:
            f.write("### 1.3 结节大小\n\n")
            stats = results['基本统计']['结节大小统计']
            f.write(f"- 平均大小: {stats['mean']:.2f}mm\n")
            f.write(f"- 中位大小: {stats['50%']:.2f}mm\n")
            f.write(f"- 标准差: {stats['std']:.2f}mm\n")
            f.write(f"- 最小大小: {stats['min']:.2f}mm\n")
            f.write(f"- 最大大小: {stats['max']:.2f}mm\n\n")

        # 结节部位
        if results['基本统计']['结节部位分布']:
            f.write("### 1.4 结节部位分布\n\n")
            total = sum(results['基本统计']['结节部位分布'].values())
            f.write("| 部位 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            for location, count in results['基本统计']['结节部位分布'].items():
                f.write(f"| {location} | {count} | {count / total * 100:.1f}% |\n")
            f.write("\n")

        # 2. 影像学特征
        f.write("## 2. 影像学特征\n\n")

        feature_freq = {f: df[f].sum() for f in nodule_features if f in df.columns}
        if feature_freq:
            # 按频率降序排列
            feature_freq = dict(sorted(feature_freq.items(), key=lambda x: x[1], reverse=True))

            f.write("| 特征 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            for feature, count in feature_freq.items():
                percentage = count / len(df) * 100
                f.write(f"| {feature} | {count} | {percentage:.1f}% |\n")
            f.write("\n")

        # 3. 分类分布
        f.write("## 3. 分类与分级\n\n")

        if results['分类统计']:
            f.write("### 3.1 分类分布\n\n")
            total = sum(results['分类统计'].values())
            f.write("| 分类 | 描述 | 数量 | 百分比 |\n")
            f.write("|------|------|------|--------|\n")

            class_descriptions = {
                2: "原位癌",
                3: "微浸润",
                4: "浸润"
            }

            for cls, count in sorted(results['分类统计'].items()):
                cls_int = int(cls)
                desc = class_descriptions.get(cls_int, "")
                f.write(f"| {cls_int} | {desc} | {count} | {count / total * 100:.1f}% |\n")
            f.write("\n")

        # 浸润性结节分级
        if results['浸润性分级']['分级分布']:
            f.write("### 3.2 浸润性结节(分类4)分级分布\n\n")
            f.write(f"- 浸润性结节总数: {results['浸润性分级']['样本数']}例\n\n")

            grade_descriptions = {
                1: "高分化",
                2: "中分化",
                3: "低分化"
            }

            total = sum(results['浸润性分级']['分级分布'].values())
            f.write("| 分级 | 描述 | 数量 | 百分比 |\n")
            f.write("|------|------|------|--------|\n")

            for grade, count in sorted(results['浸润性分级']['分级分布'].items()):
                grade_int = int(grade)
                desc = grade_descriptions.get(grade_int, "")
                f.write(f"| {grade_int} | {desc} | {count} | {count / total * 100:.1f}% |\n")
            f.write("\n")

        # 4. 结节大小与分类的关系
        if 'F' in results.get('大小与分类ANOVA', {}):
            f.write("## 4. 结节大小与分类的关系\n\n")

            # 计算各分类的结节大小统计量
            f.write("| 分类 | 样本数 | 平均大小(mm) | 标准差 | 中位大小(mm) |\n")
            f.write("|------|--------|--------------|--------|------------|\n")

            for cls in sorted(df['分类'].unique()):
                if pd.notna(cls):
                    cls_data = df[df['分类'] == cls]['大小mm'].dropna()
                    if len(cls_data) > 0:
                        f.write(
                            f"| {int(cls)} | {len(cls_data)} | {cls_data.mean():.2f} | {cls_data.std():.2f} | {cls_data.median():.2f} |\n")

            # ANOVA结果
            f_stat = results['大小与分类ANOVA']['F']
            p_value = results['大小与分类ANOVA']['p']

            f.write(f"\n- ANOVA检验结果: F = {f_stat:.2f}, p = {p_value:.4f}")
            if p_value < 0.05:
                f.write(" (不同分类间结节大小有统计学显著差异)\n\n")

                # Tukey HSD结果
                if 'tukey' in results['大小与分类ANOVA']:
                    f.write("- Tukey HSD事后多重比较结果:\n\n")
                    f.write("| 组1 | 组2 | 均值差(mm) | p值 | 显著性 |\n")
                    f.write("|-----|-----|------------|-----|--------|\n")

                    for result in results['大小与分类ANOVA']['tukey']:
                        sig = "是" if result['significant'] else "否"
                        f.write(
                            f"| {result['group1']} | {result['group2']} | {result['meandiff']:.2f} | {result['p']:.4f} | {sig} |\n")
            else:
                f.write(" (不同分类间结节大小无统计学显著差异)\n")

            f.write("\n")

        # 5. 结节特征与分类的关系
        if results.get('特征与分类关系'):
            f.write("## 5. 结节特征与分类的关系\n\n")

            for feature, data in results['特征与分类关系'].items():
                f.write(f"### 5.{nodule_features.index(feature) + 1} {feature}与分类的关系\n\n")

                # 各分类中特征出现的比例
                f.write("| 分类 | 特征出现比例 | p值 | 统计显著性 |\n")
                f.write("|------|------------|-----|------------|\n")

                for cls in sorted(data['proportions'].keys()):
                    prop = data['proportions'][cls]
                    p_value = data['p_values'].get(cls, float('nan'))
                    sig = "是" if p_value < 0.05 else "否"

                    f.write(f"| {cls} | {prop:.1f}% | {p_value:.4f} | {sig} |\n")
                f.write("\n")

        # 6. 浸润性结节特征与分级的关系
        if results.get('浸润性结节特征与分级'):
            f.write("## 6. 浸润性结节特征与分级的关系\n\n")

            for feature, data in results.get('浸润性结节特征与分级', {}).items():
                if isinstance(data, dict) and 'chi2' in data:
                    f.write(f"### 6.{nodule_features.index(feature) + 1} {feature}与分级的关系\n\n")

                    # 卡方检验结果
                    f.write(f"- 卡方检验: χ² = {data['chi2']:.2f}, p = {data['p']:.4f}")
                    if data['significant']:
                        f.write(" (特征与分级存在统计学显著关联)\n\n")
                    else:
                        f.write(" (特征与分级无统计学显著关联)\n\n")

                    # 列联表
                    if 'table' in data:
                        f.write("- 特征与分级列联表:\n\n")
                        f.write("| 特征 | " + " | ".join(
                            [f"分级{grade}" for grade in sorted(data['table'].keys())]) + " |\n")
                        f.write("|------|" + "|".join(["-----" for _ in data['table']]) + "|\n")

                        # 假设table是一个嵌套字典，外层是特征(0/1)，内层是分级
                        for feature_val in [0, 1]:
                            feature_status = "不存在" if feature_val == 0 else "存在"
                            row_data = []

                            for grade in sorted(data['table'].keys()):
                                grade_data = data['table'][grade]
                                count = grade_data.get(feature_val, 0)
                                row_data.append(str(count))

                            f.write(f"| {feature_status} | " + " | ".join(row_data) + " |\n")
                        f.write("\n")

        # 7. 结论和建议
        f.write("## 7. 结论和建议\n\n")

        f.write("### 7.1 主要发现\n\n")

        # 提取显著相关的特征
        significant_features = []
        for feature, data in results.get('特征与分类关系', {}).items():
            for cls, p_value in data['p_values'].items():
                if p_value < 0.05:
                    significant_features.append(f"{feature}与分类{cls}")

        if significant_features:
            f.write("1. 在统计分析中发现以下结节特征与分类之间存在显著相关性:\n")
            for sf in significant_features:
                f.write(f"   - {sf}\n")
            f.write("\n")

        # 结节大小与分类的关系
        if 'F' in results.get('大小与分类ANOVA', {}) and results['大小与分类ANOVA']['significant']:
            f.write("2. 结节大小在不同分类间存在统计学显著差异 (p < 0.05)，可能是重要的预测因子。\n\n")

        # 浸润性结节特征与分级的关系
        significant_grade_features = []
        for feature, data in results.get('浸润性结节特征与分级', {}).items():
            if isinstance(data, dict) and data.get('significant'):
                significant_grade_features.append(feature)

        if significant_grade_features:
            f.write(f"3. 在浸润性结节(分类4)中，以下特征与分级显著相关:\n")
            for feature in significant_grade_features:
                f.write(f"   - {feature}\n")
            f.write("\n")

        f.write("### 7.2 建议\n\n")
        f.write("基于统计分析结果，对于深度学习模型的建议如下:\n\n")

        f.write("1. **特征选择**:\n")
        f.write("   - 应包含年龄、性别、结节大小和结节部位等基本特征\n")
        f.write("   - 纳入统计显著的影像学特征，特别是")

        if significant_features:
            f.write("以下特征: " + ", ".join([f.split("与")[0] for f in significant_features[:3]]) + "\n")
        else:
            f.write("所有提取的影像学特征\n")

        f.write("\n2. **模型设计**:\n")
        f.write("   - 考虑使用层级分类方法: 先分类，对于浸润性结节(分类4)再进行分级\n")
        f.write("   - 结节大小应作为重要特征纳入模型\n")

        f.write("\n3. **验证策略**:\n")
        f.write("   - 建议使用分层抽样进行交叉验证，确保各类别样本在训练集和测试集中的比例一致\n")
        f.write("   - 对于样本量较少的类别，考虑使用数据增强技术\n")

    print(f"统计分析报告已生成: {output_dir}/statistical_report.md")
    return True