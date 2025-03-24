#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据可视化模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_visualizations(df, results, output_dir):
    """
    生成可视化图表

    Args:
        df: 处理后的DataFrame
        results: 统计分析结果
        output_dir: 输出目录路径
    """
    if df is None:
        print("无法生成可视化：数据未加载")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 定义结节特征列表
    nodule_features = ['磨玻璃结节', '实性结节', '半实性结节', '分叶状', '毛刺',
                       '胸膜凹陷', '空泡', '钙化', '血管集束征']

    # 1. 基本分布可视化
    visualize_basic_distributions(df, output_dir)

    # 2. 特征频率分布
    visualize_feature_frequency(df, nodule_features, output_dir)

    # 3. 分类和分级分布
    visualize_classification_distribution(df, output_dir)

    # 4. 结节大小与分类的关系
    visualize_size_vs_classification(df, results, output_dir)

    # 5. 特征与分类的关系热力图
    visualize_feature_class_relationship(results, output_dir)

    print(f"可视化图表已保存到 {output_dir} 目录")


def visualize_basic_distributions(df, output_dir):
    """可视化基本分布"""
    # 1.1 年龄分布
    if df['年龄'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.histplot(df['年龄'].dropna(), bins=20, kde=True)
        plt.title('患者年龄分布')
        plt.xlabel('年龄')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        age_stats = df['年龄'].describe()
        stats_text = f"n = {int(age_stats['count'])}\n"
        stats_text += f"均值 = {age_stats['mean']:.1f}\n"
        stats_text += f"中位数 = {age_stats['50%']:.1f}\n"
        stats_text += f"标准差 = {age_stats['std']:.1f}"

        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.savefig(f"{output_dir}/age_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 1.2 性别分布
    if df['性别'].notna().any():
        plt.figure(figsize=(8, 6))
        sex_counts = df['性别'].map({1: '男', 2: '女'}).value_counts()
        sex_counts.plot(kind='bar', color=sns.color_palette("pastel"))
        plt.title('患者性别分布')
        plt.xlabel('性别')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        for i, count in enumerate(sex_counts):
            percentage = count / sex_counts.sum() * 100
            plt.text(i, count + 0.5, f"{count} ({percentage:.1f}%)", ha='center', va='bottom')

        plt.savefig(f"{output_dir}/sex_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 1.3 结节部位分布
    if df['结节部位'].notna().any():
        plt.figure(figsize=(10, 6))
        location_mapping = {1: '左上', 2: '左中', 3: '左下', 4: '右上', 5: '右中', 6: '右下'}
        location_counts = df['结节部位'].map(location_mapping).value_counts()
        location_counts.plot(kind='bar', color=sns.color_palette("pastel"))
        plt.title('结节部位分布')
        plt.xlabel('结节部位')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        for i, count in enumerate(location_counts):
            percentage = count / location_counts.sum() * 100
            plt.text(i, count + 0.5, f"{count} ({percentage:.1f}%)", ha='center', va='bottom')

        plt.savefig(f"{output_dir}/location_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()


def visualize_feature_frequency(df, nodule_features, output_dir):
    """可视化特征频率分布"""
    feature_freq = {f: df[f].sum() for f in nodule_features if f in df.columns}
    if feature_freq:
        # 按频率降序排列
        feature_freq = dict(sorted(feature_freq.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(12, 6))
        bars = plt.bar(feature_freq.keys(), feature_freq.values(), color=sns.color_palette("muted"))
        plt.title('结节影像特征频率分布')
        plt.xlabel('特征')
        plt.ylabel('频数')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        for bar, feature in zip(bars, feature_freq.keys()):
            count = feature_freq[feature]
            percentage = count / len(df) * 100
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()


def visualize_classification_distribution(df, output_dir):
    """可视化分类和分级分布"""
    # 2.1 分类分布
    if df['分类'].notna().any():
        plt.figure(figsize=(8, 6))
        class_counts = df['分类'].value_counts().sort_index()
        class_counts.plot(kind='bar', color=sns.color_palette("pastel"))
        plt.title('结节分类分布')
        plt.xlabel('分类')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        for i, count in enumerate(class_counts):
            percentage = count / class_counts.sum() * 100
            plt.text(i, count + 0.5, f"{count}\n({percentage:.1f}%)", ha='center', va='bottom')

        plt.savefig(f"{output_dir}/classification_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2.2 浸润性结节的分级分布
    invasive_df = df[df['分类'] == 4]
    if len(invasive_df) > 0 and invasive_df['分级'].notna().any():
        plt.figure(figsize=(8, 6))
        grade_counts = invasive_df['分级'].value_counts().sort_index()
        grade_counts.plot(kind='bar', color=sns.color_palette("pastel"))
        plt.title('浸润性结节(分类4)的分级分布')
        plt.xlabel('分级')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

        for i, count in enumerate(grade_counts):
            percentage = count / grade_counts.sum() * 100
            plt.text(i, count + 0.5, f"{count}\n({percentage:.1f}%)", ha='center', va='bottom')

        plt.savefig(f"{output_dir}/grade_distribution_invasive.png", dpi=300, bbox_inches='tight')
        plt.close()


def visualize_size_vs_classification(df, results, output_dir):
    """可视化结节大小与分类的关系"""
    if df['分类'].notna().any() and df['大小mm'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='分类', y='大小mm', data=df)
        plt.title('结节大小与分类的关系')
        plt.xlabel('分类')
        plt.ylabel('结节大小(mm)')
        plt.grid(True, alpha=0.3)

        # 添加ANOVA结果标注
        if 'F' in results.get('大小与分类ANOVA', {}):
            f_stat = results['大小与分类ANOVA']['F']
            p_value = results['大小与分类ANOVA']['p']
            significant = results['大小与分类ANOVA']['significant']

            stats_text = f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}"
            if significant:
                stats_text += "\n各分类间结节大小存在显著差异"
            else:
                stats_text += "\n各分类间结节大小无显著差异"

            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                         va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.savefig(f"{output_dir}/size_vs_classification.png", dpi=300, bbox_inches='tight')
        plt.close()


def visualize_feature_class_relationship(results, output_dir):
    """可视化特征与分类的关系热力图"""
    if not results.get('特征与分类关系'):
        print("无法生成特征与分类关系热力图：缺少分析结果数据")
        return

    # 提取数据
    features = []
    classes = set()
    for feature, data in results['特征与分类关系'].items():
        features.append(feature)
        classes.update(data['proportions'].keys())

    classes = sorted(classes)

    # 创建热力图数据矩阵
    heatmap_data = np.zeros((len(features), len(classes)))
    p_values = np.ones((len(features), len(classes)))

    for i, feature in enumerate(features):
        for j, cls in enumerate(classes):
            if cls in results['特征与分类关系'][feature]['proportions']:
                heatmap_data[i, j] = results['特征与分类关系'][feature]['proportions'][cls]

            if cls in results['特征与分类关系'][feature]['p_values']:
                p_values[i, j] = results['特征与分类关系'][feature]['p_values'][cls]

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f',
                xticklabels=classes, yticklabels=features)
    plt.title('结节特征与分类的关联热力图(百分比)')
    plt.xlabel('分类')
    plt.ylabel('结节特征')

    # 标记统计显著的单元格
    for i in range(len(features)):
        for j in range(len(classes)):
            if p_values[i, j] < 0.05:
                plt.text(j + 0.5, i + 0.5, '*', ha='center', va='center', color='red', fontsize=15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_vs_classification_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()