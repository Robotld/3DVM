#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计分析模块
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_statistical_analysis(df):
    """
    执行统计分析

    Args:
        df: 处理后的DataFrame

    Returns:
        统计分析结果字典
    """
    if df is None:
        print("无法执行统计分析：数据未加载")
        return None

    # 定义结节特征列表
    nodule_features = ['磨玻璃结节', '实性结节', '半实性结节', '分叶状', '毛刺',
                       '胸膜凹陷', '空泡', '钙化', '血管集束征']

    results = {}

    # 1. 基本描述性统计
    results['基本统计'] = {
        '样本数': len(df),
        '年龄统计': df['年龄'].describe().to_dict() if '年龄' in df else None,
        '性别分布': df['性别'].map({1: '男', 2: '女'}).value_counts().to_dict() if '性别' in df else None,
        '结节大小统计': df['大小mm'].describe().to_dict() if '大小mm' in df else None,
        '结节部位分布': df['结节部位'].map(
            {1: '左上', 2: '左中', 3: '左下', 4: '右上', 5: '右中', 6: '右下'}
        ).value_counts().to_dict() if '结节部位' in df else None
    }

    # 2. 分类分级统计
    results['分类统计'] = df['分类'].value_counts().to_dict() if '分类' in df else None

    # 仅对浸润性结节(分类4)分析分级
    invasive_df = df[df['分类'] == 4].copy()
    results['浸润性分级'] = {
        '样本数': len(invasive_df),
        '分级分布': invasive_df['分级'].value_counts().to_dict() if '分级' in invasive_df else None
    }

    # 3. 特征与分类的关系分析
    feature_class_association = analyze_feature_class_relationship(df, nodule_features)
    results['特征与分类关系'] = feature_class_association

    # 4. 结节大小与分类的关系(ANOVA)
    results['大小与分类ANOVA'] = analyze_size_classification_relationship(df)

    # 5. 浸润性结节特征与分级的关系
    feature_grade_association = analyze_feature_grade_relationship(invasive_df, nodule_features)
    results['浸润性结节特征与分级'] = feature_grade_association

    return results


def analyze_feature_class_relationship(df, nodule_features):
    """分析结节特征与分类的关系"""
    feature_class_association = {}

    if df is None or not '分类' in df.columns:
        return feature_class_association

    for feature in nodule_features:
        if feature in df.columns and df['分类'].notna().any():
            # 计算每个分类中特征的出现比例
            class_proportions = {}
            for cls in sorted(df['分类'].unique()):
                if pd.notna(cls):
                    cls_df = df[df['分类'] == cls]
                    prop = cls_df[feature].mean() * 100  # 百分比
                    class_proportions[int(cls)] = prop

            # Fisher精确检验(对每个分类)
            p_values = {}
            for cls in sorted(df['分类'].unique()):
                if pd.notna(cls):
                    contingency = pd.crosstab(df[feature], df['分类'] == cls)
                    if contingency.shape == (2, 2):  # 确保有足够数据
                        _, p_value = stats.fisher_exact(contingency)
                        p_values[int(cls)] = p_value

            feature_class_association[feature] = {
                'proportions': class_proportions,
                'p_values': p_values
            }

    return feature_class_association


def analyze_size_classification_relationship(df):
    """分析结节大小与分类的关系"""
    result = {}

    if df is None or not ('分类' in df.columns and '大小mm' in df.columns):
        return result

    if df['分类'].notna().any() and df['大小mm'].notna().any():
        categories = sorted([c for c in df['分类'].unique() if pd.notna(c)])
        size_by_class = [df[df['分类'] == c]['大小mm'].dropna() for c in categories]

        if len(categories) >= 2 and all(len(s) > 0 for s in size_by_class):
            try:
                f_stat, p_value = stats.f_oneway(*size_by_class)

                result = {
                    'F': f_stat,
                    'p': p_value,
                    'significant': p_value < 0.05
                }

                # Tukey HSD事后检验(如果ANOVA显著)
                if p_value < 0.05:
                    all_data = np.concatenate(size_by_class)
                    all_labels = np.concatenate([[c] * len(size_by_class[i]) for i, c in enumerate(categories)])
                    tukey = pairwise_tukeyhsd(all_data, all_labels, alpha=0.05)

                    # 转换Tukey结果为字典
                    tukey_results = []
                    for i in range(len(tukey.pvalues)):
                        group1 = int(tukey.groupsunique[tukey.data[i, 0]])
                        group2 = int(tukey.groupsunique[tukey.data[i, 1]])
                        tukey_results.append({
                            'group1': group1,
                            'group2': group2,
                            'meandiff': tukey.meandiffs[i],
                            'p': tukey.pvalues[i],
                            'significant': tukey.reject[i]
                        })

                    result['tukey'] = tukey_results
            except Exception as e:
                print(f"执行ANOVA检验时出错: {str(e)}")
                result = {"error": "执行ANOVA检验时出错"}

    return result


def analyze_feature_grade_relationship(invasive_df, nodule_features):
    """分析浸润性结节特征与分级的关系"""
    feature_grade_association = {}

    if invasive_df is None or len(invasive_df) == 0:
        return feature_grade_association

    for feature in nodule_features:
        if feature in invasive_df.columns and invasive_df['分级'].notna().any():
            # 特征与分级的列联表
            contingency = pd.crosstab(invasive_df[feature], invasive_df['分级'])

            if contingency.shape[0] == 2 and contingency.shape[1] >= 2:  # 确保有足够数据
                try:
                    chi2, p, _, _ = stats.chi2_contingency(contingency)
                    feature_grade_association[feature] = {
                        'chi2': chi2,
                        'p': p,
                        'significant': p < 0.05,
                        'table': contingency.to_dict()
                    }
                except Exception as e:
                    print(f"执行卡方检验时出错: {str(e)}")
                    feature_grade_association[feature] = {"error": "执行卡方检验时出错"}

    return feature_grade_association