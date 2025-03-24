#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结节特征提取模块
"""

import pandas as pd
import numpy as np


def extract_nodule_features(df):
    """
    从合并报告中提取结节特征

    Args:
        df: 包含合并报告的DataFrame

    Returns:
        更新后的DataFrame，包含结节特征列
    """
    if df is None or '合并报告' not in df.columns:
        print("无法提取特征：数据未加载或无报告文本")
        return df

    # 定义要提取的结节特征关键词
    nodule_features = {
        '磨玻璃结节': ['磨玻璃', 'GGO', 'GGN', '毛玻璃'],
        '实性结节': ['实性结节', '实性肿块', '实质性结节'],
        '半实性结节': ['半实性', '部分实性', '混合型'],
        '分叶状': ['分叶', '分叶状'],
        '毛刺': ['毛刺', '棘刺'],
        '胸膜凹陷': ['胸膜凹陷', '胸膜牵拉'],
        '空泡': ['空泡', '空洞', '空气支气管征'],
        '钙化': ['钙化'],
        '血管集束征': ['血管集束', '血管聚集']
    }

    # 提取特征
    for feature, keywords in nodule_features.items():
        pattern = '|'.join(keywords)
        df[feature] = df['合并报告'].str.contains(
            pattern, case=False, na=False, regex=True
        ).astype(int)

    # 统计特征频率
    feature_counts = {f: df[f].sum() for f in nodule_features.keys()}
    feature_counts = dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))

    print("结节特征提取完成. 特征出现频率:")
    for feature, count in feature_counts.items():
        print(f"{feature}: {count}例 ({count / len(df) * 100:.1f}%)")

    return df