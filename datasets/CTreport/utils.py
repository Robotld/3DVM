#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
"""

import numpy as np
import re


def extract_size_value(size_str):
    """
    从可能复杂的大小字符串(如 "0.5cm×0.4cm")中提取数值
    如果有多个数值，取其平均值

    Args:
        size_str: 结节大小字符串

    Returns:
        提取的数值，无法提取则返回np.nan
    """
    if pd.isna(size_str) or size_str == '':
        return np.nan

    try:
        # 如果已经是数字，直接返回
        if isinstance(size_str, (int, float)):
            return float(size_str)

        # 尝试直接转换字符串为浮点数
        try:
            return float(size_str)
        except:
            pass

        # 使用正则表达式提取所有数字
        numbers = re.findall(r'\d+\.?\d*', str(size_str))
        if not numbers:
            return np.nan

        # 将所有提取的字符串转换为浮点数
        values = [float(num) for num in numbers]

        # 返回平均值
        return sum(values) / len(values)
    except:
        print(f"无法处理的大小值: {size_str}")
        return np.nan


# 添加缺失的pandas导入
import pandas as pd