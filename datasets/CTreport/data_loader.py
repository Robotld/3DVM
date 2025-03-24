#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
import os
import re
from utils import extract_size_value


def load_and_process_data(file_paths):
    """
    加载和处理数据

    Args:
        file_paths: 包含四个CSV文件路径的列表

    Returns:
        处理后的DataFrame，失败则返回None
    """
    # 检查文件是否存在
    for path in file_paths:
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            return None

    try:
        # 读取四个CSV文件
        dfs = [pd.read_csv(path, encoding='utf-8') for path in file_paths]
        print(f"成功加载{len(dfs)}个CSV文件")

        # 统一列名
        col_names = [
            ['姓名', '性别', '年龄', '住院号', '送检日期', '术后结果', '分化', '检查日期', 'CT号', 'CT报告', '结节部位',
             '大小mm'],
            ['姓名', '性别', '年龄', '住院号', 'CT号', 'CT报告', '标本名称', '结节部位', '大小cm', '分级', '分类'],
            ['CT号', '放射编号', '姓名', '住院号', '性别', '年龄', '吸烟史', '分类', '分级', '影像学表现', '结节部位',
             '大小mm', '病理报告'],
            ['姓名', '性别', '年龄', '住院号', 'CT号', '影像学表现', '影像学诊断', '结节部位', '大小mm', '标本名称',
             '分类', '分级']
        ]

        for i, df in enumerate(dfs):
            df.columns = col_names[i]

        # 合并报告字段
        for df in dfs:
            report_cols = [col for col in df.columns if col in ['CT报告', '影像学表现', '影像学诊断']]
            reports = []
            for _, row in df.iterrows():
                row_reports = [str(row[col]) for col in report_cols if pd.notna(row.get(col, pd.NA))]
                reports.append(' '.join(row_reports) if row_reports else np.nan)
            df['合并报告'] = reports

        # 标准化字段
        for df in dfs:
            # 性别编码
            if '性别' in df.columns:
                df['性别'] = df['性别'].replace({'男': 1, '女': 2, '1': 1, '2': 2})
                df['性别'] = pd.to_numeric(df['性别'], errors='coerce')

            # 结节部位编码
            if '结节部位' in df.columns:
                df['结节部位'] = df['结节部位'].replace({
                    '左上': 1, '左中': 2, '左下': 3, '右上': 4, '右中': 5, '右下': 6
                })
                df['结节部位'] = pd.to_numeric(df['结节部位'], errors='coerce')

            # 大小统一为mm - 使用提取函数
            if '大小cm' in df.columns:
                print("处理大小cm字段...")
                df['大小mm'] = df['大小cm'].apply(
                    lambda x: extract_size_value(x) * 10 if pd.notna(x) else np.nan
                )

            # 处理已有的大小mm字段，确保数字格式
            if '大小mm' in df.columns:
                df['大小mm'] = df['大小mm'].apply(extract_size_value)

        # 合并数据集
        merged_df = dfs[0]
        for i in range(1, len(dfs)):
            merged_df = pd.merge(
                merged_df, dfs[i],
                on=['CT号', '住院号', '姓名'],
                how='outer',
                suffixes=(None, f'_{i}')
            )

        # 统一字段 (取第一个非空值)
        for feature in ['年龄', '性别', '结节部位', '大小mm', '分级', '分类', '合并报告']:
            cols = [col for col in merged_df.columns if feature in col]
            if cols:
                merged_df[feature] = np.nan
                for col in cols:
                    mask = merged_df[feature].isna() & merged_df[col].notna()
                    merged_df.loc[mask, feature] = merged_df.loc[mask, col]
                # 删除多余的列
                for col in cols:
                    if col != feature:
                        merged_df.drop(col, axis=1, inplace=True)

        # 数据类型转换
        merged_df['年龄'] = pd.to_numeric(merged_df['年龄'], errors='coerce')
        merged_df['大小mm'] = pd.to_numeric(merged_df['大小mm'], errors='coerce')
        merged_df['分类'] = pd.to_numeric(merged_df['分类'], errors='coerce')
        merged_df['分级'] = pd.to_numeric(merged_df['分级'], errors='coerce')

        # 分级只在分类为4(浸润性结节)时有意义
        merged_df['有效分级'] = np.where(merged_df['分类'] == 4, merged_df['分级'], np.nan)

        # 清除重复项
        merged_df = merged_df.drop_duplicates(subset=['CT号', '住院号', '姓名'])

        print(f"数据预处理完成，合并后的数据集包含 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列")
        return merged_df

    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None