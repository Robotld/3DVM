#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
肺结节数据统计分析主程序
"""

import os
from data_loader import load_and_process_data
from feature_extractor import extract_nodule_features
from statistical_analyzer import run_statistical_analysis
from visualizer import generate_visualizations
from report_generator import generate_report


def main():
    """主函数，执行完整的分析流程"""
    print("肺结节数据统计分析程序")
    print("=" * 50)

    output_dir = "nodule_analysis_results"
    os.makedirs(output_dir, exist_ok=True)


    # 获取用户确认的文件路径
    file_paths = [r"E:\workplace\3D\datasets\CTreport\1.csv", r"E:\workplace\3D\datasets\CTreport\2.csv", r"E:\workplace\3D\datasets\CTreport\3.csv", r"E:\workplace\3D\datasets\CTreport\4.csv"]


    print("\n开始数据分析流程...")

    # 1. 加载和处理数据
    print("\n第1步: 加载和处理数据")
    df = load_and_process_data(file_paths)
    if df is None:
        print("数据处理失败，程序退出")
        return

    # 2. 提取结节特征
    print("\n第2步: 提取结节特征")
    df = extract_nodule_features(df)

    # 3. 执行统计分析
    print("\n第3步: 执行统计分析")
    results = run_statistical_analysis(df)

    # 4. 生成可视化
    print("\n第4步: 生成可视化图表")
    generate_visualizations(df, results, output_dir)

    # 5. 生成统计报告
    print("\n第5步: 生成统计分析报告")
    generate_report(df, results, output_dir)

    print(f"\n分析完成! 所有结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main()