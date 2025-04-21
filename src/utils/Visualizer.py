import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats


def set_journal_style():
    """设置符合医学顶刊要求的图表样式"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'  # 医学期刊常用字体
    plt.rcParams['svg.fonttype'] = 'none'  # 确保文本在SVG中可编辑
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.figsize'] = (4, 3.5)  # 单栏图尺寸


def plot_learning_curves(train_metrics, val_metrics, save_path=None):
    """
    绘制训练过程中的学习曲线，符合医学期刊标准

    参数:
        train_metrics: 包含训练指标的字典列表，每个元素对应一个epoch
        val_metrics: 包含验证指标的字典列表，每个元素对应一个epoch
        save_path: 保存路径，如果为None则显示图像而不保存
    """
    metrics = ['loss', 'accuracy', 'f1', 'auc']
    epochs = range(1, len(train_metrics) + 1)

    # 设置最佳指标方向
    best_direction = {'loss': 'min', 'accuracy': 'max', 'f1': 'max', 'auc': 'max'}

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        if metric in train_metrics[0] and metric in val_metrics[0]:
            train_values = [m.get(metric, 0) for m in train_metrics]
            val_values = [m.get(metric, 0) for m in val_metrics]

            # 绘制曲线
            plt.plot(epochs, train_values, 'b-o', label=f'Training',
                     linewidth=1.5, markersize=3, alpha=0.8)
            plt.plot(epochs, val_values, 'r-o', label=f'Validation',
                     linewidth=1.5, markersize=3, alpha=0.8)

            # 标记训练集最佳值
            if best_direction[metric] == 'min':
                best_idx = np.argmin(train_values)
                best_value = min(train_values)
            else:
                best_idx = np.argmax(train_values)
                best_value = max(train_values)

            best_epoch = epochs[best_idx]
            plt.plot(best_epoch, best_value, 'bo', markersize=6,
                     markeredgecolor='black', markeredgewidth=1)
            plt.annotate(f'{best_value:.3f}', (best_epoch, best_value),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

            # 标记验证集最佳值
            if best_direction[metric] == 'min':
                best_idx = np.argmin(val_values)
                best_value = min(val_values)
            else:
                best_idx = np.argmax(val_values)
                best_value = max(val_values)

            best_epoch = epochs[best_idx]
            plt.plot(best_epoch, best_value, 'ro', markersize=6,
                     markeredgecolor='black', markeredgewidth=1)
            plt.annotate(f'{best_value:.3f}', (best_epoch, best_value),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

            # 设置图表属性
            plt.title(f'{metric.upper()} Curve', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel(metric.upper(), fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.3)
            plt.legend(loc='best')

            # 为loss设置下限为0，为其他指标设置0-1范围
            if metric == 'loss':
                ylim = plt.ylim()
                plt.ylim(0, ylim[1])
            elif metric in ['accuracy', 'f1', 'auc']:
                plt.ylim(-0.05, 1.05)

    plt.suptitle('Training and Validation Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(metrics_dict_list, labels=None, title="ROC Curve",
                   figsize=(5, 4.5), save_path=None):
    """
    绘制ROC曲线图表，支持多条曲线比较

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径，如不提供则显示图表
    """

    plt.figure(figsize=figsize)
    ax = plt.gca()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=1.5,
                label=f'{labels[i]} (AUC = {roc_auc:.3f})')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_pr_curve(metrics_dict_list, labels=None, title="Precision-Recall Curve",
                 figsize=(5, 4.5), save_path=None):
    """
    绘制精确率-召回率曲线，对不平衡数据集特别有用

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径，如不提供则显示图表
    """

    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        precision, recall, _ = precision_recall_curve(true_labels, probs)
        pr_auc = auc(recall, precision)

        # 计算基线水平 (即正类比例)
        baseline = np.mean(true_labels)

        plt.plot(recall, precision, color=colors[i % len(colors)], lw=1.5,
                label=f'{labels[i]} (AUPRC = {pr_auc:.3f})')

    # 绘制基线
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                label=f'Baseline (Prevalence = {baseline:.3f})')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(conf_matrix=None, metrics_dict=None, labels=None,
                          normalize=False, title='Confusion Matrix',
                          figsize=(5, 4.5), save_path=None,
                          cmap='Blues', annot_format='.2f'):
    """
    绘制混淆矩阵热力图

    参数:
        conf_matrix: 混淆矩阵数组，若不提供则从metrics_dict中提取
        metrics_dict: 包含混淆矩阵数据的字典
        labels: 类别标签，默认为['Negative', 'Positive']
        normalize: 是否归一化
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
        cmap: 热力图配色方案
        annot_format: 标注格式
    """
    set_journal_style()

    if conf_matrix is None and metrics_dict is not None:
        # 从metrics_dict中构建混淆矩阵
        tn = metrics_dict['tn']
        fp = metrics_dict['fp']
        fn = metrics_dict['fn']
        tp = metrics_dict['tp']
        conf_matrix = np.array([[tn, fp], [fn, tp]])

    if conf_matrix is None:
        raise ValueError("Either conf_matrix or metrics_dict must be provided")

    if labels is None:
        labels = ['Negative', 'Positive']

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)

    # 创建更适合医学期刊的对比度更高的配色方案
    if cmap == 'Blues':
        colors = ["#ffffff", "#0d47a1"]  # 从白色到深蓝
        custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors)
        cmap = custom_cmap

    sns.heatmap(conf_matrix, annot=True, fmt=annot_format, cmap=cmap,
                xticklabels=labels, yticklabels=labels, square=True,
                cbar=False, annot_kws={"size": 10})

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)

    # 添加性能指标注释
    if metrics_dict is not None:
        footnote = f"Sensitivity: {metrics_dict['sensitivity']:.2f}  "
        footnote += f"Specificity: {metrics_dict['specificity']:.2f}\n"
        footnote += f"PPV: {metrics_dict['ppv']:.2f}  "
        footnote += f"NPV: {metrics_dict['npv']:.2f}"

        plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_calibration_curve(metrics_dict_list, labels=None, n_bins=10,
                           title="Calibration Curve", figsize=(5, 4.5),
                           save_path=None):
    """
    绘制模型校准曲线

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        n_bins: 概率分箱数量
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """

    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=n_bins)

        plt.plot(prob_pred, prob_true, marker='o', markersize=4,
                 color=colors[i % len(colors)], lw=1.5, label=labels[i])

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_decision_curve_analysis(metrics_dict_list, labels=None,
                                title="Decision Curve Analysis",
                                figsize=(6, 4.5), save_path=None):
    """
    绘制决策曲线分析图 (Decision Curve Analysis)，医学顶刊常用

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每条曲线的标签列表
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """

    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    thresholds = np.linspace(0, 1, 101)[1:-1]  # 0.01 to 0.99

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    # 计算治疗所有人的净获益（水平线）
    all_patients = []

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']
        all_patients.append(np.mean(true_labels))

    # 使用第一个模型的正类率
    pt_prevalence = np.mean(all_patients)
    y_all = np.ones(len(thresholds)) * pt_prevalence

    # 绘制两条参考线
    plt.plot(thresholds, y_all, 'k--', lw=1, alpha=0.5, label='Treat All')
    plt.plot(thresholds, np.zeros_like(thresholds), 'k-', lw=1, alpha=0.5, label='Treat None')

    # 为每个模型计算并绘制决策曲线
    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]
        true_labels = metrics_dict['labels']

        net_benefit = []

        for threshold in thresholds:
            # 阈值处的决策
            pred_positive = (probs >= threshold).astype(int)

            # 真阳性和假阳性
            tp = np.sum((pred_positive == 1) & (true_labels == 1))
            fp = np.sum((pred_positive == 1) & (true_labels == 0))
            n = len(true_labels)

            # 计算净获益
            if np.sum(pred_positive) == 0:
                # 如果没有预测为阳性的样本
                nb = 0
            else:
                nb = (tp/n) - (fp/n) * (threshold/(1-threshold))

            net_benefit.append(nb)

        plt.plot(thresholds, net_benefit, color=colors[i % len(colors)],
                 lw=1.5, label=labels[i])

    plt.xlim([0, 1])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_metrics_comparison(metrics_dict_list, metric_names=None,
                           labels=None, title="Performance Metrics Comparison",
                           figsize=(7, 5), save_path=None):
    """
    绘制多个模型多指标对比条形图

    参数:
        metrics_dict_list: 包含多个模型评估指标的字典列表
        metric_names: 要比较的指标名称列表，默认为['auc', 'sensitivity', 'specificity', 'precision', 'f1']
        labels: 每个模型的标签
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """

    if metric_names is None:
        metric_names = ['auc', 'sensitivity', 'specificity', 'precision', 'f1']

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 确保标签列表长度匹配
    labels = labels[:len(metrics_dict_list)]

    # 准备数据
    metric_values = []
    for metrics_dict in metrics_dict_list:
        values = [metrics_dict.get(metric, np.nan) for metric in metric_names]
        metric_values.append(values)

    # 将数据转换为DataFrame
    df = pd.DataFrame(metric_values, columns=metric_names, index=labels)

    # 绘制条形图
    fig, ax = plt.subplots(figsize=figsize)

    # 获取条形间距
    n_bars = len(df)
    n_metrics = len(metric_names)
    bar_width = 0.8 / n_metrics
    r = np.arange(n_bars)

    # 为每个指标绘制条形
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']
    for i, metric in enumerate(metric_names):
        position = r + i * bar_width
        bars = ax.bar(position, df[metric], bar_width, alpha=0.7,
                      color=colors[i % len(colors)], label=metric.capitalize())

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 设置x轴标签位置和标题
    ax.set_xticks(r + bar_width * (n_metrics-1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Performance')
    ax.set_title(title)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # 添加网格线
    plt.grid(True, axis='y', linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_threshold_impact(metrics_dict, threshold_range=None,
                         metrics_to_plot=None, title="Threshold Impact Analysis",
                         figsize=(6, 4.5), save_path=None):
    """
    绘制阈值对各性能指标影响的曲线图

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        threshold_range: 阈值范围，默认为np.linspace(0.05, 0.95, 19)
        metrics_to_plot: 要绘制的指标列表，默认为['sensitivity', 'specificity', 'f1', 'precision']
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """


    if threshold_range is None:
        threshold_range = np.linspace(0.05, 0.95, 19)

    if metrics_to_plot is None:
        metrics_to_plot = ['sensitivity', 'specificity', 'f1', 'precision']

    # 提取预测概率和真实标签
    probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
    true_labels = metrics_dict['labels']

    # 计算每个阈值下的性能指标
    results = {metric: [] for metric in metrics_to_plot}

    for threshold in threshold_range:
        predictions = (probs >= threshold).astype(int)

        # 计算混淆矩阵元素
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tp = np.sum((predictions == 1) & (true_labels == 1))

        # 计算各指标
        if 'sensitivity' in metrics_to_plot:
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['sensitivity'].append(sensitivity)

        if 'specificity' in metrics_to_plot:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['specificity'].append(specificity)

        if 'precision' in metrics_to_plot:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            results['precision'].append(precision)

        if 'f1' in metrics_to_plot:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            results['f1'].append(f1)

    # 绘制曲线
    plt.figure(figsize=figsize)

    colors = {'sensitivity': '#1f77b4', 'specificity': '#ff7f0e',
              'precision': '#2ca02c', 'f1': '#d62728'}

    for metric in metrics_to_plot:
        plt.plot(threshold_range, results[metric], 'o-', lw=1.5,
                 label=metric.capitalize(), color=colors.get(metric, None))

    # 标记出最佳阈值
    if 'f1' in metrics_to_plot:
        best_idx = np.argmax(results['f1'])
        best_threshold = threshold_range[best_idx]
        plt.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7,
                   label=f'Best F1 @ {best_threshold:.2f}')

    plt.xlim([0, 1])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Decision Threshold')
    plt.ylabel('Performance')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_prediction_distribution(metrics_dict, bins=20,
                                title="Prediction Probability Distribution",
                                figsize=(6, 4), save_path=None):
    """
    绘制预测概率分布直方图，分离正例和负例

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        bins: 直方图的箱数
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """
    set_journal_style()

    probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
    true_labels = metrics_dict['labels']
    threshold = metrics_dict.get('threshold', 0.5)

    # 分离正例和负例
    pos_probs = probs[true_labels == 1]
    neg_probs = probs[true_labels == 0]

    plt.figure(figsize=figsize)

    # 绘制分布
    plt.hist(neg_probs, bins=bins, alpha=0.6, color='#ff7f0e',
             label=f'Negative (n={len(neg_probs)})', density=True)
    plt.hist(pos_probs, bins=bins, alpha=0.6, color='#1f77b4',
             label=f'Positive (n={len(pos_probs)})', density=True)

    # 标记阈值线
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Threshold = {threshold:.2f}')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend(loc='best')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_multiclass_roc(metrics_dict, title="Multi-class ROC Curve",
                      figsize=(6, 5), save_path=None):
    """
    绘制多分类ROC曲线 (One-vs-Rest)

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        title: 图表标题
        figsize: 图表尺寸
        save_path: 保存路径
    """
    set_journal_style()

    # 检查是否为多分类问题
    probs = metrics_dict['probabilities']
    true_labels = metrics_dict['labels']

    if probs.shape[1] <= 2:
        print("This function is intended for multi-class problems. Use plot_roc_curve for binary classification.")
        return

    n_classes = probs.shape[1]

    # 将标签进行独热编码
    y_onehot = np.zeros((len(true_labels), n_classes))
    for i in range(n_classes):
        y_onehot[:, i] = (true_labels == i).astype(int)

    plt.figure(figsize=figsize)

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    # 为每个类别计算ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f'Class {i} (AUC = {roc_auc:.3f})')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right')

    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def create_combined_performance_figure(metrics_dict_list, labels=None,
                                     figsize=(12, 10), save_path=None):
    """
    创建综合性能评估图表，包含ROC曲线、PR曲线、校准曲线和混淆矩阵

    参数:
        metrics_dict_list: 包含预测概率和真实标签的字典列表
        labels: 每个模型的标签列表
        figsize: 图表尺寸
        save_path: 保存路径
    """
    set_journal_style()

    if not isinstance(metrics_dict_list, list):
        metrics_dict_list = [metrics_dict_list]

    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(metrics_dict_list))]

    # 创建2x2网格图表
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # ROC曲线
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 绘制对角线
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=1.5,
                label=f'{labels[i]} (AUC = {roc_auc:.3f})')

    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_xlabel('1 - Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.3)

    # PR曲线
    ax2 = fig.add_subplot(gs[0, 1])

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        precision, recall, _ = precision_recall_curve(true_labels, probs)
        pr_auc = auc(recall, precision)

        # 计算基线水平 (即正类比例)
        baseline = np.mean(true_labels)

        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=1.5,
                label=f'{labels[i]} (AUPRC = {pr_auc:.3f})')

    # 绘制基线
    ax2.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
               label=f'Baseline (Prevalence = {baseline:.3f})')

    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle=':', alpha=0.3)

    # 校准曲线
    ax3 = fig.add_subplot(gs[1, 0])

    # 绘制对角线
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    for i, metrics_dict in enumerate(metrics_dict_list):
        probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
        true_labels = metrics_dict['labels']

        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=10)

        ax3.plot(prob_pred, prob_true, marker='o', markersize=4,
                color=colors[i % len(colors)], lw=1.5, label=labels[i])

    ax3.set_xlim([-0.01, 1.01])
    ax3.set_ylim([-0.01, 1.01])
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Observed Frequency')
    ax3.set_title('Calibration Curve')
    ax3.legend(loc='best')
    ax3.grid(True, linestyle=':', alpha=0.3)

    # 混淆矩阵 (使用第一个模型)
    ax4 = fig.add_subplot(gs[1, 1])

    metrics_dict = metrics_dict_list[0]
    tn = metrics_dict['tn']
    fp = metrics_dict['fp']
    fn = metrics_dict['fn']
    tp = metrics_dict['tp']
    conf_matrix = np.array([[tn, fp], [fn, tp]])

    # 归一化
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # 创建更适合医学期刊的对比度更高的配色方案
    colors = ["#ffffff", "#0d47a1"]  # 从白色到深蓝
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", colors)

    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap=custom_cmap,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                square=True, cbar=False, annot_kws={"size": 10}, ax=ax4)

    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    ax4.set_title(f'Confusion Matrix ({labels[0]})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
    else:
        plt.show()