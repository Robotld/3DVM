import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import f1_score

def calculate_metrics(metrics_dict, num_classes, threshold_method='f1'):
    """
    计算所有需要的评估指标

    参数:
        metrics_dict: 包含预测概率和真实标签的字典
        num_classes: 类别数量
        threshold_method: 阈值选择方法，可选:
                         'youden' - 约登指数最大化 (TPR-FPR)
                         'f1' - F1分数最大化
                         'pr_balance' - 精确率和召回率平衡点
                         'fixed_0.5' - 固定阈值0.5

    返回:
        metrics_dict: 包含计算出的所有指标的字典
    """
    if num_classes > 2:
        return metrics_dict, None
    if 'probabilities' not in metrics_dict or 'labels' not in metrics_dict:
        return metrics_dict, None

    probs = metrics_dict['probabilities'][:, 1]  # 取正类概率
    labels = metrics_dict['labels']

    # 根据选择的方法确定阈值
    if threshold_method == '0.5':
        threshold = 0.5
    elif threshold_method == 'f1':
        # F1分数最大化
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        # 计算每个阈值的F1分数
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0
                     for p, r in zip(precision[:-1], recall[:-1])]
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
    elif threshold_method == 'pr_balance':
        # 精确率和召回率平衡点
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        # 找到精确率和召回率最接近的点
        optimal_idx = np.argmin(np.abs(precision[:-1] - recall[:-1]))
        threshold = thresholds[optimal_idx]
    elif threshold_method == "youden":  # 默认使用约登指数
        # 计算ROC曲线和寻找约登指数最大的阈值点
        fpr, tpr, thresholds = roc_curve(labels, probs)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    else:
        threshold = threshold_method

    # 使用确定的阈值进行预测
    predictions = (probs >= threshold).astype(int)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # 基本指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 敏感性/召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # 精确率和F1
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率/PPV
    f1 = 2 * (precision_val * sensitivity) / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
    # 额外指标
    ppv = precision_val  # Positive Predictive Value就是精确率
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')  # Positive Likelihood Ratio
    nlr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')  # Negative Likelihood Ratio

    # AUC需要单独计算
    auc_score = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0

    metrics = {
        'threshold': threshold,
        'threshold_method': threshold_method,
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # 敏感性
        'specificity': specificity,  # 特异性
        'precision': precision_val,  # 精确率
        'f1': f1,
        'auc': auc_score,
        'ppv': ppv,  # 阳性预测值
        'npv': npv,  # 阴性预测值
        'plr': plr,  # 阳性似然比
        'nlr': nlr,  # 阴性似然比
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,  # 保存原始混淆矩阵值便于后续分析
        'predictions': predictions  # 保存预测结果
    }
    metrics_dict.update(metrics)
    return metrics_dict