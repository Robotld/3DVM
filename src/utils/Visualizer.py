"""
可视化工具模块
用于生成训练和评估过程中的各种可视化图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title="Confusion matrix"):
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径，如果为None则显示图像而不保存
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)

    # 标准化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 使用seaborn绘制更美观的热图
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")

    plt.title(title, fontsize=15)
    plt.ylabel('Ture label', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_curves(train_metrics, val_metrics, save_path=None):
    """
    绘制训练过程中的学习曲线

    参数:
        train_metrics: 包含训练指标的字典列表，每个元素对应一个epoch
        val_metrics: 包含验证指标的字典列表，每个元素对应一个epoch
        save_path: 保存路径，如果为None则显示图像而不保存
    """
    metrics = ['loss', 'accuracy', 'f1', 'auc']
    epochs = range(1, len(train_metrics) + 1)

    plt.figure(figsize=(16, 12))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        if metric in train_metrics[0] and metric in val_metrics[0]:
            train_values = [m.get(metric, 0) for m in train_metrics]
            val_values = [m.get(metric, 0) for m in val_metrics]

            plt.plot(epochs, train_values, 'b-', label=f'train {metric}')
            plt.plot(epochs, val_values, 'r-', label=f'Val {metric}')
            plt.title(f'{metric.upper()} curve', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel(metric.upper(), fontsize=12)
            plt.grid(True)
            plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curves(y_true, y_score, num_classes, save_path=None, class_names=None):
    """
    绘制ROC曲线

    参数:
        y_true: 真实标签
        y_score: 预测概率
        num_classes: 类别数量
        save_path: 保存路径
        class_names: 类别名称列表
    """
    plt.figure(figsize=(10, 8))

    # 为每个类别计算ROC曲线和ROC面积
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 将标签转换为one-hot编码
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i, val in enumerate(y_true):
        if val < num_classes:  # 确保标签在有效范围内
            y_true_onehot[i, val] = 1

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        class_label = class_names[i] if class_names and i < len(class_names) else f"class {i}"
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve ({class_label}) (area = {roc_auc[i]:.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True positive rate', fontsize=12)
    plt.title('ROC curves for each class', fontsize=15)
    plt.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_score, num_classes, save_path=None, class_names=None):
    """
    Plot Precision-Recall curve

    Parameters:
        y_true: true labels (1D array)
        y_score: prediction probabilities (2D array: samples x classes)
        num_classes: number of classes
        save_path: path to save the plot
        class_names: list of class names
    """
    plt.figure(figsize=(10, 8))

    # Use label_binarize for safer one-hot encoding conversion
    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
    else:
        # Special handling for binary classification
        y_true_bin = np.zeros((len(y_true), num_classes))
        for i, val in enumerate(y_true):
            y_true_bin[i, val] = 1

    # Create color map for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Store precision and recall values for each class for micro and macro averaging
    all_precisions = []
    all_recalls = []
    average_precisions = []

    # Plot PR curve for each class
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        all_precisions.append(precision)
        all_recalls.append(recall)

        # Calculate Average Precision (AP) - approximation of area under curve
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        average_precisions.append(ap)

        # Determine class label
        class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"

        # Plot PR curve
        plt.plot(recall, precision, color=colors[i], lw=2,
                 label=f'{class_label} (AP={ap:.3f})')

    # Calculate and display macro-average AP
    macro_ap = np.mean(average_precisions)

    # Add baseline (random classifier)
    plt.plot([0, 1], [sum(y_true) / len(y_true), sum(y_true) / len(y_true)],
             linestyle='--', color='gray', alpha=0.8,
             label=f'Random Classifier (AP={sum(y_true) / len(y_true):.3f})')

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves (Macro-Average AP={macro_ap:.3f})', fontsize=15)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    # Add text annotation
    plt.annotate('The closer the curve is to the\nupper right corner,\nthe better the model performs',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
                 fontsize=9, ha='right')

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    # Return average precision metrics for further use
    return {
        'average_precision_per_class': {i: ap for i, ap in enumerate(average_precisions)},
        'macro_average_precision': macro_ap
    }

def plot_feature_importance(feature_importance, feature_names=None, top_n=None, save_path=None):
    """
    绘制特征重要性图

    参数:
        feature_importance: 特征重要性数组
        feature_names: 特征名称列表
        top_n: 显示前n个最重要特征
        save_path: 保存路径
    """
    # 如果没有提供特征名称，则使用索引
    if feature_names is None:
        feature_names = [f'feature {i}' for i in range(len(feature_importance))]

    # 创建特征重要性和名称的数据框
    importance_data = list(zip(feature_names, feature_importance))
    importance_data.sort(key=lambda x: x[1], reverse=True)

    # 选择前top_n个特征
    if top_n is not None and top_n < len(importance_data):
        importance_data = importance_data[:top_n]

    # 解压数据
    names, values = zip(*importance_data)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), values, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance')
    plt.title('Feature importance')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_distribution(predictions, true_labels, class_names=None, save_path=None):
    """
    绘制预测分布图

    参数:
        predictions: 预测类别
        true_labels: 真实类别
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))

    # 计算每个类别的预测和真实分布
    unique_classes = np.unique(np.concatenate([predictions, true_labels]))
    pred_counts = np.bincount(predictions, minlength=len(unique_classes))
    true_counts = np.bincount(true_labels, minlength=len(unique_classes))

    # 如果有类别名称，则使用类别名称作为x轴标签
    if class_names and len(class_names) >= len(unique_classes):
        x_labels = [class_names[i] for i in unique_classes]
    else:
        x_labels = [f'class {i}' for i in unique_classes]

    x = np.arange(len(unique_classes))
    width = 0.35

    plt.bar(x - width / 2, true_counts, width, label='真实')
    plt.bar(x + width / 2, pred_counts, width, label='预测')

    plt.xlabel('class')
    plt.ylabel('Number of samples')
    plt.title('True and predicted distributions for each class')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_embedding_visualization(features, labels, class_names=None, method='tsne', save_path=None):
    """
    使用降维技术可视化高维特征

    参数:
        features: 特征向量，形状为(n_samples, n_features)
        labels: 标签
        class_names: 类别名称列表
        method: 降维方法，'tsne'或'umap'
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))

    # 使用t-SNE降维
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE feature visualization'
    # 使用UMAP降维
    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
        title = 'UMAP feature visualization'
    else:
        raise ValueError(f"Unsupported dimensionality reduction methods: {method}")

    # 降维到2维
    embedded = reducer.fit_transform(features)

    # 获取唯一类别
    unique_labels = np.unique(labels)

    # 为每个类别绘制散点图
    for i in unique_labels:
        indices = labels == i
        label_name = class_names[i] if class_names and i < len(class_names) else f"class {i}"
        plt.scatter(embedded[indices, 0], embedded[indices, 1], label=label_name, alpha=0.7)

    plt.title(title, fontsize=15)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_confidence(probabilities, true_labels, class_names=None, save_path=None):
    """
    绘制预测置信度分布

    参数:
        probabilities: 预测概率，形状为(n_samples, n_classes)
        true_labels: 真实标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    # 提取每个样本的最大概率值
    max_probs = np.max(probabilities, axis=1)
    # 提取预测的类别
    predictions = np.argmax(probabilities, axis=1)
    # 判断预测是否正确
    is_correct = (predictions == true_labels)

    plt.figure(figsize=(10, 6))

    # 分别绘制正确和错误预测的置信度分布
    plt.hist(max_probs[is_correct], bins=20, alpha=0.7, label='Correct Predictions', color='green')
    plt.hist(max_probs[~is_correct], bins=20, alpha=0.7, label='Incorrect Predictions', color='red')

    plt.xlabel('Prediction Confidence')
    plt.ylabel('Sample Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_class_wise_metrics(results, class_names=None, save_path=None):
    """
    绘制每个类别的性能指标

    参数:
        results: 测试结果字典，包含'per_class_accuracy'和其他指标
        class_names: 类别名称列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn.metrics import precision_recall_fscore_support

    # 获取类别标签
    unique_classes = sorted(results['per_class_accuracy'].keys())

    # 如果没有提供类别名称，则使用索引
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]
    else:
        class_names = [class_names[i] if i < len(class_names) else f'Class {i}' for i in unique_classes]

    # 从per_class_accuracy获取准确率
    accuracies = [results['per_class_accuracy'][cls] for cls in unique_classes]

    # 从classification report中提取精确度、召回率和F1分数
    # 计算每个类别的precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        results['true_labels'],
        results['predictions'],
        labels=unique_classes,
        average=None
    )

    # 创建指标字典
    metrics = {
        'accuracy': accuracies,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # 每个指标一个子图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 6))

    if n_metrics == 1:
        axes = [axes]

    # 绘制每个指标
    for i, (metric_name, values) in enumerate(metrics.items()):
        axes[i].bar(class_names, values, color='skyblue')
        axes[i].set_title(f'{metric_name.capitalize()} by Class')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(metric_name.capitalize())
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, linestyle='--', alpha=0.7)

        # 在每个条形上方显示数值
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f'{v:.2f}', ha='center')

        # 旋转x轴标签以避免重叠
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_rate(learning_rates, save_path=None):
    """
    绘制学习率变化曲线

    参数:
        learning_rates: 学习率列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # 对数尺度更好地显示学习率变化
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_maps(attention_maps, slice_indices=None, title="Attention Map", save_path=None):
    """
    绘制3D注意力图的切片可视化

    参数:
        attention_maps: 3D注意力图，形状为(D, H, W)
        slice_indices: 要显示的切片索引，默认为中心切片
        title: 图表标题
        save_path: 保存路径
    """
    D, H, W = attention_maps.shape

    # 如果没有指定切片索引，则使用中心切片
    if slice_indices is None:
        slice_indices = {
            'axial': D // 2,
            'coronal': H // 2,
            'sagittal': W // 2
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 轴向切片 (Axial)
    axial_slice = attention_maps[slice_indices['axial'], :, :]
    im1 = axes[0].imshow(axial_slice, cmap='hot')
    axes[0].set_title(f'Axial Slice - Depth Index: {slice_indices["axial"]}')
    axes[0].axis('off')

    # 冠状切片 (Coronal)
    coronal_slice = attention_maps[:, slice_indices['coronal'], :]
    im2 = axes[1].imshow(coronal_slice, cmap='hot')
    axes[1].set_title(f'Coronal Slice - Height Index: {slice_indices["coronal"]}')
    axes[1].axis('off')

    # 矢状切片 (Sagittal)
    sagittal_slice = attention_maps[:, :, slice_indices['sagittal']]
    im3 = axes[2].imshow(sagittal_slice, cmap='hot')
    axes[2].set_title(f'Sagittal Slice - Width Index: {slice_indices["sagittal"]}')
    axes[2].axis('off')

    # 添加颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_landscape_2d(model, loss_fn, data_loader, device, parameter_x, parameter_y,
                           x_range=(-1, 1), y_range=(-1, 1), n_points=20, save_path=None):
    """
    绘制2D损失景观

    参数:
        model: 模型
        loss_fn: 损失函数
        data_loader: 数据加载器
        device: 设备
        parameter_x: 要可视化的第一个参数名称
        parameter_y: 要可视化的第二个参数名称
        x_range: x轴范围
        y_range: y轴范围
        n_points: 每个维度的点数
        save_path: 保存路径
    """
    # 保存原始参数
    original_params = {}
    for name, param in model.named_parameters():
        if name == parameter_x or name == parameter_y:
            original_params[name] = param.data.clone()

    # 生成网格点
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 计算每个网格点的损失
    model.eval()
    with torch.no_grad():
        # 获取一批数据用于可视化
        batch = next(iter(data_loader))
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        for i in range(n_points):
            for j in range(n_points):
                # 修改参数
                for name, param in model.named_parameters():
                    if name == parameter_x:
                        param.data = original_params[name] * (1 + X[i, j])
                    elif name == parameter_y:
                        param.data = original_params[name] * (1 + Y[i, j])

                # 计算损失
                outputs, _ = model(inputs)  # 假设模型输出包含特征
                loss = loss_fn(outputs, targets)
                Z[i, j] = loss.item()

        # 恢复原始参数
        for name, param in model.named_parameters():
            if name in original_params:
                param.data = original_params[name]

    # 绘制3D曲面
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel(f'{parameter_x} Change')
    ax.set_ylabel(f'{parameter_y} Change')
    ax.set_zlabel('Loss')
    ax.set_title('2D Loss Landscape')

    # 添加颜色条
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()