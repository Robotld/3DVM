"""
测试脚本
负责加载预训练模型并在测试集上进行评估
"""
import os
import time
import numpy as np
import torch
from ruamel.yaml import YAML
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.preprocessing import normalize

# 导入自定义模块
from models import create_transforms
from models import ViT3D
from models import NoduleDataset
from utils import parse_args, ConfigManager


def test_model(model, test_loader, device, class_names=None):
    """测试模型在测试集上的性能"""
    model.eval()  # 设置为评估模式

    all_preds = []
    all_labels = []
    all_probs = []

    print("开始测试...")
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in tqdm(test_loader, desc="测试进度"):
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs, features, flow = model(inputs)
            # 获取预测和概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 确保概率和为1
    all_probs = normalize(all_probs, norm='l1', axis=1)

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    num_classes = all_probs.shape[1]
    if num_classes > 2:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        auc = roc_auc_score(all_labels, all_probs[:, 1])

    # 计算每类准确率
    unique_classes = np.unique(all_labels)
    per_class_accuracy = {
        cls: np.mean(all_preds[all_labels == cls] == cls)
        for cls in unique_classes
    }

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 生成详细分类报告
    if class_names:
        cls_report = classification_report(all_labels, all_preds, target_names=class_names)
    else:
        cls_report = classification_report(all_labels, all_preds)

    # 返回测试结果
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'classification_report': cls_report,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

    return results


def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config_test.yaml')

    # 解析命令行参数
    args = parse_args(config)

    # 设置随机种子
    torch.manual_seed(config.training["random_seed"])
    np.random.seed(config.training["random_seed"])

    # 优化CUDA性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    print(f"使用设备: {config.training['device']}")
    device = config.training['device']

    # 创建测试数据转换
    _, test_transforms = create_transforms(config, args)

    # 加载测试数据集
    print("加载测试数据集...")
    test_dir = args.data_dir
    test_dataset = NoduleDataset(test_dir, num_classes=args.num_classes, transform=test_transforms)
    print(f"测试数据集样本数: {len(test_dataset)}")

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training["num_workers"],
        pin_memory=True
    )

    model_config = {
        "num_classes": config.data["num_classes"],
        "image_size": args.image_size,
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_dim": config.model["params"]["mlp_dim"],
        "pool": args.pool
    }

    # 初始化模型
    print("初始化模型...")
    model = ViT3D(**model_config).to(device)

    # 加载模型权重
    print(f"从 {args.pretrained_path} 加载模型权重...")
    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 测试模型
    results = test_model(model, test_loader, device, None)

    existing_dirs = [d for d in os.listdir(args.output_dir) if
                     d.startswith('test_') and os.path.isdir(os.path.join(args.output_dir, d))]
    next_num = 1
    if existing_dirs:
        existing_nums = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        if existing_nums:
            next_num = max(existing_nums) + 1
    output_dir = os.path.join(args.output_dir, f'test_{next_num}')
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打印测试结果
    print("\n测试结果:")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"F1 分数 (macro): {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("\n每类准确率:")
    for cls, acc in results['per_class_accuracy'].items():
        class_name = args.class_names[cls] if args.class_names and cls < len(args.class_names) else f"类别 {cls}"
        print(f"{class_name}: {acc:.4f}")

    print("\n分类报告:")
    print(results['classification_report'])

    print("\n混淆矩阵:")
    print(results['confusion_matrix'])

    # 保存详细报告
    with open(os.path.join(output_dir, 'test_report.txt'), 'w', encoding='UTF-8') as f:
        f.write("测试结果:\n")
        f.write(f"准确率 (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"F1 分数 (macro): {results['f1']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n\n")
        f.write("每类准确率:\n")
        for cls, acc in results['per_class_accuracy'].items():
            class_name = args.class_names[cls] if args.class_names and cls < len(args.class_names) else f"类别 {cls}"
            f.write(f"{class_name}: {acc:.4f}\n")
        f.write("\n分类报告:\n")
        f.write(results['classification_report'])
        f.write("\n混淆矩阵:\n")
        f.write(str(results['confusion_matrix']))

    from utils import plot_confusion_matrix, plot_prediction_confidence, plot_class_wise_metrics, plot_roc_curves
    plot_confusion_matrix(results['true_labels'],
                          results['predictions'],
                          class_names=args.class_names,
                          save_path=os.path.join(output_dir, 'confusion_matrix.png'))

    # 绘制预测置信度分布
    plot_prediction_confidence(results['probabilities'],
                               results['true_labels'],
                               class_names=args.class_names,
                               save_path=os.path.join(output_dir, 'prediction_confidence.png'))

    # 绘制每类性能
    plot_class_wise_metrics(results,
                            class_names=args.class_names,
                            save_path=os.path.join(output_dir, 'class_performance.png'))

    # 绘制ROC曲线
    plot_roc_curves(results['true_labels'],
                    results['probabilities'],
                    num_classes=config.data['num_classes'],
                    class_names=args.class_names,
                    save_path=os.path.join(output_dir, 'roc_curve.png'))

    # 打印总运行时间
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n总测试时间: {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"详细结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
