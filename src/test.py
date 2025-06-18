import time
import torch
import pandas as pd
import numpy as np
import os

from monai.data import DataLoader
from tqdm import tqdm

# 导入自定义模块
from models import create_transforms
from models import ViT3D
from models import NoduleDataset
from one_epoch_train import one_epoch_train
from models import build_loss
from utils import calculate_metrics
from utils.Visualizer import *
from utils import parse_args, ConfigManager, set_seed
from train import save_metrics


def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config_test.yaml')

    # 解析命令行参数
    args = parse_args(config)

    # 添加错误样本相关参数
    args.save_errors = True  # 是否保存错误预测

    set_seed()

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=4,
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
        "pool": args.pool,
        'cpt_num': args.cpt_num,
        'mlp_num': args.mlp_num,
    }

    # 初始化模型
    print("初始化模型...")
    model = ViT3D(**model_config).to(device)

    # 加载模型权重
    print(f"从 {args.pretrained_path} 加载模型权重...")
    try:
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
    except Exception as e:
        print(f"加载预训练模型时出错: {str(e)}")

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    fn = build_loss(args.loss1, "CrossEntropyLoss", config)
    loss = fn if fn else build_loss(True, "FocalLoss", config)
    loss2 = build_loss(args.loss2, "BoundaryFlowLoss", config)
    loss3 = args.loss3

    # 测试模型
    loss, metrics, similarity_loss = one_epoch_train(model, test_loader, None, loss, device, train=False, loss2=loss2,
                                                     loss3=loss3)
    # 使用配置文件中指定的阈值方法
    test_metrics = calculate_metrics(metrics, config.data['num_classes'], threshold_method=config.model['threshold'])

    # 创建输出目录
    existing_dirs = [d for d in os.listdir(args.output_dir) if
                     d.startswith('test_') and os.path.isdir(os.path.join(args.output_dir, d))]
    next_num = 1
    if existing_dirs:
        existing_nums = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        if existing_nums:
            next_num = max(existing_nums) + 1
    output_dir = os.path.join(args.output_dir, f'test_{next_num}')
    os.makedirs(output_dir, exist_ok=True)

    # 打印测试结果
    print("\n测试结果:")
    print(f' Test Loss: {loss:.4f}, ACC: {test_metrics["accuracy"]:.4f} AUC: {test_metrics["auc"]:.4f}')
    print(
        f'   test   F1: {test_metrics["f1"]:.4f}, 敏感性: {test_metrics["sensitivity"]:.4f}, 特异性: {test_metrics["specificity"]:.4f}')
    print(f'阈值: {test_metrics["threshold"]:.4f}')
    print(f'PPV: {test_metrics["ppv"]:.4f}, NPV: {test_metrics["npv"]:.4f}')

    print("\n混淆矩阵:")  # tn, fp, fn, tp
    print(test_metrics['tn'], test_metrics['fp'], "\n", test_metrics['fn'], test_metrics['tp'])

    # 保存详细报告
    with open(os.path.join(output_dir, 'test_report.txt'), 'w', encoding='UTF-8') as f:
        f.write("测试结果:\n")
        f.write(f"准确率 (Accuracy): {test_metrics['accuracy']:.4f}\n")
        f.write(f"F1:: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n\n")
        f.write(f'敏感性: {test_metrics["sensitivity"]:.4f}, 特异性: {test_metrics["specificity"]:.4f}\n')
        f.write(f'PPV: {test_metrics["ppv"]:.4f}, NPV: {test_metrics["npv"]:.4f}')

        f.write("\n混淆矩阵:\n")
        f.write(f"[{test_metrics['tn']}, {test_metrics['fp']}]\n")
        f.write(f"[{test_metrics['fn']}, {test_metrics['tp']}]\n")

    save_metrics(test_metrics, os.path.join(output_dir, 'test_metrics.csv'))

    # 绘制各种性能曲线
    plot_confusion_matrix(
        metrics_dict=test_metrics,
        save_path=os.path.join(output_dir, f'confusion_matrix.png')
    )
    plot_roc_curve(
        metrics_dict_list=test_metrics,
        save_path=os.path.join(output_dir, f'ROC_Curve.png')
    )
    plot_pr_curve(
        metrics_dict_list=test_metrics,
        save_path=os.path.join(output_dir, f'PR_Curve.png')
    )

    plot_decision_curve_analysis(
        metrics_dict_list=test_metrics,
        save_path=os.path.join(output_dir, f'Decision Curve Analysis.png')
    )

    # 收集和保存错误预测的样本
    if args.save_errors:
        print("\n收集错误预测的样本...")
        model.eval()

        # 错误预测列表
        error_predictions = []
        all_predictions = []

        # 进度条
        test_iter = tqdm(test_loader, desc="评估样本中", leave=False)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_iter):
                # 获取图像和标签 - 根据修改后的数据集格式调整
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

                # 获取文件路径 - 从测试数据集中获取
                file_paths = []
                start_idx = batch_idx * test_loader.batch_size
                for i in range(len(inputs)):
                    sample_idx = start_idx + i
                    if sample_idx < len(test_dataset):
                        file_paths.append(test_dataset.samples[sample_idx][0])
                    else:
                        file_paths.append("unknown")

                # 前向传播
                outputs, *_ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # 使用配置的阈值进行预测（对于二分类任务）
                if config.data['num_classes'] == 2 and hasattr(test_metrics, 'threshold'):
                    bin_preds = (probs[:, 1] >= test_metrics["threshold"]).long()
                else:
                    bin_preds = predicted

                # 收集所有预测结果
                for i, (pred, bin_pred, label, path) in enumerate(zip(predicted, bin_preds, labels, file_paths)):
                    prediction_info = {
                        "path": path,
                        "true_label": label.item(),
                        "pred_label": bin_pred.item(),  # 使用基于阈值的预测
                        "raw_pred": pred.item(),  # 保存原始类别预测
                        "confidence": probs[i, pred].item(),
                        "probability_0": probs[i, 0].item() if probs.shape[1] > 0 else 0,
                        "probability_1": probs[i, 1].item() if probs.shape[1] > 1 else 0
                    }

                    all_predictions.append(prediction_info)

                    # 检查是否为错误预测
                    if bin_pred != label:
                        error_predictions.append(prediction_info)

        # 保存错误预测结果
        error_file_path = os.path.join(output_dir, 'error_predictions.csv')
        with open(error_file_path, 'w', encoding='utf-8') as f:
            f.write("样本路径,真实标签,预测标签,原始预测,预测置信度,类别0概率,类别1概率\n")
            for err in error_predictions:
                f.write(
                    f"{err['path']},{err['true_label']},{err['pred_label']},{err['raw_pred']},{err['confidence']:.4f},{err['probability_0']:.4f},{err['probability_1']:.4f}\n")

        # 保存所有预测结果
        all_preds_file_path = os.path.join(output_dir, 'all_predictions.csv')
        with open(all_preds_file_path, 'w', encoding='utf-8') as f:
            f.write("样本路径,真实标签,预测标签,原始预测,预测置信度,类别0概率,类别1概率\n")
            for pred in all_predictions:
                f.write(
                    f"{pred['path']},{pred['true_label']},{pred['pred_label']},{pred['raw_pred']},{pred['confidence']:.4f},{pred['probability_0']:.4f},{pred['probability_1']:.4f}\n")

        # 错误样本分析
        if len(error_predictions) > 0:
            # 按真实标签分组
            error_by_class = {}
            for err in error_predictions:
                label = err["true_label"]
                if label not in error_by_class:
                    error_by_class[label] = []
                error_by_class[label].append(err)

            # 保存分析结果
            with open(os.path.join(output_dir, 'error_analysis.txt'), 'w', encoding='utf-8') as f:
                f.write("错误预测分析\n\n")
                f.write(f"总样本数: {len(all_predictions)}\n")
                f.write(
                    f"总错误数: {len(error_predictions)} ({len(error_predictions) / len(all_predictions) * 100:.2f}%)\n\n")

                for class_label in sorted(error_by_class.keys()):
                    errors = error_by_class[class_label]
                    class_samples = sum(1 for p in all_predictions if p["true_label"] == class_label)
                    error_rate = len(errors) / class_samples * 100 if class_samples > 0 else 0
                    f.write(f"类别 {class_label}: {len(errors)}/{class_samples} 错误 ({error_rate:.2f}%)\n")

        print(f"\n错误预测的样本数量: {len(error_predictions)}")
        print(f"错误预测详情已保存到: {error_file_path}")
        print(f"所有预测结果已保存到: {all_preds_file_path}")
        print(f"错误分析已保存到: {os.path.join(output_dir, 'error_analysis.txt')}")

    # 打印总运行时间
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n总测试时间: {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"详细结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()