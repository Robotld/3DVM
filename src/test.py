"""
测试脚本
负责加载预训练模型并在测试集上进行评估
"""
import time
import torch

# 导入自定义模块
from models import create_transforms
from models import ViT3D
from models import NoduleDataset
from one_epoch_train import one_epoch_train
from models import build_loss
from utils import calculate_metrics
from utils.Visualizer import *
from utils import parse_args, ConfigManager, set_seed
from train import save_metrics_to_csv

def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config_test.yaml')

    # 解析命令行参数
    args = parse_args(config)

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
        "pool": args.pool,
        'cpt_num': args.cpt_num,
        'mlp_num': args.mlp_num,
    }

    # 初始化模型
    print("初始化模型...")
    model = ViT3D(**model_config).to(device)

    # 加载模型权重
    print(f"从 {args.pretrained_path} 加载模型权重...")
    model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict = False)

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    fn = build_loss(args.loss1, "CrossEntropyLoss", config)
    loss = fn if fn else build_loss(True, "FocalLoss", config)
    loss2 = build_loss(args.loss2, "BoundaryFlowLoss", config)
    loss3 = args.loss3
    # 测试模型
    loss, metrics, similarity_loss = one_epoch_train(model, test_loader, None, loss, device, train=False, loss2=loss2, loss3=loss3)
    test_metrics = calculate_metrics(metrics, config.data['num_classes'], threshold_method='0.5')
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
    # 打印进度和指标
    print(
        f' Test Loss: {loss:.4f}, ACC: {test_metrics["accuracy"]:.4f} AUC: {test_metrics["auc"]:.4f}')
    print(
        f'   test   F1: {test_metrics["f1"]:.4f}, 敏感性: {test_metrics["sensitivity"]:.4f}, 特异性: {test_metrics["specificity"]:.4f}')
    print(f'阈值: {test_metrics["threshold"]:.4f}')
    print(f'PPV: {test_metrics["ppv"]:.4f}, NPV: {test_metrics["npv"]:.4f}')

    print("\n混淆矩阵:") # tn, fp, fn, tp
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

    save_metrics_to_csv(test_metrics, os.path.join(output_dir, 'test_metrics.csv'))

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
    create_combined_performance_figure(
        metrics_dict_list=test_metrics,
        save_path=os.path.join(output_dir, f'combined_performance.tiff')
    )


    # 打印总运行时间
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\n总测试时间: {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"详细结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
