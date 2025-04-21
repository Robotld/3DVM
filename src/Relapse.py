"""
主训练脚本
负责模型训练、验证和性能评估的核心逻辑
"""
import os
import time
import numpy as np
import torch

# 导入自定义模块
from models import create_transforms
from models import CrossValidator
from models import ViT3D
from models import RecurrenceDataset

from utils import update_config_from_args, parse_args, ConfigManager, freeze_encoder_keep_prompts, set_seed
from train import train


def main():
    # 记录开始时间
    start_time = time.time()
    # 加载配置
    config = ConfigManager('config/config_relapse.yaml')

    # 解析命令行参数
    args = parse_args(config)
    config = update_config_from_args(config, args)

    set_seed()

    # 优化CUDA性能
    if torch.cuda.is_available():
        # 设置GPU优先模式为性能优先
        torch.backends.cudnn.benchmark = False
        # 确定性算法提高一致性
        torch.backends.cudnn.deterministic = True

        # 清除缓存
        torch.cuda.empty_cache()

    print(f"使用设备: {config.training['device']}")
    print(f"缓存数据集: {args.cache_dataset}")
    if args.cache_dir:
        print(f"使用磁盘缓存目录: {args.cache_dir}")

    # 创建输出目录（只创建一个总目录）
    os.makedirs(args.output_dir, exist_ok = True)
    # 创建单个训练输出目录
    train_dir = os.path.join(args.output_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(train_dir, exist_ok = True)

    # 创建MONAI转换管道
    train_transforms, val_transforms = create_transforms(config, args)

    # 创建数据集
    print("加载数据集...")
    dataset = RecurrenceDataset("../datasets/CTreport/协和预后随访复发与否.csv", args.data_dir, transform=None)
    # 显示数据集基本信息
    print(f"数据集样本总数: {len(dataset.samples)}")

    # 初始化交叉验证器，传入缓存目录
    cv = CrossValidator(dataset, config)
    device = config.training['device']

    # 存储每折的最佳F1, AUC分数
    fold_scores = []

    # 预加载模型配置，避免在循环中重复访问
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

    best_f1, best_auc = 0.0, 0.0
    # 进行交叉验证
    for fold, train_loader, val_loader, train_counter in cv.get_folds(train_transforms, val_transforms):
        # 记录每个fold的数据计数器，用于计算类别权重
        train_loader.dataset._data_counter = train_counter

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_start_time = time.time()
        print(f'\nTraining Fold {fold + 1}')

        # 初始化新的模型
        model = ViT3D(**model_config)
        # 加载预训练权重
        if args.pretrained_path:
            print(f"Loading pretrained weights from {args.pretrained_path}...")
            try:
                if os.path.isdir(args.pretrained_path):
                    model.load_pretrained_dino(args.pretrained_path)
                else:
                    # 加载时跳过不匹配的参数
                    model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=False)
                    print(f"成功加载权重: {args.pretrained_path}")
            except Exception as e:
                print(f"加载预训练权重出错: {str(e)}")
            print("随机初始化CLStoken")
            model.cls_token = torch.nn.Parameter(torch.randn(1, 1, model.embed_dim))

        model.to(device)
        if args.frozen:
            freeze_encoder_keep_prompts(model)

        # 打印可训练参数数量及名称
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"可训练参数总数: {total_params}")
        print(f"可训练参数列表: {trainable_params}")
        # 训练模型
        f1, auc = train(model=model,
                                                       train_loader=train_loader,
                                                       val_loader=val_loader,
                                                       config=config,
                                                       device=device,
                                                       args=args,
                                                       fold=fold,
                                                       train_dir=train_dir,
                                                       best_f1=best_f1,
                                                       best_auc=best_auc)
        fold_scores.append((f1, auc))
        fold_time = time.time() - fold_start_time
        print(
            f'Fold {fold + 1} completed in {fold_time:.2f}s with Best F1: {f1:.4f} with Best AUC: {auc:.4f}')

        # 释放模型内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_scores = np.array(fold_scores)
    # 打印最终结果
    print('\nCross-Validation Results:')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}')
    mean_f1 = np.mean(fold_scores[:, 0])
    std_f1 = np.std(fold_scores[:, 0])
    mean_auc = np.mean(fold_scores[:, 1])
    std_auc = np.std(fold_scores[:, 1])
    print(f'Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n'
          f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')

    cv_results_path = os.path.join(train_dir, 'cv_results.txt')
    with open(cv_results_path, 'w') as f:
        f.write("Cross-Validation Results:\n")
        for fold, score in enumerate(fold_scores):
            f.write(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}\n')
        f.write(f'\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}\n')
        f.write(f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')

    # 打印总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")


if __name__ == '__main__':
    main()
