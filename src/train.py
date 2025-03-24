"""
主训练脚本
负责模型训练、验证和性能评估的核心逻辑
"""
import os
import time
import numpy as np
import torch
import torch.cuda.amp as amp  # 添加混合精度训练
from ruamel.yaml import YAML
from torch import optim
from tqdm import tqdm
from models import WarmupScheduler

# 导入自定义模块
from models import create_transforms
from models import CrossValidator
from models import ViT3D
from models import NoduleDataset, RecurrenceDataset
from models import FocalLoss, Enhanced3DVITLoss, BoundaryFlowLoss, build_loss
from one_epoch_train import one_epoch_train
from utils import save_training_history, update_config_from_args, parse_args, save_config, ConfigManager


def train_model(model, train_loader, val_loader, config, fold, device, args,
                warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0):
    """训练单个折的模型，添加学习率预热和梯度裁剪"""
    # 创建output_dir结构
    os.makedirs(args.output_dir, exist_ok=True)
    # 确定下一个可用的train_i目录编号
    existing_dirs = [d for d in os.listdir(args.output_dir) if
                     d.startswith('train_') and os.path.isdir(os.path.join(args.output_dir, d))]
    next_num = 1
    if existing_dirs:
        existing_nums = [int(d.split('_')[1]) for d in existing_dirs if d.split('_')[1].isdigit()]
        if existing_nums:
            next_num = max(existing_nums) + 1

    # 创建当前训练的输出目录
    train_dir = os.path.join(args.output_dir, f'train_{next_num}')
    os.makedirs(train_dir, exist_ok=True)

    config_save_path = os.path.join(train_dir, 'config.yaml')
    save_config(config, config_save_path)

    # 初始化指标记录列表
    train_metrics_history = []
    val_metrics_history = []

    # 初始化优化器
    optimizer = getattr(optim, config.optimizer["name"])(
        model.parameters(), **config.get_optimizer_params(model)
    )

    # 初始化基础学习率调度器
    base_scheduler = getattr(optim.lr_scheduler, config.scheduler["name"])(
        optimizer, **config.get_scheduler_params()
    )

    # 包装预热调度器
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        base_scheduler=base_scheduler,
        warmup_type=warmup_type
    )

    print(f"启用学习率预热: {warmup_epochs} 个epoch, 类型: {warmup_type}")
    print(f"启用梯度裁剪，最大范数: {max_grad_norm}")

    # 打印训练集中每个类的数量
    print(f"\n------- 第 {fold + 1} 折训练集类别数量 -------")
    for cls, count in train_loader.dataset._data_counter.items() if hasattr(train_loader.dataset,
                                                                            '_data_counter') else []:
        print(f"类别 {cls}: {count} 个样本")
    print("-----------------------------------\n")

    # 损失函数
    # loss_fn = FocalLoss()
    loss = build_loss(args.loss1, "CrossEntropyLoss", config) if args.loss1 else build_loss(args.loss2, "FocalLoss", config)
    loss2 = None
    loss3 = build_loss(args.loss3, "BoundaryFlowLoss", config)

    print("启用损失函数：", loss, loss3)

    # 初始化混合精度训练
    scaler = amp.GradScaler()
    use_amp = device.type == 'cuda' and config.training["use_amp"]
    if use_amp:
        print("启用混合精度训练")

    # 初始化最优指标
    best_val_f1 = 0
    best_val_auc = 0
    best_model_path = os.path.join(train_dir, f'best_f1_model_fold_{fold}.pth')
    best_model_auc_path = os.path.join(train_dir, f'best_auc_model_fold_{fold}.pth')

    # 保存用于可视化的数据
    best_val_preds = None
    best_val_labels = None
    best_val_probs = None

    # 缓存验证集，优化评估过程
    print("缓存验证数据到GPU内存以加速评估...")
    val_data_cached = []
    for x, y in tqdm(val_loader, desc="缓存验证数据"):
        val_data_cached.append((x.to(device), y.to(device)))
    val_loader_cached = val_data_cached

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_metrics = one_epoch_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss,
            loss2=loss2,
            loss3=loss3,
            device=device,
            train=True,
            use_amp=use_amp,
            scaler=scaler,
            max_grad_norm=max_grad_norm
        )
        train_metrics_history.append(train_metrics)

        # 验证阶段
        val_loss, val_metrics = one_epoch_train(
            model=model,
            data_loader=val_loader_cached,
            optimizer=optimizer,
            loss_fn=loss,
            loss2=loss2,
            loss3=loss3,
            device=device,
            train=False,
            use_amp=use_amp,
            scaler=scaler,
            max_grad_norm=max_grad_norm
        )
        val_metrics_history.append(val_metrics)

        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印预热信息
        if epoch == warmup_epochs - 1:
            print("预热阶段结束，切换到基础学习率调度器")

        # 计算每轮训练时间
        epoch_time = time.time() - start_time

        # 将每个类的准确率压缩到一行字符串输出
        per_class_str = ", ".join([f"Class {cls}: {acc:.4f}" for cls, acc in val_metrics['per_class_accuracy'].items()])

        # 保存基于F1分数的最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print(f"保存新的最佳模型(F1)，F1: {val_metrics['f1']:.4f}")

            # 保存用于后续可视化的预测结果
            best_val_preds = val_metrics.get('predictions', None)
            best_val_labels = val_metrics.get('labels', None)
            best_val_probs = val_metrics.get('probabilities', None)

        # 保存基于AUC的最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), best_model_auc_path)
            print(f"保存新的最佳模型(AUC)，AUC: {val_metrics['auc']:.4f}")

        # 打印进度和指标
        print(
            f'Fold {fold + 1}, Epoch {epoch + 1}/{config.training["num_epochs"]} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(
            f' Train Loss: {train_loss:.4f},'        f'    ACC: {train_metrics["accuracy"]:.4f}   AUC: {train_metrics["auc"]:.4f}')
        print(
            f'   Val Loss: {val_loss:.4f},'          f'    ACC: {val_metrics["accuracy"]:.4f}   AUC: {val_metrics["auc"]:.4f}\n'
            f'   Val   F1: {val_metrics["f1"]:.4f}')
        print(f'Best Val F1: {best_val_f1:.4f},    Best Val     AUC: {best_val_auc:.4f}')
        print(f'Per Cls Acc: {per_class_str}\n')

        # 更新学习率调度器
        scheduler.step()

    # 保存训练历史记录
    history_path = save_training_history(
        train_dir,
        fold,
        train_metrics_history,
        val_metrics_history
    )

    # 使用外部可视化模块生成图表
    from utils import plot_confusion_matrix, plot_roc_curves
    # 如果有保存的最佳模型预测数据，绘制混淆矩阵和ROC曲线
    if best_val_preds is not None and best_val_labels is not None:
        # 绘制混淆矩阵
        plot_confusion_matrix(
            best_val_labels,
            best_val_preds,
            class_names=config.data['class_names'],  # 可以从args获取类别名称
            save_path=os.path.join(train_dir, f'confusion_matrix_fold_{fold}.png')
        )

        # 绘制ROC曲线
        if best_val_probs is not None:
            plot_roc_curves(
                best_val_labels,
                best_val_probs,
                num_classes=config.data['num_classes'],
                save_path=os.path.join(train_dir, f'roc_curves_fold_{fold}.png')
            )

    # 返回最佳指标
    return best_val_f1, best_val_auc

def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config.yaml')

    # 解析命令行参数
    args = parse_args(config)
    config = update_config_from_args(config, args)

    # 设置随机种子
    torch.manual_seed(config.training["random_seed"])
    np.random.seed(config.training["random_seed"])

    # 优化CUDA性能
    if torch.cuda.is_available():
        # 设置GPU优先模式为性能优先
        torch.backends.cudnn.benchmark = True
        # 确定性算法提高一致性
        torch.backends.cudnn.deterministic = True

        # 清除缓存
        torch.cuda.empty_cache()

    print(f"使用设备: {config.training['device']}")
    print(f"缓存数据集: {args.cache_dataset}")
    if args.cache_dir:
        print(f"使用磁盘缓存目录: {args.cache_dir}")

    # 创建MONAI转换管道
    train_transforms, val_transforms = create_transforms(config, args)

    # 创建数据集
    print("加载数据集...")
    dataset = NoduleDataset(args.data_dir, num_classes=args.num_classes, transform=None)
    # 显示数据集基本信息
    print(f"数据集样本总数: {len(dataset.samples)}")

    # 初始化交叉验证器，传入缓存目录
    cv = CrossValidator(dataset, config)
    device = config.training['device']

    # 存储每折的最佳F1, AUC分数
    fold_scores = []

    if args.crop_size:
        config.model["params"]["image_size"] = args.crop_size
        args.image_size = args.crop_size
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
        "pool": args.pool
    }

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
        model = ViT3D(**model_config).to(device)
        # 加载预训练权重
        if args.pretrained_path:
            print(f"Loading pretrained weights from {args.pretrained_path}...")
            try:
                model.load_pretrained_dino(args.pretrained_path)
            except Exception as e:
                print(f"加载预训练权重出错: {str(e)}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 训练模型
        best_f1, best_auc = train_model(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device, args=args, fold=fold)

        fold_scores.append((best_f1, best_auc))
        fold_time = time.time() - fold_start_time
        print(f'Fold {fold + 1} completed in {fold_time:.2f}s with Best F1: {best_f1:.4f} with Best AUC: {best_auc:.4f}')

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

    # 打印总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == '__main__':
    main()
