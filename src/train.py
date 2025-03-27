"""
负责模型训练、计算指标，绘图、并保存最优模型
"""
import os
import time
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from models import WarmupScheduler

from models import build_loss
from one_epoch_train import one_epoch_train
from utils import save_training_history, save_config


def train(model, train_loader, val_loader, config, fold, device, args,
          warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0, train_dir=None, best_f1=0, best_auc=0):
    """训练单个折的模型，添加学习率预热和梯度裁剪"""
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
    use_amp = device.type == 'cuda' and args.use_amp
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(
            init_scale = 2. ** 16,  # 初始缩放因子
            growth_factor = 2.0,  # 成功更新后的增长率
            backoff_factor = 0.5,  # 梯度溢出时的回退因子
            growth_interval = 2000  # 连续成功更新多少次后增长缩放因子
        )
        print("启用混合精度训练")

    # 初始化最优指标
    best_val_f1 = 0
    best_val_auc = 0
    best_auc_model = None
    best_f1_model = None

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
            loss1_weight=args.loss1_weight,
            loss3_weight=args.loss3_weight,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
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
            loss1_weight=args.loss1_weight,
            loss3_weight=args.loss3_weight,
            device=device,
            train=False,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm
        )
        val_metrics_history.append(val_metrics)

        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印预热信息
        if epoch == warmup_epochs - 1:
            print("\n预热阶段结束，切换到基础学习率调度器")

        # 计算每轮训练时间
        epoch_time = time.time() - start_time

        # 将每个类的准确率压缩到一行字符串输出
        per_class_str = ", ".join([f"Class {cls}: {acc:.4f}" for cls, acc in val_metrics['per_class_accuracy'].items()])

        # 保存基于F1分数的最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            if best_f1 < best_val_f1:
                best_f1_model = model.state_dict()
            # 保存用于后续可视化的预测结果
            best_val_preds = val_metrics.get('predictions', None)
            best_val_labels = val_metrics.get('labels', None)
            best_val_probs = val_metrics.get('probabilities', None)

        # 保存基于AUC的最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            if best_auc < best_val_auc:
                best_auc_model = model.state_dict()
            print(f"\n保存新的最佳模型(AUC)，AUC: {val_metrics['auc']:.4f}")

        # 打印进度和指标
        print(
            f'\nFold {fold + 1}, Epoch {epoch + 1}/{config.training["num_epochs"]} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
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
    return best_val_f1, best_val_auc, best_f1_model, best_auc_model

