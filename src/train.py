import os
import time
import torch
from torch import optim
from tqdm import tqdm
from models import WarmupScheduler
from models import build_loss
from one_epoch_train import one_epoch_train
from utils import save_training_history, save_config, calculate_metrics
from utils.Visualizer import *

def save_best_model_if_improved(model, metric_value, metric_name, train_dir, epoch, flod):
    """如果指标改进，保存最佳模型"""
    print(f"\n保存新的最佳模型，{metric_name}: {metric_value:.4f}，在第{epoch + 1}轮")
    torch.save(model.state_dict(), f'{train_dir}/best_{metric_name}_model_{flod}.pth')


def save_metrics(metrics_dict, file_path):
    """
    保存指标字典到CSV文件，只保存标量值

    Args:
        metrics_dict: 包含指标的字典
        file_path: 保存CSV文件的路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 筛选出标量值（非数组）
    # scalar_metrics = {}
    # for key, value in metrics_dict.items():
    #     if not isinstance(value, (np.ndarray, list)) or np.isscalar(value):
    #         scalar_metrics[key] = value

    # 创建DataFrame
    df = pd.DataFrame([metrics_dict])

    # 保存为CSV
    df.to_csv(file_path, index=False, float_format='%.6f')
    print(f"指标已保存至: {file_path}")

    return metrics_dict


def train(model, train_loader, val_loader, config, fold, device, args,
          warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0, train_dir=None, best_f1=0, best_auc=0):
    """训练单个折的模型，添加学习率预热和梯度裁剪"""
    config_save_path = os.path.join(train_dir, 'config.yaml')
    save_config(config, config_save_path)

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
    fn = build_loss(args.loss1, "CrossEntropyLoss", config)
    loss = fn if fn else build_loss(True, "FocalLoss", config)
    loss2 = build_loss(args.loss2, "BoundaryFlowLoss", config)
    loss3 = args.loss3
    print("启用损失函数：", loss, loss2)
    if loss3:
        print("启用类提示向量相似性损失")

    # 初始化混合精度训练
    use_amp = device.type == 'cuda' and args.use_amp
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(
            init_scale = 2. ** 16,  # 初始缩放因子
            growth_factor = 2.0,  # 成功更新后的增长率
            backoff_factor = 0.5,  # 梯度溢出时的回退因子
            growth_interval = 100  # 连续成功更新多少次后增长缩放因子
        )
        print("启用混合精度训练")
    # 缓存验证集，优化评估过程
    print("缓存验证数据到GPU内存以加速评估...")

    val_data_cached = []
    for x, y in tqdm(val_loader, desc="缓存验证数据"):
        val_data_cached.append((x.to(device), y.to(device)))
    val_loader_cached = val_data_cached

    # 初始化指标记录列表
    train_metrics_history = []
    val_metrics_history = []
    # 初始化最优指标
    best_train_metrics = {"f1": 0.0, "auc": 0.0}
    best_val_metrics_f1 = {"f1": 0.0, "auc": 0.0}
    best_val_metrics_auc = {"f1": 0.0, "auc": 0.0}

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_metrics, similarity_loss = one_epoch_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss,
            loss2=loss2,
            loss3=loss3,
            loss1_weight=args.loss1_weight,
            loss2_weight=args.loss2_weight,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm
        )
        # 验证阶段
        val_loss, val_metrics, similarity_loss = one_epoch_train(
            model=model,
            data_loader=val_loader_cached,
            optimizer=optimizer,
            loss_fn=loss,
            loss2=loss2,
            loss3=loss3,
            loss1_weight=args.loss1_weight,
            loss2_weight=args.loss2_weight,
            device=device,
            train=False,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm
        )
        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印预热信息
        if epoch == warmup_epochs - 1:
            print("\n预热阶段结束，切换到基础学习率调度器")

        # 计算每轮训练时间
        epoch_time = time.time() - start_time

        train_metrics = calculate_metrics(train_metrics, config.data['num_classes'])
        val_metrics = calculate_metrics(val_metrics, config.data['num_classes'])

        if val_metrics['f1'] > best_val_metrics_f1['f1']:
            best_val_metrics_f1 = val_metrics.copy()
            save_best_model_if_improved(model, val_metrics['f1'], "f1", train_dir, epoch, fold)

        if train_metrics['f1'] > best_train_metrics['f1']:
            best_train_metrics = train_metrics.copy()

        if val_metrics['auc'] > best_val_metrics_auc['auc']:
            best_val_metrics_auc = val_metrics.copy()
            save_best_model_if_improved(model, val_metrics['auc'], "auc", train_dir, epoch, fold)

        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # 打印进度和指标
        print(
            f'\nFold {fold + 1}, Epoch {epoch + 1}/{config.training["num_epochs"]} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(
            f' Train Loss: {train_loss:.4f}, ACC: {train_metrics["accuracy"]:.4f} AUC: {train_metrics["auc"]:.4f}')
        print(
            f'   Val Loss: {val_loss:.4f}, ACC: {val_metrics["accuracy"]:.4f} AUC: {val_metrics["auc"]:.4f}')
        print(
            f'   Val   F1: {val_metrics["f1"]:.4f}, 敏感性: {val_metrics["sensitivity"]:.4f}, 特异性: {val_metrics["specificity"]:.4f}')
        print(f'Best Val F1: {best_val_metrics_f1["f1"]:.4f}, Best Val AUC: {best_val_metrics_auc["auc"]:.4f}')
        print(f'最佳阈值: {val_metrics["threshold"]:.4f}')
        print(f'PPV: {val_metrics["ppv"]:.4f}, NPV: {val_metrics["npv"]:.4f}')

        # 更新学习率调度器
        scheduler.step()

    if best_val_metrics_f1['f1'] > 0 and best_val_metrics_auc['auc'] > 0:
        # F1最佳模型的图表
        plot_confusion_matrix(
            metrics_dict=best_val_metrics_f1,
            save_path=os.path.join(train_dir, f'confusion_matrix_{fold}.png')
        )
        plot_roc_curve(
            metrics_dict_list=[best_val_metrics_f1, best_val_metrics_auc],
            save_path=os.path.join(train_dir, f'ROC_Curve_{fold}.png')
        )
        plot_pr_curve(
            metrics_dict_list=[best_val_metrics_f1, best_val_metrics_auc],
            save_path=os.path.join(train_dir, f'PR_Curve_{fold}.png')
        )

        plot_decision_curve_analysis(
            metrics_dict_list=[best_val_metrics_f1, best_val_metrics_auc],
            save_path=os.path.join(train_dir, f'decision_curve_{fold}.png')
        )
        plot_prediction_distribution(best_val_metrics_f1,
            save_path=os.path.join(train_dir, f'Prediction_Probability_Distribution_{fold}.png')
        )
        plot_calibration_curves([best_val_metrics_f1, best_val_metrics_auc],
            save_path=os.path.join(train_dir, f'calibration_curve_{fold}.png')
        )

        #保存F1最佳模型的指标到文件
        best_metrics_file = os.path.join(train_dir, f'best_metrics')
        os.makedirs(best_metrics_file, exist_ok=True)

        save_metrics(best_val_metrics_f1, best_metrics_file + f'/best_val_metrics_f1_fold_{fold}.csv')
        save_metrics(best_val_metrics_auc, best_metrics_file + f'/best_val_metrics_auc_fold_{fold}.csv')
        save_metrics(best_train_metrics, best_metrics_file + f'/best_train_metrics_fold_{fold}.csv')

    plot_learning_curves(
        train_metrics=train_metrics_history,
        val_metrics=val_metrics_history,
        save_path=os.path.join(train_dir, f'learning_curves_{fold}.png'),
    )

    # 保存历史训练记录
    save_training_history(train_dir,fold,train_metrics_history, val_metrics_history)
    # 返回最佳指标
    return best_val_metrics_f1['f1'], best_val_metrics_auc['auc']
