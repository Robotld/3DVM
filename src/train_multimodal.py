import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from models import WarmupScheduler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from models import build_loss, MultitaskLoss
from utils import save_training_history, save_config, calculate_metrics
from utils.Visualizer import *

def save_best_model_if_improved(model, metric_value, metric_name, train_dir, epoch, fold):
    """如果指标改进，保存最佳模型"""
    print(f"\n保存新的最佳模型，{metric_name}: {metric_value:.4f}，在第{epoch + 1}轮")
    save_path = f'{train_dir}/best_{metric_name}_model_{fold}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    # 返回保存路径，方便后续引用
    return save_path


def save_metrics_to_csv(metrics_dict, file_path):
    """
    保存指标字典到CSV文件，只保存标量值

    Args:
        metrics_dict: 包含指标的字典
        file_path: 保存CSV文件的路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # # 筛选出标量值（非数组）
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


def one_epoch_multimodal_train(model, data_loader, optimizer, criterion, device, train=True, scaler=None, use_amp=False, max_grad_norm=1.0):
    """
    训练或评估多模态模型一个epoch

    Args:
        model: 多模态模型
        data_loader: 数据加载器
        optimizer: 优化器
        criterion: 多任务损失函数
        device: 设备
        train: 是否为训练模式
        scaler: 混合精度缩放器
        use_amp: 是否使用混合精度
        max_grad_norm: 梯度裁剪最大范数

    Returns:
        total_loss: 平均总损失
        metrics_dict: 保存预测和标签的字典
        sim_loss: 相似度损失
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    recurrence_loss_total = 0.0
    subtype_loss_total = 0.0
    similarity_loss_total = 0.0

    # 收集所有预测和标签
    all_rec_probs = []
    all_rec_labels = []
    all_sub_preds = []
    all_sub_labels = []

    loader = tqdm(data_loader, desc=f"{'训练中' if train else '评估中'}")

    for batch_data in loader:
        # 解包多模态数据
        images = batch_data["image"].to(device)
        text_inputs1 = {k: v.to(device) for k, v in batch_data["text_inputs1"].items()}
        text_inputs2 = {k: v.to(device) for k, v in batch_data["text_inputs2"].items()}
        demographics = batch_data["demographic"].to(device)
        recurrence_labels = batch_data["recurrence_label"].to(device)
        subtype_labels = batch_data["subtype_label"].to(device)
        keywords = batch_data["keywords"]  # 获取关键词列表

        # for x, y in zip(ID, recurrence_labels):
        #     print(x, y)

        # 训练模式
        if train:
            optimizer.zero_grad()

            # 混合精度训练
            if use_amp:
                with torch.cuda.amp.autocast():
                    # 多模态前向传播
                    recurrence_logits, subtype_logits, similarity_loss, contrastive_loss, _ = model(images, text_inputs1, text_inputs2, demographics, keywords)
                    # 计算多任务损失
                    loss, rec_loss, sub_loss, sim_loss = criterion(
                        recurrence_logits, subtype_logits, recurrence_labels, subtype_labels, similarity_loss
                    )
                    loss += contrastive_loss

                # 反向传播与优化
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练流程
                recurrence_logits, subtype_logits, similarity_loss, contrastive_loss, _ = model(images, text_inputs1, text_inputs2, demographics, keywords)
                loss, rec_loss, sub_loss, sim_loss = criterion(
                    recurrence_logits, subtype_logits,  recurrence_labels, subtype_labels, similarity_loss
                )
                contrastive_loss += contrastive_loss
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        else:
            # 评估模式
            with torch.no_grad():
                recurrence_logits, subtype_logits, similarity_loss, contrastive_loss, _ = model(images, text_inputs1, text_inputs2, demographics, keywords)
                loss, rec_loss, sub_loss, sim_loss = criterion(
                    recurrence_logits, subtype_logits, recurrence_labels, subtype_labels, similarity_loss
                )
                loss += contrastive_loss

        # 累计损失
        total_loss += loss.item()
        recurrence_loss_total += rec_loss.item() if isinstance(rec_loss, torch.Tensor) else rec_loss
        subtype_loss_total += sub_loss.item()
        similarity_loss_total += similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else similarity_loss

        # 保存复发预测结果
        valid_rec_mask = (recurrence_labels != -1)
        if valid_rec_mask.sum() > 0:
            rec_probs = torch.softmax(recurrence_logits[valid_rec_mask], dim=1).detach().cpu().numpy()
            all_rec_probs.extend(rec_probs)
            all_rec_labels.extend(recurrence_labels[valid_rec_mask].cpu().numpy())

        # 保存亚型分类结果
        valid_sub_mask = (subtype_labels != -1)
        if valid_sub_mask.sum() > 0:
            sub_preds = torch.argmax(subtype_logits[valid_sub_mask], dim=1).cpu().numpy()
            all_sub_preds.extend(sub_preds)
            all_sub_labels.extend(subtype_labels[valid_sub_mask].cpu().numpy())

    # 计算平均损失
    total_loss /= len(data_loader)
    recurrence_loss_total /= len(data_loader)
    subtype_loss_total /= len(data_loader)
    similarity_loss_total /= len(data_loader)

    # 构建指标字典
    metrics_dict = {
        'probabilities': np.array(all_rec_probs),
        'sub_probabilities': np.array(all_sub_preds),
        'labels': np.array(all_rec_labels) if all_rec_labels else np.array([]),
        'subtype_predictions': np.array(all_sub_preds) if all_sub_preds else np.array([]),
        'subtype_labels': np.array(all_sub_labels) if all_sub_labels else np.array([])
    }


    # 收集所有损失值
    loss_dict = {
        'total_loss': total_loss,
        'recurrence_loss': recurrence_loss_total,
        'subtype_loss': subtype_loss_total,
        'similarity_loss': similarity_loss_total
    }

    return total_loss, metrics_dict, similarity_loss_total


def train(model, train_loader, val_loader, config, fold, device, args,
          warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0, train_dir=None, best_f1=0, best_auc=0):
    """训练多模态多任务模型，支持学习率预热和梯度裁剪"""
    # 创建训练目录并保存配置
    os.makedirs(train_dir, exist_ok=True)
    config_save_path = os.path.join(train_dir, 'config.yaml')
    save_config(config, config_save_path)

    # 初始化多任务损失函数
    criterion = MultitaskLoss(
        recurrence_weight=config.losses['MultitaskLoss']['recurrence_weight'],
        subtype_weight=config.losses['MultitaskLoss']['subtype_weight'],
        similarity_weight=config.losses.get('similarity_weight', 0.1)
    )

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

    # 初始化混合精度训练
    use_amp = device.type == 'cuda' and args.use_amp
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2. ** 16,  # 初始缩放因子
            growth_factor=2.0,    # 成功更新后的增长率
            backoff_factor=0.5,   # 梯度溢出时的回退因子
            growth_interval=2000  # 连续成功更新多少次后增长缩放因子
        )
        print("启用混合精度训练")

    # 缓存验证集，优化评估过程
    print("缓存验证数据到GPU内存以加速评估...")
    val_data_cached = []

    # 检查验证集大小，避免占用过多GPU内存
    val_size = len(val_loader.dataset)
    cache_threshold = getattr(args, 'cache_threshold', 10000)  # 默认阈值，可以通过参数调整
    should_cache = val_size <= cache_threshold or getattr(args, 'force_cache_val', False)

    if should_cache:
        try:
            # 尝试缓存验证数据
            for batch_data in tqdm(val_loader, desc="缓存验证数据"):
                # 将每个批次的数据移动到设备
                cached_batch = {
                    "image": batch_data["image"].to(device),
                    "text_inputs1": {k: v.to(device) for k, v in batch_data["text_inputs1"].items() if isinstance(v, torch.Tensor)},
                    "text_inputs2": {k: v.to(device) for k, v in batch_data["text_inputs2"].items() if
                                     isinstance(v, torch.Tensor)},
                    "report1":batch_data["report1"],
                    "demographic": batch_data["demographic"].to(device),
                    "recurrence_label": batch_data["recurrence_label"].to(device),
                    "subtype_label": batch_data["subtype_label"].to(device),
                    "keywords": batch_data["keywords"],
                }
                val_data_cached.append(cached_batch)
            print(f"已缓存全部 {val_size} 个验证样本")
        except RuntimeError as e:
            print(f"缓存验证数据时出错（可能是GPU内存不足）: {e}")
            print("将直接使用验证数据加载器")
            val_data_cached = val_loader
    else:
        print(f"验证集样本数 {val_size} 超过阈值 {cache_threshold}，跳过缓存以节省内存")
        val_data_cached = val_loader

    # 初始化指标记录列表和最优指标
    train_metrics_history = []
    val_metrics_history = []
    best_train_metrics = None
    best_val_metrics = None
    best_val_auc = 0  # 使用传入的初始值
    best_val_f1 = 0   # 使用传入的初始值
    best_train_f1 = 0
    best_f1_model_state = None
    best_auc_model_state = None

    # 提前停止的计数器
    patience = getattr(args, 'patience', 5)
    patience_counter = 0
    last_best_epoch = 0

    # 训练主循环
    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_metrics, train_sim_loss = one_epoch_multimodal_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm
        )

        # 验证阶段
        val_loss, val_metrics, val_sim_loss = one_epoch_multimodal_train(
            model=model,
            data_loader=val_data_cached,
            optimizer=optimizer,
            criterion=criterion,
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

        # 计算训练和验证指标
        val_metrics = calculate_metrics(val_metrics, config.data['num_classes'])
        train_metrics = calculate_metrics(train_metrics, config.data['num_classes'])

        # 检查F1分数是否提高
        improved = False
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics.copy()
            best_f1_model_state = model.state_dict().copy()
            # save_best_model_if_improved(model, val_metrics['f1'], "f1", train_dir, epoch, fold)
            last_best_epoch = epoch
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        # 检查训练集F1分数
        if train_metrics['f1'] > best_train_f1:
            best_train_f1 = train_metrics['f1']
            best_train_metrics = train_metrics.copy()

        # 检查AUC是否提高
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_auc_model_state = model.state_dict().copy()
            plot_roc_curve(val_metrics, figsize=(6, 7), save_path=os.path.join(train_dir, f'ROC_Curve_fold_{fold}.png'))
            # save_best_model_if_improved(model, val_metrics['auc'], "auc", train_dir, epoch, fold)
            if not improved:  # 如果F1没有提高但AUC提高了，也重置耐心计数器
                patience_counter = 0
                last_best_epoch = epoch
            improved = True
        # 记录历史指标
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # 保存当前epoch的检查点
        # checkpoint_path = os.path.join(train_dir, f'checkpoint_fold_{fold}_epoch_{epoch+1}.pth')
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        #     'val_metrics': val_metrics,
        #     'train_metrics': train_metrics,
        # }, checkpoint_path)

        # 打印进度和指标
        print(
            f'\nFold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(
            f' Train Loss: {train_loss:.4f}, ACC: {train_metrics["accuracy"]:.4f} AUC: {train_metrics["auc"]:.4f}')
        print(
            f'   Val Loss: {val_loss:.4f}, ACC: {val_metrics["accuracy"]:.4f} AUC: {val_metrics["auc"]:.4f}')
        print(
            f'   Val   F1: {val_metrics["f1"]:.4f}, 敏感性: {val_metrics["sensitivity"]:.4f}, 特异性: {val_metrics["specificity"]:.4f}')
        print(f'train F1:{train_metrics["f1"]:.4f}')
        print(f'Best Val F1: {best_val_f1:.4f}, Best Val AUC: {best_val_auc:.4f}')
        print(f'最佳阈值: {val_metrics["threshold"]:.4f}')
        print(f'PPV: {val_metrics["ppv"]:.4f}, NPV: {val_metrics["npv"]:.4f}')

        # 更新学习率调度器
        scheduler.step()

        # # 检查提前停止
        if patience_counter >= patience:
            print(f"\n连续 {patience} 个epoch未改善性能，提前停止训练")
            print(f"最后的最佳epoch是第 {last_best_epoch + 1} 轮")
            break

    if best_val_metrics:
        # F1最佳模型的图表
        plot_confusion_matrix(
            metrics_dict=best_val_metrics,
            save_path=os.path.join(train_dir, f'confusion_matrix_fold_{fold}.png')
        )
        plot_pr_curve(
            metrics_dict_list=best_val_metrics,
            save_path=os.path.join(train_dir, f'PR_Curve_fold_{fold}.png')
        )

        plot_decision_curve_analysis(
            metrics_dict_list=best_val_metrics,
            save_path=os.path.join(train_dir, f'decision_curve_{fold}.png')
        )
        plot_calibration_curves(
            metrics_dict_list=best_val_metrics,
            save_path=os.path.join(train_dir, f'calibration_{fold}.png')
        )
        # 保存F1最佳模型的指标到文件
        best_metrics_file = os.path.join(train_dir, f'best_metrics')
        os.makedirs(best_metrics_file, exist_ok=True)

        save_metrics_to_csv(best_val_metrics, best_metrics_file + f'/best_val_metrics_fold_{fold}.csv')
        save_metrics_to_csv(best_train_metrics, best_metrics_file + f'/best_train_metrics_fold_{fold}.csv')

    plot_learning_curves(
        train_metrics=train_metrics_history,
        val_metrics=val_metrics_history,
        save_path=os.path.join(train_dir, f'learning_curves_{fold}.png'),
    )
    # 保存历史训练记录
    save_training_history(train_dir, fold, train_metrics_history, val_metrics_history)

    # 返回最佳指标和模型状态
    return best_val_f1, best_val_auc, best_f1_model_state, best_auc_model_state