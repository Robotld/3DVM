"""
主训练脚本
负责模型训练、验证和性能评估的核心逻辑
"""

import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.cuda.amp as amp  # 添加混合精度训练
from ruamel.yaml import YAML
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch import optim
from tqdm import tqdm

from models import WarmupScheduler
# 导入自定义模块
from models import create_transforms
from models import CrossValidator
from models import ViT3D
from models import NoduleDataset
from models import FocalLoss
import os

class ConfigManager:
    def __init__(self, config_path):
        yaml = YAML()
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.load(f)

        # 处理设备配置
        self.config["training"]["device"] = torch.device(
            "cuda" if torch.cuda.is_available() and self.config["training"]["device"] == "cuda"
            else "cpu"
        )

    def __getattr__(self, name):
        return self.config.get(name)

    def get_optimizer_params(self, model):
        return {
            "lr": float(self.optimizer["params"]["lr"]),
            "weight_decay": self.optimizer["params"]["weight_decay"]
        }

    def get_scheduler_params(self):
        return self.scheduler["params"]

def train_model(model, train_loader, val_loader, config, fold, device,
                warmup_epochs=3, warmup_type='linear', max_grad_norm=1.0):
    """训练单个折的模型，添加学习率预热和梯度裁剪"""
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
    for cls, count in train_loader.dataset._data_counter.items() if hasattr(train_loader.dataset, '_data_counter') else []:
        print(f"类别 {cls}: {count} 个样本")
    print("-----------------------------------\n")

    # # 初始化损失函数
    # if config.training["loss"]["class_weights"] == "auto":
    #     # 从数据集中获取类别分布
    #     if hasattr(train_loader.dataset, '_data_counter'):
    #         class_counter = train_loader.dataset._data_counter
    #     else:
    #         # 统计训练集中的标签分布
    #         class_counter = Counter()
    #         for batch in train_loader:
    #             _, labels = batch
    #             class_counter.update(labels.cpu().numpy())
    #
    #     # 计算类别权重
    #     classes = sorted(class_counter.keys())
    #     class_counts = [class_counter[cls] for cls in classes]
    #     class_weights = np.array([1.0 / (count if count > 0 else 1) for count in class_counts])
    #     class_weights = class_weights / class_weights.sum() * len(class_weights)
    #     class_weights = torch.FloatTensor(class_weights).to(device)
    # else:
    #     class_weights = None
    #
    # print(f"类别权重: {class_weights}")
    # loss_fn = LabelSmoothingCrossEntropy(
    #     smoothing=config.training["loss"]["smoothing"],
    #     class_weights=class_weights
    # )

    loss_fn = FocalLoss()

    # 初始化混合精度训练
    scaler = amp.GradScaler()
    use_amp = device.type == 'cuda' and config.training["use_amp"]
    if use_amp:
        print("启用混合精度训练")

    best_val_f1 = 0
    best_model_path = f'best_model_fold_{fold}.pth'



    #缓存验证集，优化评估过程
    print("缓存验证数据到GPU内存以加速评估...")
    val_data_cached = []
    for x, y in tqdm(val_loader, desc="缓存验证数据"):
        val_data_cached.append((x.to(device), y.to(device)))

    for epoch in range(config.training["num_epochs"]):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # 使用tqdm显示进度条
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} 训练", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            if use_amp:
                # 使用自动混合精度
                with amp.autocast():
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                # 混合精度缩放梯度
                scaler.scale(loss).backward()

                # 实施梯度裁剪
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss = loss_fn(outputs, y)
                loss.backward()

                # 实施梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for x, y in tqdm(val_data_cached, desc=f"Epoch {epoch+1} 验证", leave=False):  # 使用缓存的验证数据
                if use_amp:
                    with amp.autocast():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)
                else:
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                val_loss += loss.item()

                # 预测类别
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

                # 获取预测概率
                probs = torch.softmax(outputs, dim=1)
                val_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组，优化后续计算
        val_preds_np = np.array(val_preds)
        val_labels_np = np.array(val_labels)
        val_probs_np = np.array(val_probs)

        # 计算整体指标
        val_acc = accuracy_score(val_labels_np, val_preds_np)
        val_f1 = f1_score(val_labels_np, val_preds_np, average='macro')

        # 处理AUC计算 - 根据类别数量自适应
        num_classes = len(np.unique(val_labels_np))
        if num_classes > 2:
            # 多分类
            val_auc = roc_auc_score(val_labels_np, val_probs_np, multi_class="ovr", average="macro")
        else:
            # 二分类
            if val_probs_np.shape[1] > 1:
                val_auc = roc_auc_score(val_labels_np, val_probs_np[:, 1])
            else:
                val_auc = roc_auc_score(val_labels_np, val_preds_np)

        # 计算每个类的准确率 - 使用numpy操作而非循环
        per_class_accuracy = {}
        unique_classes = np.unique(val_labels_np)
        for cls in unique_classes:
            mask = (val_labels_np == cls)
            correct = np.sum((val_preds_np == cls) & mask)
            per_class_accuracy[cls] = correct / np.sum(mask) if np.sum(mask) > 0 else 0

        # 将每个类的准确率压缩到一行字符串输出
        per_class_str = ", ".join([f"Class {cls}: {acc:.4f}" for cls, acc in per_class_accuracy.items()])

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # 使用torch.save的_use_new_zipfile_serialization=True选项加快保存速度
            torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=True)
            print(f"保存新的最佳模型，F1: {val_f1:.4f}")

        # 计算当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印预热信息
        if epoch == warmup_epochs - 1:
            print("预热阶段结束，切换到基础学习率调度器")

        # 计算每轮训练时间
        epoch_time = time.time() - start_time

        # 打印进度和指标
        print(
            f'Fold {fold + 1}, Epoch {epoch + 1}/{config.training["num_epochs"]} - 耗时: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_correct / train_total:.4f}')
        print(f'Val Loss: {val_loss / len(val_data_cached):.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}, '
              f'Val AUC: {val_auc:.4f}')
        print(f'Per Class Accuracies: {per_class_str}')

        # 更新学习率调度器
        scheduler.step()

    return best_val_f1

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, nargs='+', default=config.data["root_dirs"],
                        help='Override data directory in config')
    parser.add_argument('--batch_size', type=int, default=config.training["batch_size"],
                        help='Override batch size in config')
    parser.add_argument('--use_amp', action='store_true', default=config.training["use_amp"],
                        help='Use automatic mixed precision training')
    parser.add_argument('--cache_dataset', action='store_true', default=config.training["cache_dataset"],
                        help='Cache entire dataset in memory (speeds up training but uses more RAM)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for caching dataset on disk (using PersistentDataset)')
    parser.add_argument('--warmup_epochs', type=int, default=config.training["warmup_epochs"],
                        help='Number of warmup epochs')
    parser.add_argument('--warmup_type', type=str, default=config.training["warmup_type"],
                        choices=['linear', 'exponential'],
                        help='Type of learning rate warmup')
    parser.add_argument('--max_grad_norm', type=float, default=config.training["max_grad_norm"],
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--center_crop', type=str, default=config.training["center_crop"],
                        help='Apply center cropping with size format "DxHxW", e.g. "64x64x64"')
    parser.add_argument('--augment', action='store_true', default=config.training["augment"],
                        help='Apply medical image augmentation')
    parser.add_argument('--image_size', type=int, default=config.model['params']["image_size"],
                        help='3D input image size')
    parser.add_argument('--num_classes', type=int, default=config.data['num_classes'],
                        help='num_classes')
    parser.add_argument('--pretrained_path', type=str, default=config.training['pretrained_path'],
                        help='pretrained_path')
    parser.add_argument('--lr', type=float, default=config.optimizer['params']['lr'],
                        help='pretrained_path')
    return parser.parse_args()

def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config.yaml')

    # 解析命令行参数
    args = parse_args(config)

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
    dataset = NoduleDataset(args.data_dir, transform=None)

    # 显示数据集基本信息
    print(f"数据集样本总数: {len(dataset.samples)}")

    # 初始化交叉验证器，传入缓存目录
    cv = CrossValidator(dataset, config)
    device = config.training['device']

    # 存储每折的最佳F1分数
    fold_scores = []

    # 预加载模型配置，避免在循环中重复访问
    model_config = {
        "num_classes": config.data["num_classes"],
        "image_size": args.image_size,
        "patch_size": config.model["params"]["patch_size"],
        "dim": config.model["params"]["dim"],
        "depth": config.model["params"]["depth"],
        "heads": config.model["params"]["heads"],
        "mlp_dim": config.model["params"]["mlp_dim"],
        "pool": config.model["params"]["pool"],
        "center_crop": config.training["center_crop"]
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

        # 打印模型结构
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 加载预训练权重
        if args.pretrained_path:
            print(f"Loading pretrained weights from {args.pretrained_path}...")
            try:
                model.load_pretrained_dino(args.pretrained_path)
            except Exception as e:
                print(f"加载预训练权重出错: {str(e)}")

        # 训练模型
        best_f1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            fold=fold,
            warmup_epochs=args.warmup_epochs,
            warmup_type=args.warmup_type,
            max_grad_norm=args.max_grad_norm,
        )

        fold_scores.append(best_f1)
        fold_time = time.time() - fold_start_time
        print(f'Fold {fold + 1} completed in {fold_time:.2f}s with Best F1: {best_f1:.4f}')

        # 释放模型内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印最终结果
    print('\nCross-Validation Results:')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold + 1}: {score:.4f}')
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f'Mean F1: {mean_f1:.4f} ± {std_f1:.4f}')

    # 打印总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == '__main__':
    main()