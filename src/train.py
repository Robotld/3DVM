import numpy as np
from ruamel.yaml import YAML

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import optim
from torch.utils.data import Subset, DataLoader

from models.losses import LabelSmoothingCrossEntropy
from src.models.ViT_3D import ViT3D
from src.models.dataset import NoduleDataset
from src.models.transform import make_3d_transform


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

class CrossValidator:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config.cross_validation
        self.train_config = config.training

        # 获取所有标签用于分层采样
        self.labels = [label for _, label in dataset]

        # 初始化分层K折交叉验证器
        self.skf = StratifiedKFold(
            n_splits=self.config["n_splits"],
            shuffle=self.config["shuffle"],
            random_state=self.train_config["random_seed"]
        )

    def get_folds(self):
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(np.zeros(len(self.labels)), self.labels)):
            # 创建训练集和验证集的子集
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            # 创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=self.train_config["batch_size"],
                shuffle=True,
                num_workers=self.train_config["num_workers"]
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.train_config["batch_size"],
                shuffle=False,
                num_workers=self.train_config["num_workers"]
            )

            yield fold, train_loader, val_loader


def train_model(model, train_loader, val_loader, config, fold, device):
    """训练单个折的模型"""
    # 初始化优化器
    optimizer = getattr(optim, config.optimizer["name"])(
        model.parameters(), **config.get_optimizer_params(model)
    )

    # 初始化学习率调度器
    scheduler = getattr(optim.lr_scheduler, config.scheduler["name"])(
        optimizer, **config.get_scheduler_params()
    )
    class_counts = []
    # 初始化损失函数
    if config.training["loss"]["class_weights"] == "auto":
        labels = []
        for _, y in train_loader.dataset:
            labels.append(y)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.FloatTensor(class_weights).to(config.training["device"])
    else:
        class_weights = None
    print(class_weights)
    loss_fn = LabelSmoothingCrossEntropy(
        smoothing=config.training["loss"]["smoothing"],
        class_weights=class_weights
    )
    # loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_model_path = f'best_model_fold_{fold}.pth'

    for epoch in range(config.training["num_epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        val_probs = []  # 用于计算AUC

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
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

        # 计算整体指标
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if len(class_counts) > 2:
            val_auc = roc_auc_score(val_labels, np.array(val_probs), multi_class="ovr", average="macro")
        else:
            val_auc = roc_auc_score(val_labels, np.array(np.argmax(val_probs, axis=1)).ravel())

        # 计算每个类的准确率
        per_class_accuracy = {}
        unique_classes = np.unique(val_labels)
        for cls in unique_classes:
            idxs = [i for i, label in enumerate(val_labels) if label == cls]
            correct = sum(1 for i in idxs if val_preds[i] == cls)
            per_class_accuracy[cls] = correct / len(idxs)
        # 将每个类的准确率压缩到一行字符串输出
        per_class_str = ", ".join([f"Class {cls}: {acc:.4f}" for cls, acc in per_class_accuracy.items()])

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

        # 打印进度和指标
        print(f'Fold {fold + 1}, Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_correct / train_total:.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}, '
              f'Val AUC: {val_auc:.4f}')
        print(f'Per Class Accuracies: {per_class_str}')

        scheduler.step()

    return best_val_f1


def main():
    # 加载配置
    config = ConfigManager(r"E:\workplace\3D\src\config\config.yaml")

    # 设置随机种子
    torch.manual_seed(config.training["random_seed"])
    np.random.seed(config.training["random_seed"])

    # 加载数据集
    dataset = NoduleDataset(
        root_dir=config.data["root_dir"],
        transform=make_3d_transform(train=True) if config.data["transform"] == "3d" else None
    )

    # 初始化交叉验证器
    cv = CrossValidator(dataset, config)
    device = config.training['device']
    # 存储每折的最佳F1分数
    fold_scores = []

    # 进行交叉验证
    for fold, train_loader, val_loader in cv.get_folds():
        print(f'\nTraining Fold {fold + 1}')

        # 初始化新的模型
        model = ViT3D(
            num_classes=config.data["num_classes"],
            image_size=config.model["params"]["image_size"],
            patch_size=config.model["params"]["patch_size"],
            dim=config.model["params"]["dim"],
            depth=config.model["params"]["depth"],
            heads=config.model["params"]["heads"],
            mlp_dim=config.model["params"]["mlp_dim"],
        ).to(config.training["device"])
        model.load_pretrained_dino(model, r"E:\ultralytics\facebookdinov2-with-registers-small-imagenet1k-1-layer")
        print(model)
        # 训练模型
        best_f1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            fold=fold
        )

        fold_scores.append(best_f1)
        print(f'Fold {fold + 1} Best F1: {best_f1:.4f}')

    # 打印最终结果
    print('\nCross-Validation Results:')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold + 1}: {score:.4f}')
    print(f'Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')


if __name__ == '__main__':
    main()