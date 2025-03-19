import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


def one_epoch_train(model, data_loader, optimizer, loss_fn,  device, loss2=None,
                    train=True, use_amp=False, scaler=None, max_grad_norm=1.0):
    """执行一个epoch的训练或验证"""
    # 设置模型状态
    model.train() if train else model.eval()

    losses, all_preds, all_labels, all_probs = [], [], [], []
    data_iter = tqdm(data_loader, desc="训练中" if train else "验证中", leave=False, position=0)
    # 训练或验证循环
    with torch.set_grad_enabled(train):
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)

            # 训练模式
            if train:
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, features = model(x)

                        loss = loss_fn(outputs, y)
                        if loss2:
                            ChannelLoss, loss_dict = loss2(outputs, y, features)
                            loss += ChannelLoss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs, features = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            # 验证模式
            else:
                outputs, features = model(x)
                loss = loss_fn(outputs, y)
                if loss2:
                    ChannelLoss, loss_dict = loss2(outputs, y, features)
                    loss += ChannelLoss

            # 收集结果
            losses.append(loss.item())
            _, preds = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'loss': np.mean(losses)
    }

    # 计算AUC

    num_classes = all_probs.shape[1]
    if num_classes > 2:
        metrics["auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        # 二分类情况下，使用正类概率而不是预测标签
        metrics["auc"] = roc_auc_score(all_labels, all_probs[:, 1])
        # print(roc_auc_score(all_labels, all_probs[:, 0]), roc_auc_score(all_labels, all_probs[:, 1]), roc_auc_score(all_labels, all_preds))
    # 计算每类准确率
    unique_classes = np.unique(all_labels)
    metrics['per_class_accuracy'] = {
        cls: np.mean(all_preds[all_labels == cls] == cls)
        for cls in unique_classes
    }

    return np.mean(losses), metrics
