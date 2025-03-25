import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


def one_epoch_train(model, data_loader, optimizer, loss_fn, device, loss1_weight=1.0, loss2=None, loss3=None, loss3_weight=0.5,
                    train=True, use_amp=False, scaler=None, max_grad_norm=1.0):
    """执行一个epoch的训练或验证"""
    # 设置模型状态
    model.train() if train else model.eval()

    losses, all_preds, all_labels, all_probs = [], [], [], []
    flow_losses = []
    data_iter = tqdm(data_loader, desc="训练中" if train else "验证中", leave=False, position=0)

    # 定义前向传播和损失计算函数
    def forward_pass(inputs):
        model_outputs, features, flow = model(inputs)
        loss_all = loss1_weight * loss_fn(model_outputs, y)
        info = {}
        if loss3:
            flow_loss, info = loss3(features, y)
            loss_all += loss3_weight*flow_loss

        return model_outputs, loss_all, info

    # 训练或验证循环
    with torch.set_grad_enabled(train):
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, loss, flow_info = forward_pass(x)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs, loss, flow_info = forward_pass(x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            else:
                outputs, loss, flow_info = forward_pass(x)

            # 记录流场损失
            if loss3 and 'total_flow_loss' in flow_info:
                flow_losses.append(flow_info['total_flow_loss'])

            # 收集结果
            losses.append(loss.item())
            _, preds = outputs.max(1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    # 转换为NumPy数组以计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'loss': np.mean(losses)
    }

    # 添加流场损失指标
    if loss3 is not None and flow_losses:
        metrics['flow_loss'] = np.mean(flow_losses)

    # 计算AUC
    num_classes = all_probs.shape[1]
    if num_classes > 2:
        metrics["auc"] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        metrics["auc"] = roc_auc_score(all_labels, all_probs[:, 1])

    # 计算每类准确率
    unique_classes = np.unique(all_labels)
    metrics['per_class_accuracy'] = {
        cls: np.mean(all_preds[all_labels == cls] == cls)
        for cls in unique_classes
    }
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs

    return np.mean(losses), metrics
