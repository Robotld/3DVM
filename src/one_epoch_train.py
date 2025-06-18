import numpy as np
import torch
from tqdm import tqdm


def one_epoch_train(model, data_loader, optimizer, loss_fn, device,
                    loss1_weight=1.0, loss2=None, loss3=None, loss2_weight=0.5,
                    train=True, use_amp=True, scaler=None, max_grad_norm=1.0):
    """执行一个epoch的训练或验证，支持AMP"""

    # 设置模型状态
    model.train() if train else model.eval()

    losses, all_labels, all_probs = [], [], []
    flow_losses, similarity_losses = [], []  # 添加similarity_losses列表

    data_iter = tqdm(data_loader,
                     desc = "训练中" if train else "验证中",
                     leave = False,
                     ncols = 100,
                     mininterval = 0.01)

    # 重新定义前向传播函数，将y作为参数传入
    def forward_pass(inputs, targets):
        model_outputs, features, flow, similarity = model(inputs)
        loss_all = loss1_weight * loss_fn(model_outputs, targets)
        info = {}
        if loss2:
            flow_loss, info = loss2(features, targets)
            loss_all += loss2_weight * flow_loss
        if loss3:
            loss_all += 0.5*similarity
        return model_outputs, loss_all, info, similarity

    # 训练或验证循环
    with torch.set_grad_enabled(train):
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)

            if train:
                if use_amp:
                    with torch.amp.autocast(device_type = 'cuda', enabled = True):
                        outputs, loss, flow_info, similarity_loss = forward_pass(x, y)  # 修改调用方式
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs, loss, flow_info, similarity_loss = forward_pass(x, y)  # 修改调用方式
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                outputs, loss, flow_info, similarity_loss = forward_pass(x, y)  # 修改调用方式

            # 记录损失
            losses.append(loss.item())
            if loss3:
                similarity_losses.append(similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else similarity_loss)
            
            # 记录流场损失
            if loss2 and 'total_flow_loss' in flow_info:
                flow_losses.append(flow_info['total_flow_loss'])

            # 收集结果
            _, preds = outputs.max(1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(y.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        # 关闭进度条
        data_iter.close()
        
    # 转换为NumPy数组以计算指标
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    metrics = {
        'loss': np.mean(losses),
    }

    if flow_losses:
        metrics['flow_loss'] = np.mean(flow_losses)

    # 计算平均similarity_loss
    avg_similarity_loss = np.mean(similarity_losses) if similarity_losses else 0.0

    # 保存预测结果
    metrics['labels'] = all_labels
    metrics['probabilities'] = all_probs

    return np.mean(losses), metrics, avg_similarity_loss  # 返回平均similarity_loss