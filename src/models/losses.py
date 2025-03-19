from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        """
        :param smoothing: Smoothing factor for label smoothing.
        :param class_weights: Tensor of shape (num_classes,) containing weights for each class.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, pred, target):
        """
        :param pred: Predictions of shape (batch_size, num_classes)
        :param target: Ground truth labels of shape (batch_size,)
        :return: Loss (scalar)
        """
        n_classes = pred.size(1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(pred, dim=1)

        if self.class_weights is not None:
            # Ensure class_weights is on the same device as pred
            self.class_weights = self.class_weights.to(pred.device)
            loss = -(true_dist * log_probs * self.class_weights.unsqueeze(0)).sum(dim=1)
        else:
            loss = -(true_dist * log_probs).sum(dim=1)

        return loss.mean()



class AttentionMCLoss(nn.Module):
    """
    基于注意力权重的MC-Loss实现
    """

    def __init__(self, embed_dim, num_classes, lambda_disc=0.1, temperature=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lambda_disc = lambda_disc
        self.temperature = temperature

        # 用于分类的投影层
        self.classifier = nn.Linear(embed_dim, num_classes)

    def create_mask(self, batch_size, device):
        """创建通道级随机掩码，保持2:1的开启/关闭比例"""
        mask_template = [1, 1, 0]  # 2:1的开启/关闭比例
        mask = []

        for _ in range(self.embed_dim // len(mask_template)):
            shuffled = random.sample(mask_template, len(mask_template))
            mask.extend(shuffled)

        # 处理剩余维度
        remainder = self.embed_dim % len(mask_template)
        if remainder > 0:
            mask.extend([1] * remainder)

        # 扩展到批次维度
        mask = np.array([mask for _ in range(batch_size)], dtype=np.float32)
        return torch.from_numpy(mask).to(device)

    def forward(self, features, attention_weights, targets):
        """
        计算基于注意力的MC-Loss

        Args:
            features: ViT最后层特征 [batch_size, seq_len, embed_dim]
            attention_weights: 注意力权重 [batch_size, layers, heads, seq_len, seq_len]
            targets: 目标类别 [batch_size]

        Returns:
            total_loss: 总损失
            cls_loss: 分类损失
            disc_loss: 判别损失
        """
        batch_size = features.size(0)
        device = features.device

        # 提取最后一层的CLS token注意力权重 [batch_size, heads, seq_len]
        # attention_weights的形状为 [batch, layers, heads, seq_len, seq_len]
        # 我们取最后一层(索引-1)的注意力权重
        last_layer_attn = attention_weights[:, -1]  # [batch, heads, seq_len, seq_len]
        cls_attn = last_layer_attn[:, :, 0, 1:]  # [batch, heads, seq_len-1] (剔除CLS token自注意)

        # 提取CLS token特征
        cls_features = features[:, 0]  # [batch_size, embed_dim]

        # 创建并应用掩码
        mask = self.create_mask(batch_size, device)
        masked_cls = cls_features * mask

        # 计算判别损失：鼓励每个注意力头关注不同区域
        # 对每个头的注意力应用softmax
        cls_attn_softmax = F.softmax(cls_attn / self.temperature, dim=2)

        # 获取每个头最大的注意力权重
        max_attn, _ = torch.max(cls_attn_softmax, dim=2)  # [batch, heads]

        # 判别损失：鼓励每个头有明确的注意力焦点
        disc_loss = 1.0 - torch.mean(max_attn)

        # 分类损失：使用掩码后的CLS token进行分类
        logits = self.classifier(masked_cls)
        cls_loss = F.cross_entropy(logits, targets)

        # 总损失
        total_loss = cls_loss + self.lambda_disc * disc_loss

        return total_loss, cls_loss, disc_loss, logits


import torch
import torch.nn as nn
import torch.nn.functional as F


class Enhanced3DVITLoss(nn.Module):
    def __init__(self,
                 cls_weight=1.0,
                 diversity_weight=0.5,
                 orthogonal_weight=0.3,
                 local_weight=0.2,
                 temp=0.5):
        """
        增强型3DVIT损失函数 - 促使不同图像块(tokens)学习不同特征

        参数:
            cls_weight: 分类损失权重
            diversity_weight: Token多样性损失权重
            orthogonal_weight: Token正交性损失权重
            local_weight: 局部特征增强损失权重
            temp: 温度参数，用于调节相似度计算
        """
        super(Enhanced3DVITLoss, self).__init__()
        self.cls_weight = cls_weight
        self.diversity_weight = diversity_weight
        self.orthogonal_weight = orthogonal_weight
        self.local_weight = local_weight
        self.temp = temp
        self.cls_criterion = nn.CrossEntropyLoss()

    def token_diversity_loss(self, features):
        """Token多样性损失：鼓励不同图像块捕获不同特征"""
        # 输入features形状: [B, num_tokens, hidden_dim]

        # 只使用patch tokens，排除CLS token
        patch_tokens = features[:, 1:, :]  # [B, num_patches, hidden_dim]
        b, num_patches, hidden_dim = patch_tokens.shape

        # 通过特征向量的余弦相似度衡量不同图像块的相似程度
        # 归一化每个patch token的特征向量
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=2)  # 沿hidden_dim维度归一化

        # 计算不同patch tokens之间的相似度
        similarity = torch.bmm(patch_tokens_norm, patch_tokens_norm.transpose(1, 2))  # [B, num_patches, num_patches]

        # 排除自身相似度(对角线)
        mask = (1 - torch.eye(num_patches, device=features.device)).unsqueeze(0)
        similarity = similarity * mask

        # 图像块多样性损失：相似度越低越好
        diversity_loss = torch.mean(F.relu(similarity - self.temp))

        return diversity_loss

    def token_orthogonal_loss(self, features):
        """Token正交性损失：促使不同图像块的特征具有正交性"""
        # 输入features形状: [B, num_tokens, hidden_dim]

        # 只使用patch tokens，排除CLS token
        patch_tokens = features[:, 1:, :]  # [B, num_patches, hidden_dim]
        b, num_patches, hidden_dim = patch_tokens.shape

        # 计算不同patch tokens之间的内积
        gram = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))  # [B, num_patches, num_patches]

        # 计算与单位矩阵的差异
        identity = torch.eye(num_patches, device=features.device).unsqueeze(0).expand(b, -1, -1)
        ortho_loss = F.mse_loss(gram, identity)

        return ortho_loss

    def local_feature_enhancement_loss(self, features):
        """局部特征增强损失：鼓励每个图像块捕获独特特征"""
        # 输入features形状: [B, num_tokens, hidden_dim]

        # 只使用patch tokens，排除CLS token
        patch_tokens = features[:, 1:, :]  # [B, num_patches, hidden_dim]
        b, num_patches, hidden_dim = patch_tokens.shape

        # 计算相邻图像块之间的差异，鼓励捕获局部变化
        # 假设图像块在空间上是按行排列的
        patch_size = int(np.sqrt(num_patches))

        if patch_size ** 2 == num_patches:  # 确保是完美平方数
            # 将token重排为二维网格以反映空间位置关系
            tokens_grid = patch_tokens.reshape(b, patch_size, patch_size, hidden_dim)

            # 计算水平和垂直方向的差异
            h_diff = tokens_grid[:, :, 1:, :] - tokens_grid[:, :, :-1, :]  # 水平差异
            v_diff = tokens_grid[:, 1:, :, :] - tokens_grid[:, :-1, :, :]  # 垂直差异

            # 鼓励差异明显，表示每个图像块都有独特特征
            h_loss = -torch.mean(torch.norm(h_diff, dim=3))
            v_loss = -torch.mean(torch.norm(v_diff, dim=3))

            local_loss = (h_loss + v_loss) / 2
        else:
            # 如果不是完美平方数，只考虑序列上的相邻关系
            seq_diff = patch_tokens[:, 1:, :] - patch_tokens[:, :-1, :]
            local_loss = -torch.mean(torch.norm(seq_diff, dim=2))

        return local_loss

    def forward(self, pred_logits, target, last_hidden_states):
        """
        前向计算损失

        参数:
            pred_logits: 模型的分类预测输出，形状为 [B, num_classes]
            target: 真实标签，形状为 [B]
            last_hidden_states: ViT最后一层的隐藏状态，形状为 [B, num_tokens, hidden_dim]
        """
        # 1. 基础分类损失
        cls_loss = self.cls_criterion(pred_logits, target)

        # 2. Token多样性损失
        diversity_loss = self.token_diversity_loss(last_hidden_states)

        # 3. Token正交性损失
        ortho_loss = self.token_orthogonal_loss(last_hidden_states)

        # 4. 局部特征增强损失
        local_loss = self.local_feature_enhancement_loss(last_hidden_states)

        # 5. 总损失
        total_loss = (self.cls_weight * cls_loss +
                      self.diversity_weight * diversity_loss +
                      self.orthogonal_weight * ortho_loss +
                      self.local_weight * local_loss)

        # 返回总损失和各部分损失（用于监控）
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'diversity_loss': diversity_loss,
            'ortho_loss': ortho_loss,
            'local_loss': local_loss
        }

        return total_loss, loss_dict

