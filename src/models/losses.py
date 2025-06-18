import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import tensor

LOSS_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "FocalLoss": None,  # 将在下面定义
    "None": None  # 无
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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


"""
边界流场损失模块
用于增强3D结节浸润性判断的边界流场特征学习
"""


class BoundaryFlowLoss(nn.Module):
    """
    简化的边界流场损失
    """
    def __init__(self, weight=0.5, margin=0.3):
        super().__init__()
        self.weight = weight
        self.margin = margin

    def forward(self, features, labels):
        """计算边界流场损失"""
        # 处理特征输入
        if isinstance(features, (list, tuple)):
            # 选择中间层特征
            features = features[len(features) // 2]

        # 确保特征是标准张量
        if hasattr(features, 'as_tensor'):
            features = features.as_tensor()

        # 跳过不符合形状要求的特征
        if features.ndim != 5:  # 需要[B,C,D,H,W]格式
            return torch.tensor(0.0, device=labels.device), {'total_flow_loss': 0.0}

        # 安全地计算3D梯度 - 避免维度问题
        # 1. 计算各方向梯度
        with torch.no_grad():  # 提高效率，梯度计算不需要自动微分
            # 深度方向梯度
            if features.shape[2] > 1:
                grad_d = features[:, :, 1:] - features[:, :, :-1]
            else:
                grad_d = torch.zeros_like(features[:, :, :1])

            # 高度方向梯度
            if features.shape[3] > 1:
                grad_h = features[:, :, :, 1:] - features[:, :, :, :-1]
            else:
                grad_h = torch.zeros_like(features[:, :, :, :1])

            # 宽度方向梯度
            if features.shape[4] > 1:
                grad_w = features[:, :, :, :, 1:] - features[:, :, :, :, :-1]
            else:
                grad_w = torch.zeros_like(features[:, :, :, :, :1])

        # 2. 计算边界复杂度特征
        # 边界强度 - 梯度幅值
        grad_d_abs = torch.mean(torch.abs(grad_d), dim=[1, 2, 3, 4])
        grad_h_abs = torch.mean(torch.abs(grad_h), dim=[1, 2, 3, 4])
        grad_w_abs = torch.mean(torch.abs(grad_w), dim=[1, 2, 3, 4])

        # 边界复杂度指标 - 三个方向梯度的综合
        boundary_complexity = grad_d_abs + grad_h_abs + grad_w_abs

        # 3. 分离浸润性和非浸润性样本
        invasive_mask = (labels == 1)
        non_invasive_mask = (labels == 0)

        # 4. 计算对比损失
        loss = torch.tensor(0.0, device=features.device)

        if torch.any(invasive_mask) and torch.any(non_invasive_mask):
            # 获取两类样本的边界复杂度
            invasive_complexity = boundary_complexity[invasive_mask].mean()
            non_invasive_complexity = boundary_complexity[non_invasive_mask].mean()

            # 浸润性结节应有更高的边界复杂度
            loss = F.relu(non_invasive_complexity - invasive_complexity + self.margin)

        # 5. 计算最终损失值
        weighted_loss = self.weight * loss

        # 返回损失和诊断信息
        loss_info = {
            'flow_loss': loss.item(),
            'total_flow_loss': weighted_loss.item(),
            'invasive_complexity': invasive_complexity.item() if 'invasive_complexity' in locals() else 0,
            'non_invasive_complexity': non_invasive_complexity.item() if 'non_invasive_complexity' in locals() else 0
        }

        return weighted_loss, loss_info


class EdgeAwareFlowModule(nn.Module):
    """
    边缘感知流场模块

    为ViT模型提取边缘流场特征

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        # 边缘注意力机制
        self.edge_attention = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向传播"""
        # 提取特征
        feat = F.gelu(self.norm1(self.conv1(x)))
        feat = self.norm2(self.conv2(feat))

        # 计算边缘注意力 - 关注梯度高的区域
        edge_map = self.edge_attention(torch.abs(feat))

        # 边缘加权特征
        refined_features = feat * edge_map

        return refined_features, edge_map


LOSS_REGISTRY["FocalLoss"] = FocalLoss
LOSS_REGISTRY["BoundaryFlowLoss"] = BoundaryFlowLoss


def build_loss(enabled, name, config):
    """根据配置构建损失函数"""
    if enabled is False:
        return None
    # 构建损失函数实例
    loss = LOSS_REGISTRY[name]()  # 注意加 () 才算调用构造函数
    loss_params = config.losses[name]['params']
    if loss_params:
        loss = LOSS_REGISTRY[name](**loss_params)

    return loss


class MultitaskLoss(nn.Module):
    """多任务损失函数，处理复发预测和亚型分类"""

    def __init__(self, recurrence_weight=1.0, subtype_weight=1.0, similarity_weight=0.5):
        """
        参数:
            recurrence_weight: 复发预测任务的权重
            subtype_weight: 亚型分类任务的权重
            similarity_weight: 提示向量相似度损失的权重
        """
        super().__init__()
        self.recurrence_weight = recurrence_weight
        self.subtype_weight = subtype_weight
        self.similarity_weight = similarity_weight

        self.recurrence_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device='cuda'), ignore_index=-1)
        # self.recurrence_criterion = FocalLoss()
        self.subtype_criterion = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, recurrence_logits, subtype_logits, recurrence_labels, subtype_labels, similarity_loss):
        """
        计算多任务损失
        参数:
            recurrence_logits: 复发预测的输出 [batch_size, 2]
            subtype_logits: 亚型分类的输出 [batch_size, 5]
            similarity_loss: 提示向量相似度损失
            recurrence_labels: 复发标签 [batch_size]
            subtype_labels: 亚型标签 [batch_size]
        """
        # 复发预测损失 - 直接使用CrossEntropyLoss的ignore_index功能
        recurrence_loss = self.recurrence_criterion(
            recurrence_logits,
            recurrence_labels  # 确保标签是长整型
        )

        # 亚型分类损失
        subtype_loss = self.subtype_criterion(
            subtype_logits,
            subtype_labels  # 确保标签是长整型
        )

        # 总损失 = 复发预测权重 * 复发损失 + 亚型分类权重 * 亚型损失 + 相似度损失权重 * 相似度损失
        total_loss = self.recurrence_weight * recurrence_loss + self.subtype_weight * subtype_loss + similarity_loss


        return total_loss, recurrence_loss, subtype_loss, similarity_loss