import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import VisionTransformer
from transformers import Dinov2ForImageClassification
import numpy as np
import matplotlib.pyplot as plt


class ViT3D(VisionTransformer):
    def __init__(self,
                 num_classes=5,
                 dim=384,
                 depth=4,
                 heads=8,
                 mlp_dim=512,
                 dropout=0.5,
                 emb_dropout=0.1,
                 patch_size=16,
                 image_size=96,
                 crop_size=36,
                 in_chans=1,
                 pool='max',
                 cpt_num=5,
                 mlp_num=3):
        super().__init__(img_size = image_size, patch_size = patch_size, in_chans = in_chans,
                         num_classes = num_classes, embed_dim = dim, depth = depth, num_heads = heads)

        self.pool = pool
        self.num_classes = num_classes

        if crop_size:
            image_size = crop_size
        # 计算3D补丁数量
        patch_dim = patch_size
        num_patches = (image_size // patch_dim) ** 3
        patch_size_3d = (patch_dim, patch_dim, patch_dim)
        self.num_patches = num_patches

        # 创建3D补丁嵌入
        self.patch_embedding = nn.Conv3d(in_chans, dim,
                                         kernel_size = patch_size_3d,
                                         stride = patch_size_3d)

        # 保留原始位置编码用于备选
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # 添加类别令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 添加类原型向量
        self.class_prompts = nn.Parameter(torch.randn(1, cpt_num, dim))

        # 创建Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim,
            nhead = heads,
            dim_feedforward = mlp_dim,
            dropout = dropout,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = depth
        )

        # 根据池化方法确定MLP输入维度
        if pool in ['mean', 'max']:
            mlp_input_dim = dim * 2  # CLS token + pooled tokens
        else:
            raise ValueError(f"不支持的池化方法: {pool}")

        # 创建MLP分类头
        if mlp_num == 1:
            self.mlp_head = nn.Sequential(
                nn.Linear(mlp_input_dim, num_classes)
            )
        elif mlp_num == 2:
            self.mlp_head = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(mlp_dim, num_classes)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.BatchNorm1d(mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, num_classes)
            )

        # 保存注意力权重用于可视化
        self.attention_weights = None

    def prompt_cosine_similarity_loss(self, low=0.4, high=0.8):
        """
        批量计算 prompts (类提示向量）间的余弦相似性软范围约束损失
        :param low: 下限阈值（低于此值提示过于不同）
        :param high: 上限阈值（高于此值提示过于相似）
        :return: tensor loss
        """
        prompts = self.class_prompts  # [batch_size, num_classes, embed_dim]

        batch_size, num_classes, embed_dim = prompts.shape

        if num_classes < 2:
            return torch.tensor(0.0, device=prompts.device, requires_grad=True)

        # 批量计算 prompts 的相似度矩阵
        sim_matrix = F.cosine_similarity(
            prompts.unsqueeze(2),  # [batch_size, num_classes, 1, embed_dim]
            prompts.unsqueeze(1),  # [batch_size, 1, num_classes, embed_dim]
            dim=-1  # 在 embed_dim维度计算余弦相似度
        )  # 最终得到 [batch_size, num_classes, num_classes]

        # 对角线掩码（批量进行mask）
        mask = ~torch.eye(num_classes, dtype=torch.bool, device=prompts.device).unsqueeze(
            0)  # [1, num_classes, num_classes]
        sim_values = sim_matrix[mask.expand(batch_size, -1, -1)]  # [batch_size, num_classes*(num_classes -1)]

        # 软约束损失计算
        loss_high = torch.clamp(sim_values - high, min=0) ** 2
        loss_low = torch.clamp(low - sim_values, min=0) ** 2
        loss = torch.mean(loss_high + loss_low)

        return loss

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, channels, depth, height, width)
        """

        x = x.float()
        batch_size = x.shape[0]
        # 生成3D补丁嵌入
        x = self.patch_embedding(x)  # (batch_size, dim, d', h', w')

        # 获取实际补丁数量
        d_patches = x.shape[2]
        h_patches = x.shape[3]
        w_patches = x.shape[4]

        # 展平补丁并转置为序列形式
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # 添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)

        # 添加类别提示向量 - 新增代码
        class_prompts = self.class_prompts.expand(batch_size, -1, -1)
        x = torch.cat((x, class_prompts), dim = 1)

        # 为所有token添加位置编码，包括原始位置编码和类别提示位置编码 - 新增代码
        original_tokens_count = self.num_patches + 1  # patch_tokens + cls_token
        x[:, :original_tokens_count] = x[:, :original_tokens_count] + self.pos_embedding[:, :x[:, :original_tokens_count].size(1)]

        # Dropout
        x = self.dropout(x)

        # 存储中间特征
        intermediate_features = []
        # 通过Transformer层
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 提取只包含补丁的特征(排除CLS token和类别提示向量)
            patch_tokens = x[:, 1:original_tokens_count].clone()
            # 重塑回3D体积形状用于流场分析
            reshaped = patch_tokens.transpose(1, 2).reshape(
                batch_size, self.embed_dim, d_patches, h_patches, w_patches
            )
            intermediate_features.append(reshaped)

        # 提取类别提示向量 - 新增代码
        class_prompt_features = x[:, original_tokens_count:]

        similarity_loss = self.prompt_cosine_similarity_loss(0.3, 0.7)

        # 提取分类令牌和补丁令牌（不包括类别提示向量）
        cls_token_out = x[:, 0]  # (batch_size, dim)
        patch_tokens = x[:, 1:original_tokens_count]  # (batch_size, num_patches, dim)

        class_prompts_pooled = class_prompt_features
        pooled_patches = patch_tokens
        features = x

        # 根据池化方法聚合特征
        if self.pool == 'mean':
            pooled_patches = torch.mean(patch_tokens, dim = 1)
            class_prompts_pooled = torch.mean(class_prompt_features, dim = 1)  # (batch_size, dim)
        elif self.pool == 'max':
            pooled_patches = torch.max(patch_tokens, dim = 1)[0]
            class_prompts_pooled = torch.max(class_prompt_features, dim = 1)[0]  # (batch_size, dim)


        combined_features = torch.cat([cls_token_out, pooled_patches], dim = 1)
        # 最终分类头
        out = self.mlp_head(combined_features)
        return out, features, intermediate_features, similarity_loss

    def load_pretrained_dino(self, path):
        """加载预训练的DINOv2权重"""
        try:
            basic_model = Dinov2ForImageClassification.from_pretrained(
                path,
                num_labels = self.num_classes,
                ignore_mismatched_sizes = True
            )
            self.load_state_dict(basic_model.state_dict(), strict = False)
            # 对于加载的预训练模型，直接把CSL随机初始化。
            print("成功加载DINOv2预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
