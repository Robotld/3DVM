import torch
import torch.nn as nn
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
                 pool='max'):
        super().__init__(img_size = image_size, patch_size = patch_size, in_chans = in_chans,
                         num_classes = num_classes, embed_dim = dim, depth = depth, num_heads = heads)

        self.pool = pool

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
        elif pool == 'all':
            mlp_input_dim = dim * (num_patches + 1)  # All tokens concatenated
        else:
            raise ValueError(f"不支持的池化方法: {pool}")

        # 创建MLP分类头
        self.mlp_head = nn.Sequential(
            # nn.Linear(mlp_input_dim, mlp_dim),
            # nn.GELU(),
            # nn.BatchNorm1d(mlp_dim),
            # nn.Linear(mlp_dim, num_classes)
            nn.Linear(mlp_input_dim, num_classes),
            # nn.Sigmoid()
        )

        # 保存注意力权重用于可视化
        self.attention_weights = None

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, channels, depth, height, width)
            output_attention: 是否返回注意力权重用于可视化
        """

        x = x.float()
        batch_size = x.shape[0]
        # 生成3D补丁嵌入
        x = self.patch_embedding(x)  # (batch_size, dim, d', h', w')

        # 获取实际补丁数量
        d_patches = x.shape[2]
        h_patches = x.shape[3]
        w_patches = x.shape[4]
        actual_patches = d_patches * h_patches * w_patches

        # 展平补丁并转置为序列形式
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # 添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)

        x = x + self.pos_embedding
        # Dropout
        x = self.dropout(x)
        # Transformer编码器

        attention = []
        # 存储中间特征
        intermediate_features = []
        # 通过Transformer层
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 每隔几层保存一次特征
            # if i % 2 == 0 or i == len(self.blocks) - 1:
            # 提取只包含补丁的特征(排除CLS token)
            patch_tokens = x[:, 1:].clone()
            # 重塑回3D体积形状用于流场分析
            reshaped = patch_tokens.transpose(1, 2).reshape(
                batch_size, self.embed_dim, d_patches, h_patches, w_patches
            )
            intermediate_features.append(reshaped)

        # 提取分类令牌和补丁令牌
        cls_token_out = x[:, 0]  # (batch_size, dim)
        patch_tokens = x[:, 1:]  # (batch_size, num_patches, dim)
        features = x

        # 根据池化方法聚合特征
        if self.pool == 'mean':
            pooled_patches = torch.mean(patch_tokens, dim = 1)
            combined_features = torch.cat([cls_token_out, pooled_patches], dim = 1)
        elif self.pool == 'max':
            pooled_patches = torch.max(patch_tokens, dim = 1)[0]
            combined_features = torch.cat([cls_token_out, pooled_patches], dim = 1)
        else:  # pool == 'all'
            combined_features = x.reshape(batch_size, -1)

        # 最终分类头
        out = self.mlp_head(combined_features)
        return out, features, intermediate_features

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
