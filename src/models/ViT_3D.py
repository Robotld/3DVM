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
                 dropout=0.2,
                 emb_dropout=0.1,
                 patch_size=16,
                 image_size=96,
                 in_chans=1,
                 pool='mean',
                 center_crop=64,
                 use_3d_pos_embed=True):
        super().__init__(img_size=image_size, patch_size=patch_size, in_chans=in_chans,
                         num_classes=num_classes, embed_dim=dim, depth=depth, num_heads=heads)

        self.pool = pool
        self.use_3d_pos_embed = use_3d_pos_embed

        # 计算3D补丁数量
        patch_dim = patch_size
        if center_crop:
            image_size = center_crop
        num_patches = (image_size // patch_dim) ** 3
        patch_size_3d = (patch_dim, patch_dim, patch_dim)
        self.num_patches = num_patches

        # 创建3D补丁嵌入
        self.patch_embedding = nn.Conv3d(in_chans, dim,
                                         kernel_size=patch_size_3d,
                                         stride=patch_size_3d)

        # 保留原始位置编码用于备选
        self.orig_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # 添加类别令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 如果使用3D位置编码，则添加位置编码生成器
        if use_3d_pos_embed:
            pos_dim = dim // 3
            # 深度位置编码器
            self.pos_d_encoding = nn.Sequential(
                nn.Linear(1, pos_dim),
                nn.LayerNorm(pos_dim),
                nn.GELU(),
                nn.Linear(pos_dim, pos_dim)
            )

            # 高度位置编码器
            self.pos_h_encoding = nn.Sequential(
                nn.Linear(1, pos_dim),
                nn.LayerNorm(pos_dim),
                nn.GELU(),
                nn.Linear(pos_dim, pos_dim)
            )

            # 宽度位置编码器
            self.pos_w_encoding = nn.Sequential(
                nn.Linear(1, pos_dim),
                nn.LayerNorm(pos_dim),
                nn.GELU(),
                nn.Linear(pos_dim, pos_dim)
            )

            # 类别令牌位置编码
            self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, dim))

        # 创建Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
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
            nn.Linear(mlp_input_dim, mlp_dim),
            nn.GELU(),
            nn.BatchNorm1d(mlp_dim),
            nn.Linear(mlp_dim, num_classes)
        )

        # 保存注意力权重用于可视化
        self.attention_weights = None

    def generate_3d_positional_embedding(self, d, h, w, dim):
        """
        生成3D感知的位置编码
        参数:
            d: 深度维度补丁数
            h: 高度维度补丁数
            w: 宽度维度补丁数
            dim: 编码维度
        返回:
            形状为(1, d*h*w, dim)的位置编码张量
        """
        # 生成归一化的坐标
        d_pos = torch.arange(d).float().unsqueeze(1).unsqueeze(1) / d
        h_pos = torch.arange(h).float().unsqueeze(1).unsqueeze(0) / h
        w_pos = torch.arange(w).float().unsqueeze(0).unsqueeze(0) / w

        # 转换为设备匹配的张量
        device = next(self.parameters()).device
        d_pos = d_pos.to(device)
        h_pos = h_pos.to(device)
        w_pos = w_pos.to(device)

        # 编码每个维度
        pos_d = self.pos_d_encoding(d_pos.reshape(-1, 1)).reshape(d, 1, 1, dim // 3)
        pos_h = self.pos_h_encoding(h_pos.reshape(-1, 1)).reshape(1, h, 1, dim // 3)
        pos_w = self.pos_w_encoding(w_pos.reshape(-1, 1)).reshape(1, 1, w, dim // 3)

        # 结合三维位置信息
        pos_embedding = torch.cat([
            pos_d.expand(-1, h, w, -1),  # 扩展到所有位置
            pos_h.expand(d, -1, w, -1),
            pos_w.expand(d, h, -1, -1)
        ], dim=-1)  # 将三个维度的编码拼接

        # 将形状从(d, h, w, dim)转为(1, d*h*w, dim)
        pos_embedding = pos_embedding.reshape(1, d * h * w, dim)

        return pos_embedding

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, channels, depth, height, width)
            output_attention: 是否返回注意力权重用于可视化
        """


        x = x.float()
        batch_size = x.shape[0]
        input_shape = x.shape

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
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        if self.use_3d_pos_embed:
            # 生成3D位置编码
            patch_pos_embed = self.generate_3d_positional_embedding(
                d_patches, h_patches, w_patches, self.embed_dim
            )
            # 确保位置编码与补丁数匹配
            if patch_pos_embed.shape[1] != actual_patches:
                # 如果不匹配，使用插值调整大小
                patch_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.transpose(1, 2),
                    size=actual_patches,
                    mode='linear'
                ).transpose(1, 2)

            # 拼接类别令牌位置编码
            pos_embed = torch.cat([self.cls_pos_embed, patch_pos_embed], dim=1)
            # 添加位置编码
            x = x + pos_embed
        else:
            # 使用原始位置编码
            x = x + self.orig_pos_embedding

        # Dropout
        x = self.dropout(x)
        # Transformer编码器
        x = self.transformer(x)  # (batch_size, num_patches + 1, dim)

        # 提取分类令牌和补丁令牌
        cls_token_out = x[:, 0]  # (batch_size, dim)
        patch_tokens = x[:, 1:]  # (batch_size, num_patches, dim)

        # 根据池化方法聚合特征
        if self.pool == 'mean':
            pooled_patches = torch.mean(patch_tokens, dim=1)
            combined_features = torch.cat([cls_token_out, pooled_patches], dim=1)
        elif self.pool == 'max':
            pooled_patches = torch.max(patch_tokens, dim=1)[0]
            combined_features = torch.cat([cls_token_out, pooled_patches], dim=1)
        else:  # pool == 'all'
            combined_features = x.reshape(batch_size, -1)

        # 最终分类头
        out = self.mlp_head(combined_features)
        return out

    def load_pretrained_dino(self, path):
        """加载预训练的DINOv2权重"""
        try:
            basic_model = Dinov2ForImageClassification.from_pretrained(
                path,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
            self.load_state_dict(basic_model.state_dict(), strict=False)
            print("成功加载DINOv2预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
