import torch
import torch.nn as nn
import torch.nn.functional as F
from .TextEncoder import TextEncoder


class MultimodalMultitaskModel(nn.Module):
    """多模态肺癌复发预测与亚型分类多任务模型"""

    def __init__(
            self,
            vit_3d_model,  # 已有的ViT3D模型
            bert_model_name="hfl/chinese-roberta-wwm-ext",
            image_dim = 384,
            text_feature_dim=384,  # 修改为实际的文本特征维度384
            demographic_dim=2,  # 年龄和性别
            fusion_dim=384,  # 选择合适的融合维度
            num_classes_recurrence=2,
            num_classes_subtype=3,
            dropout=0.3,
            prompt_num=8,
            fusion_method="attention"  # 'attention', 'concat', 'sum'
    ):
        super().__init__()

        # 图像编码器（使用已有的ViT3D）
        self.image_encoder = vit_3d_model

        # 文本编码器（包含提示向量）
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,  # 确保与实际维度匹配
            dropout=dropout,
            prompt_num=prompt_num,
            embedding_dim=text_feature_dim
        )

        self.image_feature_dim = image_dim

        # 融合方法
        self.fusion_method = fusion_method

        # 人口学特征映射
        self.demographic_projection = nn.Sequential(
            nn.Linear(demographic_dim, 32),  # 从2维映射到适度的32维
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # 决定融合后的特征维度
        if fusion_method == "concat":
            final_dim = fusion_dim * 3 + 32  # 图像 + 文本 + 人口学
        else:
            final_dim = fusion_dim

        # 复发预测分类头
        self.recurrence_classifier = nn.Sequential(
            nn.Linear(final_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes_recurrence)
        )

        # 亚型分类头
        self.subtype_classifier = nn.Sequential(
            nn.Linear(final_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes_subtype)
        )

    def freeze_encoders(self):
        """冻结编码器主干，但保持提示向量可训练"""
        # # 冻结ViT3D编码器
        # for name, param in self.image_encoder.named_parameters():
        #     if 'class_prompts' not in name:  # 不冻结图像提示向量
        #         param.requires_grad = False

        # 冻结BERT编码器
        for name, param in self.text_encoder.named_parameters():
            if 'text_prompts' not in name:  # 不冻结文本提示向量
                param.requires_grad = False

    def forward(self, image, text_inputs, demographic):
        """
        前向传播
        参数:
            image: CT图像输入
            text_inputs: 文本输入字典，包含input_ids, attention_mask等
            demographic: 人口学特征，[年龄, 性别]
        """
        # 调试信息，可在故障排除时使用
        # print(f"Image shape: {image.shape}")

        # 1. 图像特征提取
        image_outputs = self.image_encoder(image)

        # 确保获取正确的图像特征
        # 根据实际情况选择合适的索引
        image_features = image_outputs[1]  # 获取特征向量

        # 2. 文本特征提取
        text_features, text_prompt_features = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs.get('token_type_ids', None)
        )
        #
        # # 调试信息
        # print(f"Text features shape: {text_features.shape}")
        # print(f"Demographic shape: {demographic.shape}")

        # 3. 人口学特征处理
        demographic_features = self.demographic_projection(demographic)
        # 调试信息
        # print(f"Aligned dimensions - Image: {aligned_image.shape}, Text: {aligned_text.shape}, Demo: {aligned_demographic.shape}")
        # 5. 特征融合
        if self.fusion_method == "concat":
            # 简单拼接
            fused_features = torch.cat([
                image_features, text_features, demographic_features
            ], dim=1)


        # 6. 多任务预测
        recurrence_logits = self.recurrence_classifier(fused_features)
        subtype_logits = self.subtype_classifier(fused_features)

        # 计算提示向量相似度损失（用于正则化）
        similarity_loss = 0.0
        if hasattr(self.image_encoder, 'prompt_cosine_similarity_loss'):
            similarity_loss += self.image_encoder.prompt_cosine_similarity_loss(0.3, 0.7)

        # # 添加文本提示向量的相似度损失
        # text_prompt_flat = text_prompt_features.view(text_prompt_features.size(0), -1)
        # if text_prompt_flat.size(0) > 1:
        #     text_prompt_norm = F.normalize(text_prompt_flat, p=2, dim=1)
        #     cosine_sim = torch.mm(text_prompt_norm, text_prompt_norm.t())
        #     mask = ~torch.eye(cosine_sim.size(0), dtype=torch.bool, device=cosine_sim.device)
        #     cosine_sim = cosine_sim[mask].view(cosine_sim.size(0), -1)
        #
        #     # 应用软约束
        #     text_similarity_loss = torch.mean(torch.clamp(cosine_sim - 0.7, min=0) ** 2 +
        #                                       torch.clamp(0.3 - cosine_sim, min=0) ** 2)
        #     similarity_loss += text_similarity_loss

        return recurrence_logits, subtype_logits, similarity_loss