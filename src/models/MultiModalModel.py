import torch
import torch.nn as nn
import torch.nn.functional as F
from .TextEncoder import TextEncoder

# -----------------------------------------------------------
# 局部图文 Cross-Attention 对齐模块（原样保留）
# -----------------------------------------------------------
class CrossModalAttentionBlock(nn.Module):
    def __init__(self, image_dim, text_dim, num_heads=7):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_dim, num_heads=num_heads,
            kdim=text_dim, vdim=text_dim, batch_first=True)

    def forward(self, image_tokens, keyword_tokens):
        # image_tokens: [B, N1, D]   keyword_tokens: [B, N2, D']
        out, attn_w = self.cross_attention(
            query=image_tokens, key=keyword_tokens, value=keyword_tokens)
        return out, attn_w


class CrossModalFusionModule(nn.Module):
    """关键词-图像局部对齐 + 三模态全局融合（原样保留）"""
    def __init__(self, fusion_dim, num_heads=4):
        super().__init__()
        self.image_to_keyword_align = CrossModalAttentionBlock(
            image_dim=fusion_dim, text_dim=fusion_dim, num_heads=num_heads)

        self.global_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)

        self.out_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(0.5))

        self.CLS_fusion = nn.Parameter(torch.randn(1, fusion_dim))

    def forward(self, image_tokens, keyword_tokens, text_cls):
        # 1) 图像 token attend 关键词
        B = image_tokens.size(0)
        img_aligned, _ = self.image_to_keyword_align(image_tokens, keyword_tokens)

        # 2) 构造三模态 token 序列
        cls_token = self.CLS_fusion.expand(B, -1)
        glob_tokens = torch.stack([cls_token, img_aligned, text_cls], dim=1)  # [B, 3, D]

        # 3) 全局 cross-attention
        fused, _ = self.global_attn(glob_tokens, glob_tokens, glob_tokens)     # [B, 3, D]
        return fused[:, 0]                                                     # 取 CLS
# ------------------------------------------------------------------
#                    MultimodalMultitaskModel
# ------------------------------------------------------------------
class MultimodalMultitaskModel(nn.Module):
    """
    多模态肺癌复发预测 + 亚型分类 + 跨模态对齐对比损失
    """
    def __init__(self,
                 vit_3d_model,
                 bert_model_name="hfl/chinese-roberta-wwm-ext",
                 image_dim=384,
                 text_feature_dim=384,
                 demographic_dim=4,
                 fusion_dim=384,
                 num_classes_recurrence=2,
                 num_classes_subtype=3,
                 dropout=0.5,
                 fusion_method="attention",
                 # ↓ 新增参数
                 align_dim=128,          # 对比学习共享维度
                 contrastive_tau=0.07):  # 温度系数
        super().__init__()

        # --- 编码器 --------------------------------------------------
        self.image_encoder = vit_3d_model
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            output_dim=text_feature_dim,
            dropout=0.2)

        # --- 人口学投影 ---------------------------------------------
        self.demographic_projection = nn.Sequential(
            nn.Linear(demographic_dim, 128),
            nn.LayerNorm(128),
            nn.Sigmoid())

        # --- 融合 ----------------------------------------------------
        self.fusion_method = fusion_method
        if fusion_method == 'attention':
            self.image_proj   = nn.Linear(image_dim * 2, fusion_dim)
            self.keyword_proj = nn.Linear(text_feature_dim, fusion_dim)
            self.fusion_module = CrossModalFusionModule(
                fusion_dim=fusion_dim, num_heads=3)

        # --- 主任务分类头 --------------------------------------------
        feature_len = fusion_dim + 128   # 融合CLS + 人口学
        self.recurrence_classifier = nn.Sequential(
            nn.Linear(feature_len, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes_recurrence))

        self.subtype_classifier = nn.Sequential(
            nn.Linear(feature_len, 512),
            nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes_subtype))

        # ========= 新增：对比对齐模块 =========
        self.fused_proj   = nn.Linear(fusion_dim, align_dim)
        self.text2_proj   = nn.Linear(text_feature_dim, align_dim)
        self.contrastive_tau = contrastive_tau
        # =====================================

    def freeze_encoders(self, freeze_image_encoder=False, freeze_text_encoder=False):
        """冻结编码器主干，但保持提示向量可训练"""
        # 冻结ViT3D编码器
        if freeze_image_encoder:
            # 冻结除class_prompts和cls_token外的所有参数
            for name, param in self.image_encoder.named_parameters():
                if 'class_prompts' not in name and 'cls_token' not in name:
                    param.requires_grad = False
                    print(f"已冻结: {name}")
                else:
                    print(f"保持可训练: {name}")

        # 冻结BERT编码器
        if freeze_text_encoder:
            for name, param in self.text_encoder.named_parameters():
                if 'mlp_head' not in name:
                    param.requires_grad = False

    # ---------------------------------------------------------------
    # 对比损失 (InfoNCE，双向)
    # ---------------------------------------------------------------
    def _contrastive_loss(self, x, y):
        """
        x: [B, D]  fused_features
        y: [B, D]  enhanced_text_features2
        """
        B = x.size(0)
        x = F.normalize(self.fused_proj(x), dim=-1)   # [B, D']
        y = F.normalize(self.text2_proj(y), dim=-1)   # [B, D']

        logits = x @ y.T / self.contrastive_tau       # [B, B]
        labels = torch.arange(B, device=x.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_xy + loss_yx)

    # ---------------------------------------------------------------
    def forward(self, image, text_inputs1, text_inputs2,
                demographic, keywords=None):
        # 1) 图像特征
        img_out = self.image_encoder(image)
        img_feat = img_out[1]          # [B, 2*image_dim] (假设 ViT3D 返回 (cls, patch))

        # 2) 文本编码 (CT 报告)
        enh_text1, kw_feat1, ori_text1 = self.text_encoder(
            input_ids=text_inputs1['input_ids'],
            attention_mask=text_inputs1['attention_mask'],
            keywords=keywords)

        # 3) 文本编码 (病理报告)
        enh_text2, kw_feat2, ori_text2 = self.text_encoder(
            input_ids=text_inputs2['input_ids'],
            attention_mask=text_inputs2['attention_mask'],
            keywords=keywords)

        # 4) 人口学
        demo_feat = self.demographic_projection(demographic)

        # 5) 图文 Cross-Attention 对齐  (CT 报告 <-> 图像)
        img_aligned   = self.image_proj(img_feat)
        kw_aligned    = self.keyword_proj(kw_feat1 if kw_feat1 is not None
                                          else torch.zeros_like(enh_text1))
        fused_cls = self.fusion_module(
            image_tokens=img_aligned,
            keyword_tokens=kw_aligned,
            text_cls=enh_text1
        )

        # 6) 拼接人口学
        fused_features = torch.cat([fused_cls, demo_feat], dim=1)   # [B, fusion_dim+128]

        # 7) 主任务预测
        recurrence_logits = self.recurrence_classifier(fused_features)
        subtype_logits    = self.subtype_classifier(fused_features)

        # 8) 提示相似度正则 (如有)
        similarity_loss = 0.0
        if hasattr(self.image_encoder, 'prompt_cosine_similarity_loss'):
            similarity_loss += self.image_encoder.prompt_cosine_similarity_loss(0.3, 0.7)

        # 9) ====== 关键新增：跨模态对比损失 ======
        contrastive_loss = self._contrastive_loss(fused_cls, enh_text2)
        # ========================================

        # 10) 返回
        return recurrence_logits, subtype_logits, similarity_loss, contrastive_loss, torch.cat([img_feat, ori_text1, demo_feat], dim=1)

