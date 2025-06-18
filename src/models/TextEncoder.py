import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# class TextEncoder(nn.Module):
#     """基于预训练中文BERT的病理报告编码器，支持提示向量"""
#
#     def __init__(self, bert_model_name=r"E:\workplace\3D\pred_model\bert_chinese",
#                  output_dim=384, dropout=0.1, prompt_num=8, embedding_dim=384):
#         """
#         参数:
#             bert_model_name: 预训练BERT模型名称或路径
#             output_dim: 输出特征维度
#             dropout: Dropout比率
#             prompt_num: 提示向量数量
#             embedding_dim: 提示向量维度
#         """
#         super().__init__()
#
#         # 加载预训练BERT模型
#         self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
#
#         # BERT输出特征维度（通常为768）
#         bert_hidden_dim = self.bert.config.hidden_size
#
#         # 添加类别提示向量
#         self.text_prompts = nn.Parameter(torch.randn(1, prompt_num, embedding_dim))
#
#         # 特征映射层（可选）
#         self.feature_projection = None
#         if output_dim != bert_hidden_dim:
#             self.feature_projection = nn.Sequential(
#                 nn.Linear(bert_hidden_dim, output_dim),
#                 nn.LayerNorm(output_dim),
#                 nn.Dropout(dropout)
#             )
#
#         # 创建对应的tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#
#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         """
#         前向传播
#         参数:
#             input_ids: 输入token IDs
#             attention_mask: 注意力掩码
#             token_type_ids: token类型IDs（可选）
#         返回:
#             text_features: 文本特征
#             text_prompt_features: 提示向量特征
#         """
#         # 获取BERT输出
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#
#         # 获取[CLS]标记的表示作为文本特征
#         text_features = outputs.last_hidden_state[:, 0]
#
#         return text_features


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer


class TextEncoder(nn.Module):
    """基于预训练中文BERT的病理报告编码器，使用预提取的关键词列表"""

    def __init__(self, bert_model_name=r"E:\workplace\3D\pred_model\bert_chinese",
                 output_dim=384, dropout=0.1):
        """
        参数:
            bert_model_name: 预训练BERT模型名称或路径
            output_dim: 输出特征维度
            dropout: Dropout比率
        """
        super().__init__()

        self.bert_model_name = bert_model_name
        self.output_dim = output_dim

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)
        # 创建对应的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # 线性层和Dropout层，用于处理BERT输出
        self.feature_fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 关键词注意力机制
        self.key_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, input_ids, attention_mask, keywords=None, token_type_ids=None):
        """
        前向传播
        参数:
            input_ids: 输入token IDs (批处理)
            attention_mask: 注意力掩码 (批处理)
            raw_texts: 原始文本字符串列表
            keywords: 预提取的关键词列表 [B, num_keywords, 2]，每个元素是(关键词, 值)对
            token_type_ids: token类型IDs（可选）
        返回:
            enhanced_text_features: 增强后的文本特征 [B, output_dim]
            keyword_features: 关键词特征 [B, output_dim]
            original_text_features: 原始文本特征 [B, output_dim]
        """
        # 1. 获取主文本的BERT输出及[CLS]特征
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # 获取[CLS]标记的表示作为文本特征
        cls_output = outputs.last_hidden_state[:, 0]
        original_text_features = self.feature_fc(cls_output)  # [B, output_dim]

        # 如果没有提供关键词，直接返回原始文本特征
        if keywords is None:
            return original_text_features, None, original_text_features

        # 2. 编码关键词-值对
        batch_size = input_ids.size(0)
        keyword_features_batch = []

        # print(keywords)
        for i in range(batch_size):
            # 当前样本的关键词列表
            sample_keywords = keywords[i]  # [(keyword1, value1), (keyword2, value2), ...]
            # print(sample_keywords)
            if len(sample_keywords) == 0:
                # 如果没有关键词，使用零向量
                keyword_features_batch.append(torch.zeros_like(original_text_features[i]))
                continue

            # 将关键词-值对合并为文本，形如 "关键词1: 值1; 关键词2: 值2; ..."
            keyword_text = "; ".join([str(item) for item in sample_keywords])
            # 对关键词文本进行编码
            keyword_inputs = self.tokenizer(
                keyword_text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )

            # 将输入移到与BERT相同的设备
            keyword_inputs = {k: v.to(self.bert.device) for k, v in keyword_inputs.items()}
            with torch.no_grad():
                keyword_outputs = self.bert(**keyword_inputs)

            # 获取关键词的[CLS]特征
            keyword_cls = keyword_outputs.last_hidden_state[:, 0]
            keyword_feature = self.feature_fc(keyword_cls).squeeze(0)  # [output_dim]
            keyword_features_batch.append(keyword_feature)

        # 3. 堆叠所有样本的关键词特征
        if keyword_features_batch:
            keyword_features = torch.stack(keyword_features_batch)  # [B, output_dim]
        else:
            keyword_features = torch.zeros_like(original_text_features)

        # 4. 关键词和原始文本之间的交叉注意力
        # 调整维度以适应注意力机制输入: [B, 1, D]
        text_query = original_text_features.unsqueeze(1)
        keyword_key_value = keyword_features.unsqueeze(1)

        # 应用注意力机制
        attn_output, _ = self.key_attention(
            query=text_query,
            key=keyword_key_value,
            value=keyword_key_value
        )  # [B, 1, D]

        # 5. 特征融合: 原始文本特征 + 注意力处理后的特征
        attn_output = attn_output.squeeze(1)  # [B, D]
        # enhanced_text_features = self.fusion_layer(
        #     torch.cat([original_text_features, attn_output], dim=1)
        # )  # [B, D]
        enhanced_text_features = (original_text_features+attn_output)/2
        return enhanced_text_features, keyword_features, original_text_features
