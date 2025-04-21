import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """基于预训练中文BERT的病理报告编码器，支持提示向量"""

    def __init__(self, bert_model_name="hfl/chinese-roberta-wwm-ext",
                 output_dim=384, dropout=0.1, prompt_num=8, embedding_dim=384):
        """
        参数:
            bert_model_name: 预训练BERT模型名称或路径
            output_dim: 输出特征维度
            dropout: Dropout比率
            prompt_num: 提示向量数量
            embedding_dim: 提示向量维度
        """
        super().__init__()

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)

        # BERT输出特征维度（通常为768）
        bert_hidden_dim = self.bert.config.hidden_size

        # 添加类别提示向量
        self.text_prompts = nn.Parameter(torch.randn(1, prompt_num, embedding_dim))

        # 特征映射层（可选）
        self.feature_projection = None
        if output_dim != bert_hidden_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(bert_hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout)
            )

        # 创建对应的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs（可选）
        返回:
            text_features: 文本特征
            text_prompt_features: 提示向量特征
        """
        batch_size = input_ids.shape[0]

        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # 获取[CLS]标记的表示作为文本特征
        text_features = outputs.last_hidden_state[:, 0]

        # 应用特征映射（如果定义了）
        if self.feature_projection is not None:
            text_features = self.feature_projection(text_features)

        # 扩展提示向量到批次大小
        text_prompt_features = self.text_prompts.expand(batch_size, -1, -1)

        return text_features, text_prompt_features