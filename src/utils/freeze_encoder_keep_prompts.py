def freeze_encoder_keep_prompts(model):
    """冻结编码器参数，只保留CLS token和类原型向量可训练"""
    # 默认冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻CLS token
    model.cls_token.requires_grad = True

    # 解冻类原型向量和位置编码
    if hasattr(model, 'class_prompts'):
        model.class_prompts.requires_grad = True
    if hasattr(model, 'prompt_pos_embedding'):
        model.prompt_pos_embedding.requires_grad = True

    # 可选：保持分类头可训练
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    # 打印可训练参数数量及名称
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"可训练参数总数: {total_params}")
    print(f"可训练参数列表: {trainable_params}")