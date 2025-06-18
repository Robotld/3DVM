"""
多模态肺癌复发预测与亚型分类多任务训练脚本（支持聚类交叉验证）
整合CT图像、病理报告和人口学特征进行端到端训练
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, Resized, NormalizeIntensityd, ToTensord
)

# 导入自定义模块
from models import create_transforms, MulCrossValidator, ViT3D
from models import MultimodalRecurrenceDataset
from models import MultimodalMultitaskModel
from utils import update_config_from_args, parse_args, ConfigManager, set_seed
from train_multimodal import train


def extract_features(dataset, model, device, batch_size=2):
    """
    从数据集中提取特征

    参数:
        dataset: 包含图像、文本和人口统计数据的数据集
        model: 用于特征提取的多模态模型
        device: 计算设备
        batch_size: 批次大小

    返回:
        features: 特征矩阵 [n_samples, n_features]
        valid_indices: 成功提取特征的样本索引
    """
    print("\n开始提取多模态特征用于聚类...")

    def custom_collate_fn(batch):
        """自定义批处理函数，保持关键词原有结构"""
        # 提取关键词列表
        keywords = [item.pop('keywords') for item in batch]

        # 批处理其他字段
        from torch.utils.data.dataloader import default_collate
        batched_data = default_collate(batch)

        # 不经过转换，直接放回批次数据
        batched_data['keywords'] = keywords

        return batched_data

    # 使用DataLoader以批次处理
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 避免多进程问题
        pin_memory=True,
        collate_fn = custom_collate_fn
    )

    features_list = []
    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"特征提取进度: {batch_idx + 1}/{len(loader)}")

            try:
                # 跳过无效批次
                if batch_data is None or any(item is None for item in batch_data):
                    print(f"批次 {batch_idx} 包含无效样本，跳过")
                    continue

                # 将数据移到设备
                images = batch_data["image"].to(device)
                text_inputs = {k: v.to(device) for k, v in batch_data["text"].items()}
                demographics = batch_data["demographic"].to(device)
                report_text = batch_data["report_text"]
                keywords = batch_data["keywords"]  # 获取关键词列表

                # 提取特征（使用forward方法的返回值中的all_futures）
                recurrence_logits, subtype_logits, similarity_loss, all_features = model(
                    image=images,
                    text_inputs=text_inputs,
                    demographic=demographics,
                    report_text=report_text,
                    keywords=keywords
                )

                # 保存特征
                features_list.append(all_features.cpu().numpy())

                # 记录当前批次中的样本索引
                batch_indices = range(
                    batch_idx * batch_size,
                    min(batch_idx * batch_size + len(images), len(dataset))
                )
                valid_indices.extend(batch_indices)

            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    if not features_list:
        raise ValueError("没有成功提取任何特征，请检查数据预处理流程")

    # 合并所有特征
    all_features = np.vstack(features_list)
    print(f"成功提取 {len(valid_indices)} 个样本的特征，特征维度: {all_features.shape}")

    return all_features, valid_indices


def main():
    # 记录开始时间
    start_time = time.time()

    # 加载配置
    config = ConfigManager('config/config_multimodal.yaml')

    # 解析命令行参数
    args = parse_args(config)
    config = update_config_from_args(config, args)

    set_seed(config.training['random_seed'])

    # 优化CUDA性能
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    print(f"使用设备: {config.training['device']}")

    # 创建输出目录
    os.makedirs(config.data['output_dir'], exist_ok=True)

    # 根据交叉验证类型命名输出目录
    cv_type = "clustering" if config.cross_validation.get('use_clustering', False) else "stratified"
    train_dir = os.path.join(
        config.data['output_dir'],
        f'train_multimodal_{cv_type}_{time.strftime("%Y%m%d_%H%M%S")}'
    )
    os.makedirs(train_dir, exist_ok=True)

    # 创建MONAI转换管道
    train_transforms, val_transforms = create_transforms(config, args)

    # 初始化BERT分词器
    bert_model_name = config.model['tokenizer_path']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # 创建数据集
    print("加载多模态数据集...")
    dataset = MultimodalRecurrenceDataset(
        csv_path=config.data['csv_path'],
        root_dirs=config.data['root_dirs'],
        transform=None,
        text_tokenizer=tokenizer,
        max_length=config.data.get('max_text_length', 512)
    )

    # 显示数据集基本信息
    print(f"数据集样本总数: {len(dataset.samples)}")

    # 初始化交叉验证器
    cv = MulCrossValidator(dataset, config)
    device = config.training['device']

    #如果使用聚类交叉验证，先提取特征并进行聚类
    if config.cross_validation.get('use_clustering', False):
        print("使用聚类交叉验证，进行特征提取...")

        # 创建临时模型用于特征提取
        vit3d_config = {
            "num_classes": config.data["num_classes"],
            "image_size": config.model["params"]["image_size"],
            "crop_size": config.training["crop_size"],
            "patch_size": config.model["params"]["patch_size"],
            "dim": config.model["params"]["dim"],
            "depth": config.model["params"]["depth"],
            "heads": config.model["params"]["heads"],
            "mlp_dim": config.model["params"]["mlp_dim"],
            "pool": config.model["params"]["pool"],
            'cpt_num': config.model["params"]["cpt_num"],
            'mlp_num': config.model["params"]["mlp_num"],
        }

        # 1. 初始化图像编码器（ViT3D）
        vit3d_model = ViT3D(**vit3d_config)

        # 加载ViT3D预训练权重
        if config.training['pretrained_path']:
            print(f"加载ViT3D预训练权重: {config.training['pretrained_path']}...")
            try:
                if os.path.isdir(config.training['pretrained_path']):
                    vit3d_model.load_pretrained_dino(config.training['pretrained_path'])
                else:
                    vit3d_model.load_state_dict(torch.load(config.training['pretrained_path'], map_location=device), strict=False)
                print(f"成功加载ViT3D权重")
            except Exception as e:
                print(f"加载ViT3D预训练权重出错: {str(e)}")

        # 2. 构建多模态模型（用于特征提取）
        feature_extractor = MultimodalMultitaskModel(
            vit_3d_model=vit3d_model,
            image_dim=config.model["params"]["dim"],
            bert_model_name=config.model['bert_model_name'],
            text_feature_dim=config.model['text_feature_dim'],
            demographic_dim=4,  # 年龄和性别
            fusion_dim=config.model['fusion_dim'],
            num_classes_recurrence=config.data['num_classes'],
            num_classes_subtype=config.data['num_classes_subtype'],
            dropout=config.model['dropout'],
            prompt_num=config.model['prompt_num'],
            fusion_method=config.model['fusion_method']
        )

        feature_extractor.to(device)
        dataset1 = MultimodalRecurrenceDataset(
            csv_path=config.data['csv_path'],
            root_dirs=config.data['root_dirs'],
            transform=val_transforms,
            text_tokenizer=tokenizer,
            max_length=config.data.get('max_text_length', 512),
        )

        # 提取特征
        features, valid_indices = extract_features(
            dataset=dataset1,
            model=feature_extractor,
            device=device,
            batch_size=config.cross_validation.get('feature_extraction_batch_size', 32)
        )

        # np.save(os.path.join(train_dir, "all_features.npy"),features)
        # # 2. 保存对应的样本信息（如ct_number）
        # sample_ids = [dataset.samples[i]['ct_number'] for i in valid_indices]
        # np.save(os.path.join(train_dir, "valid_ct_numbers.npy"), sample_ids)
        #
        # # 3. 可选：保存标签
        # labels = [dataset.samples[i]['recurrence_label'] for i in valid_indices]
        # np.save(os.path.join(train_dir, "recurrence_labels.npy"), labels)
        # 设置聚类分折
        cv.setup_clustering_splits(features, valid_indices)

        # 释放特征提取器内存
        del feature_extractor, vit3d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 存储每折的最佳F1, AUC分数
    fold_scores = []

    # 预加载ViT3D模型配置
    vit3d_config = {
        "num_classes": config.data["num_classes"],
        "image_size": config.model["params"]["image_size"],
        "crop_size": config.training["crop_size"],
        "patch_size": config.model["params"]["patch_size"],
        "dim": config.model["params"]["dim"],
        "depth": config.model["params"]["depth"],
        "heads": config.model["params"]["heads"],
        "mlp_dim": config.model["params"]["mlp_dim"],
        "pool": config.model["params"]["pool"],
        'cpt_num': config.model["params"]["cpt_num"],
        'mlp_num': config.model["params"]["mlp_num"],
    }

    best_f1, best_auc = 0.0, 0.0

    # 进行交叉验证
    for fold, train_loader, val_loader, train_counter in cv.get_folds(train_transforms, val_transforms):
        # # 记录每个fold的数据计数器，用于计算类别权重
        # if fold < 1 or fold > 4:
        #     continue
        train_loader.dataset._data_counter = train_counter

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_start_time = time.time()
        fold_type = "聚类折" if config.cross_validation.get('use_clustering', False) else "分层折"
        print(f'\n训练第 {fold + 1} 折 ({fold_type})')

        # 1. 初始化图像编码器（ViT3D）
        vit3d_model = ViT3D(**vit3d_config)

        # 加载ViT3D预训练权重
        if config.training['pretrained_path']:
            print(f"Loading pretrained weights for ViT3D from {config.training['pretrained_path']}...")
            try:
                if os.path.isdir(config.training['pretrained_path']):
                    vit3d_model.load_pretrained_dino(config.training['pretrained_path'])
                else:
                    vit3d_model.load_state_dict(torch.load(config.training['pretrained_path'], map_location=device), strict=False)
                print(f"成功加载ViT3D权重: {config.training['pretrained_path']}")
            except Exception as e:
                print(f"加载ViT3D预训练权重出错: {str(e)}")

        # 2. 构建多模态模型
        model = MultimodalMultitaskModel(
            vit_3d_model=vit3d_model,
            image_dim=config.model["params"]["dim"],
            bert_model_name=config.model['bert_model_name'],
            text_feature_dim=config.model['text_feature_dim'],
            demographic_dim=4,  # 年龄和性别 结节大小 是否多发
            fusion_dim=config.model['fusion_dim'],
            num_classes_recurrence=config.data['num_classes'],
            num_classes_subtype=config.data['num_classes_subtype'],
            dropout=config.model['dropout'],
            fusion_method=config.model['fusion_method']
        )

        model.to(device)

        model.freeze_encoders(freeze_image_encoder=args.frozen1, freeze_text_encoder=args.frozen2)

        # 打印可训练参数数量及名称
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 训练模型
        f1, auc, f1_model, auc_model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            args=args,
            warmup_epochs=config.training["warmup_epochs"],
            fold=fold,
            train_dir=train_dir,
            best_f1=0,
            best_auc=0
        )

        if best_auc <= auc:
            best_auc = auc
            best_auc_model = auc_model
            torch.save(best_auc_model, f'{train_dir}/best_auc_model_fold_{fold+1}.pth')

        if best_f1 <= f1:
            best_f1 = f1
            best_f1_model = f1_model
            torch.save(best_f1_model, f'{train_dir}/best_f1_model_fold_{fold+1}.pth')

        fold_scores.append((f1, auc))
        fold_time = time.time() - fold_start_time
        print(f'Fold {fold + 1} completed in {fold_time:.2f}s with F1: {f1:.4f} AUC: {auc:.4f}')

        # 释放模型内存
        del model, vit3d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 输出交叉验证结果
    fold_scores = np.array(fold_scores)
    cv_type_name = "聚类交叉验证" if config.cross_validation.get('use_clustering', False) else "分层交叉验证"
    print(f'\n{cv_type_name}结果:')
    for fold, score in enumerate(fold_scores):
        print(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}')
    mean_f1 = np.mean(fold_scores[:, 0])
    std_f1 = np.std(fold_scores[:, 0])
    mean_auc = np.mean(fold_scores[:, 1])
    std_auc = np.std(fold_scores[:, 1])
    print(f'Mean F1: {mean_f1:.4f} ± {std_f1:.4f}\n'
          f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')

    cv_results_path = os.path.join(train_dir, 'cv_results.txt')
    with open(cv_results_path, 'w') as f:
        f.write(f"{cv_type_name}结果:\n")
        for fold, score in enumerate(fold_scores):
            f.write(f'Fold {fold + 1}: F1 {score[0]:.4f}   AUC {score[1]:.4f}\n')
        f.write(f'\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}\n')
        f.write(f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n')
        f.write(f'\n交叉验证类型: {cv_type_name}\n')

    # 打印总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")


if __name__ == '__main__':
    main()