import torch
import torch.nn as nn
import timm
from timm.models import VisionTransformer
from transformers import Dinov2ForImageClassification


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
                 pool='mean',  # Pooling method: 'mean', 'max' or 'all'
                 ):  # Whether to load DINOv2-pretrained weights
        super().__init__(img_size=image_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=dim, depth=depth, num_heads=heads)


        self.pool = pool

        # Calculate number of patches for 3D image (cube)
        patch_dim = patch_size
        num_patches = (image_size // patch_dim) ** 3
        patch_size_3d = (patch_dim, patch_dim, patch_dim)
        self.num_patches = num_patches

        # Create 3D patch embedding with Conv3d layer
        self.patch_embedding = nn.Conv3d(1, dim,
                                         kernel_size=patch_size_3d,
                                         stride=patch_size_3d)

        # Position embedding and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer Encoder
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

        # MLP head input dimension based on pooling method
        if pool == 'mean':
            mlp_input_dim = dim * 2  # CLS token + mean pooled tokens
        elif pool == 'max':
            mlp_input_dim = dim * 2  # CLS token + max pooled tokens
        elif pool == 'all':
            mlp_input_dim = dim * (num_patches + 1)  # All tokens concatenated
        else:
            raise ValueError(f"Unsupported pooling method: {pool}")

        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )


    def load_pretrained_dino(self, model, path):
        """
        Load pre-trained DINOv2 weights into the transformer encoder.
        This function uses timm to load the DINOv2 model ("vit_base_dino")
        and copies the weights from its transformer block to the current model.
        Note: Only the transformer encoder weights are loaded. The patch embedding,
        positional embedding and the MLP head remain uninitialized from pretrained weights.
        """
        # Load DINOv2 pre-trained model using timm (make sure timm is installed)
        basic_model = Dinov2ForImageClassification.from_pretrained(path, num_labels=self.num_classes, ignore_mismatched_sizes=True)
        model.load_state_dict(basic_model.state_dict(), False)
    def forward(self, x):
        """
        Forward pass:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        """
        x = x.float()
        batch_size = x.shape[0]

        # Generate 3D patch embeddings
        x = self.patch_embedding(x)  # (batch_size, dim, d', h', w')
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # Append class token to patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding and dropout
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x)  # (batch_size, num_patches + 1, dim)

        # Extract CLS token and process other tokens
        cls_token_out = x[:, 0]  # (batch_size, dim)
        patch_tokens = x[:, 1:]  # (batch_size, num_patches, dim)

        if self.pool == 'mean':
            pooled_patches = torch.mean(patch_tokens, dim=1)
            combined_features = torch.cat([cls_token_out, pooled_patches], dim=1)
        elif self.pool == 'max':
            pooled_patches = torch.max(patch_tokens, dim=1)[0]
            combined_features = torch.cat([cls_token_out, pooled_patches], dim=1)
        else:  # pool == 'all'
            combined_features = x.reshape(batch_size, -1)

        # Final classification head
        out = self.mlp_head(combined_features)
        return out
