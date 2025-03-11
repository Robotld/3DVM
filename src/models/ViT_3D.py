import torch
import torch.nn as nn
import timm
from timm.models import VisionTransformer
from transformers import Dinov2ForImageClassification


class ViT3D(nn.Module):
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
                 center_crop=None,
                 in_chans=1,
                 pool='max'):
        super().__init__()

        self.pool = pool
        self.num_classes = num_classes

        # Calculate number of patches for 3D image (cube)
        patch_dim = patch_size
        if center_crop:
            if isinstance(center_crop, int):
                image_size = center_crop
            else:
                image_size = center_crop[0]
        print(image_size)
        num_patches_per_dim = image_size // patch_dim  # Number of patches in each dimension
        self.num_patches = num_patches_per_dim ** 3  # Total number of patches (D×H×W)
        print("num_patches:", self.num_patches)
        patch_size_3d = (patch_dim, patch_dim, patch_dim)

        print(f"Initializing ViT3D with {self.num_patches} patches (patch size {patch_size}, image size {image_size})")

        # Create 3D patch embedding with Conv3d layer
        self.patch_embedding = nn.Sequential(
            nn.Conv3d(in_chans, dim,
                      kernel_size=patch_size_3d,
                      stride=patch_size_3d),
            nn.LayerNorm([dim, num_patches_per_dim, num_patches_per_dim, num_patches_per_dim]),
            nn.GELU()
        )

        # Position embedding and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
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
        if pool == 'mean' or pool == 'max':
            mlp_input_dim = dim * 2  # CLS token + pooled tokens
        elif pool == 'all':
            mlp_input_dim = dim * (self.num_patches + 1)  # All tokens concatenated
        else:
            raise ValueError(f"Unsupported pooling method: {pool}")

        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    # 建议：实现3D感知的位置编码
    def generate_3d_positional_embedding(self, d, h, w, dim):
        """生成3D感知的位置编码"""
        d_pos = torch.arange(d).float().unsqueeze(1).unsqueeze(1) / d
        h_pos = torch.arange(h).float().unsqueeze(1).unsqueeze(0) / h
        w_pos = torch.arange(w).float().unsqueeze(0).unsqueeze(0) / w

        pos_d = self.pos_d_encoding(d_pos)  # (d, 1, 1, dim//3)
        pos_h = self.pos_h_encoding(h_pos)  # (1, h, 1, dim//3)
        pos_w = self.pos_w_encoding(w_pos)  # (1, 1, w, dim//3)

        # 结合三维位置信息
        pos_embedding = torch.cat([
            pos_d.expand(-1, h, w, -1),
            pos_h.expand(d, -1, w, -1),
            pos_w.expand(d, h, -1, -1)
        ], dim=-1)  # (d, h, w, dim)

        return pos_embedding.flatten(0, 2).unsqueeze(0)  # (1, d*h*w, dim)
    def load_pretrained_dino(self, path):
        """
        Load pre-trained DINOv2 weights into the transformer encoder.
        """
        try:
            basic_model = Dinov2ForImageClassification.from_pretrained(path, num_labels=self.num_classes,
                                                                       ignore_mismatched_sizes=True)
            self.load_state_dict(basic_model.state_dict(), False)
            print("Loading pretrained DINOv2 weights...")
        except Exception as e:
            print(f"Failed to load pretrained weights: {str(e)}")

    def forward(self, x):
        """
        Forward pass:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        """
        x = x.float()
        batch_size = x.shape[0]

        # Print input shape for debugging
        input_shape = x.shape

        # Generate 3D patch embeddings
        x = self.patch_embedding(x)  # (batch_size, dim, d', h', w')

        # After patch embedding shape
        after_patch_shape = x.shape

        # Flatten patches to sequence
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # The actual number of patches
        actual_patches = x.shape[1]

        # Verify patch count matches expectation
        if actual_patches != self.num_patches:
            print(f"WARNING: Expected {self.num_patches} patches but got {actual_patches}. " +
                  f"Input shape: {input_shape}, After patch embedding: {after_patch_shape}")

            # Dynamically adjust pos_embedding to match the actual number of patches
            # This is a runtime fix - ideally, patch calculation should be corrected
            pos_emb = self.pos_embedding
            cls_pos_emb = pos_emb[:, 0:1, :]
            patch_pos_emb = pos_emb[:, 1:, :]

            # Interpolate to match the actual number of patches
            if actual_patches != self.num_patches:
                patch_pos_emb = nn.functional.interpolate(
                    patch_pos_emb.transpose(1, 2),
                    size=actual_patches,
                    mode='linear'
                ).transpose(1, 2)

                self.pos_embedding = nn.Parameter(torch.cat([cls_pos_emb, patch_pos_emb], dim=1))
                print(f"Adjusted position embedding from {self.num_patches + 1} to {actual_patches + 1} tokens")
                self.num_patches = actual_patches

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