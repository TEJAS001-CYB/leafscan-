"""
model.py
--------
EfficientNetB3-based leaf disease classifier.
  - Pretrained on ImageNet for strong generalization
  - Custom classification head
  - Supports feature extraction + fine-tuning phases
"""

import torch
import torch.nn as nn
import timm


class LeafDiseaseModel(nn.Module):
    """
    EfficientNetB3 with custom classification head for leaf disease detection.

    Architecture:
        EfficientNetB3 backbone (ImageNet pretrained)
        → Global Average Pooling
        → BatchNorm → Dense(512) → GELU → Dropout(0.4)
        → BatchNorm → Dense(256) → GELU → Dropout(0.3)
        → Dense(num_classes) → Softmax
    """

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,          # remove default head
            global_pool="avg",
        )
        backbone_out = self.backbone.num_features   # 1536 for B3

        # ── Custom Head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.BatchNorm1d(backbone_out),
            nn.Linear(backbone_out, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation for the head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Phase control ─────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze backbone — only train the head (Phase 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone frozen — training head only.")

    def unfreeze_backbone(self, unfreeze_layers: int = 30):
        """
        Unfreeze the last N backbone layers for fine-tuning (Phase 2).
        EfficientNetB3 has ~360 parameters groups; last 30 covers blocks 5-7.
        """
        all_params = list(self.backbone.parameters())
        # First, freeze everything
        for p in all_params:
            p.requires_grad = False
        # Then unfreeze last N
        for p in all_params[-unfreeze_layers:]:
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  Unfrozen last {unfreeze_layers} backbone param groups "
              f"({trainable:,} params now trainable).")

    def unfreeze_all(self):
        """Fully unfreeze everything (Phase 3)."""
        for p in self.parameters():
            p.requires_grad = True
        print("  All layers unfrozen.")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)     # (B, 1536)
        logits   = self.head(features)  # (B, num_classes)
        return logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor):
        """Return (class_idx, confidence) tuple."""
        probs = self.get_probabilities(x)
        conf, idx = torch.max(probs, dim=-1)
        return idx, conf

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        return total, trainable


def build_model(num_classes: int, pretrained: bool = True) -> LeafDiseaseModel:
    """Factory function — builds and returns the model."""
    model = LeafDiseaseModel(num_classes=num_classes, pretrained=pretrained)
    return model


if __name__ == "__main__":
    m = build_model(39)
    m.count_parameters()
    x = torch.randn(4, 3, 300, 300)
    out = m(x)
    print(f"  Output shape: {out.shape}")   # (4, 39)
