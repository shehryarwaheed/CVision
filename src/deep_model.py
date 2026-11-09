from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CVisionDualBranchNet(nn.Module):
    """
    Dual-Branch Neural Network for Resume Representation and Category Classification.
    Fuses high-dimensional text representations with domain-engineered document quality metrics.
    Includes an ablation flag (use_num_branch) to evaluate the incremental gain of structural features.
    """
    def __init__(
            self,
            tfidf_dim: int = 5000,
            num_features: int = 13,
            num_classes: int = 24,
            hidden_text_dim: int = 256,
            hidden_num_dim: int = 32,
            hidden_fusion_dim: int = 128,
            dropout_rate: float = 0.4,
            use_num_branch: bool = True,
    ):
        super().__init__()

        self.tfidf_dim = tfidf_dim
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_num_branch = use_num_branch

        # -------------------------------------------------------------
        # Branch 1: High-Dimensional Text Vocabulary Branch (TF-IDF)
        # -------------------------------------------------------------
        self.text_branch = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_text_dim),
            nn.BatchNorm1d(hidden_text_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # -------------------------------------------------------------
        # Branch 2: Structural Document Quality Metrics Branch
        # Numeric dropout is capped at 0.20 to preserve dense signals.
        # -------------------------------------------------------------
        if self.use_num_branch:
            self.num_branch = nn.Sequential(
                nn.Linear(num_features, hidden_num_dim),
                nn.BatchNorm1d(hidden_num_dim),
                nn.ReLU(),
                nn.Dropout(p=min(0.20, dropout_rate * 0.50))
            )
            fused_dim = hidden_text_dim + hidden_num_dim  # e.g., 256 + 32 = 288
        else:
            self.num_branch = None
            fused_dim = hidden_text_dim  # Ablation mode: text only (256)

        # -------------------------------------------------------------
        # Multimodal Feature Fusion & Classification Head
        # Fusion dropout is scaled to 0.75x of base dropout.
        # -------------------------------------------------------------
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, hidden_fusion_dim),
            nn.BatchNorm1d(hidden_fusion_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.75)
        )

        self.classifier_out = nn.Linear(hidden_fusion_dim, num_classes)

        # Explicit parameter initialization
        self._init_weights()

    def _init_weights(self):
        """Standard Kaiming (He) normal for ReLU layers; Xavier for linear projection."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.classifier_out:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_embeddings(self, x_text: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """Extracts the learned 128-d dense representation vector before final projection."""
        h_text = self.text_branch(x_text)

        if self.use_num_branch and self.num_branch is not None:
            h_num = self.num_branch(x_num)
            h_fused = torch.cat([h_text, h_num], dim=1)
        else:
            h_fused = h_text

        latent_embedding = self.fusion_layer(h_fused)
        return latent_embedding

    def forward(self, x_text: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.
        Directly compatible with nn.CrossEntropyLoss(weight=class_weights).
        """
        latent = self.extract_embeddings(x_text, x_num)
        logits = self.classifier_out(latent)
        return logits

    @torch.no_grad()
    def predict_proba(self, x_text: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized Softmax probabilities.
        Safe: Preserves and restores model.training state to prevent silent BatchNorm/Dropout corruption.
        """
        was_training = self.training
        self.eval()
        logits = self.forward(x_text, x_num)
        probs = F.softmax(logits, dim=-1)
        self.train(was_training)
        return probs

    def parameter_summary(self) -> Dict[str, int]:
        """Provides a breakdown of parameter counts across modules for the technical report."""
        text_params = sum(p.numel() for p in self.text_branch.parameters())
        num_params = sum(p.numel() for p in self.num_branch.parameters()) if self.use_num_branch else 0
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        out_params = sum(p.numel() for p in self.classifier_out.parameters())
        total = text_params + num_params + fusion_params + out_params

        return {
            "text_branch": text_params,
            "num_branch": num_params,
            "fusion_layer": fusion_params,
            "classifier_out": out_params,
            "total_parameters": total
        }


if __name__ == "__main__":
    # Test batch verification
    batch_size = 4
    dummy_text = torch.randn(batch_size, 5000)
    dummy_num = torch.randn(batch_size, 13)

    model = CVisionDualBranchNet(
        tfidf_dim=5000,
        num_features=13,
        num_classes=24,
        use_num_branch=True
    )

    # 1. State preservation check for predict_proba
    model.train()
    assert model.training is True
    probs = model.predict_proba(dummy_text, dummy_num)
    assert model.training is True, "CRITICAL: predict_proba must restore model.training mode!"
    assert probs.shape == (batch_size, 24)

    # 2. Ablation test (text-only)
    ablation_model = CVisionDualBranchNet(use_num_branch=False)
    ablation_out = ablation_model(dummy_text, dummy_num)
    assert ablation_out.shape == (batch_size, 24)

    # 3. Print parameter summary for LaTeX report
    summary = model.parameter_summary()
    print("=" * 65)
    print("CVisionDualBranchNet Verified Successfully!")
    print(f"Text Branch (5000 -> 256):     {summary['text_branch']:,} params ({summary['text_branch']/summary['total_parameters']:.1%})")
    print(f"Num Branch (13 -> 32):         {summary['num_branch']:,} params")
    print(f"Fusion Layer (288 -> 128):     {summary['fusion_layer']:,} params")
    print(f"Output Head (128 -> 24):       {summary['classifier_out']:,} params")
    print(f"Total Trainable Parameters:    {summary['total_parameters']:,} params")
    print("=" * 65)