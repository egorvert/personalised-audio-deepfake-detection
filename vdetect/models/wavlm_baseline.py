from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import WavLMModel


class AttentiveStatsPool(nn.Module):
    # Attention-weighted mean + std across the time axis.

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.attn(x), dim=1)
        mean = torch.sum(a * x, dim=1)
        var = torch.sum(a * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-5)
        return torch.cat([mean, std], dim=-1)


class WavLMDetector(nn.Module):
    # WavLM backbone + attentive stats pooling + 2-layer MLP for spoof detection.

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        trainable_layers: int = 0,
    ):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Optionally unfreeze the last N transformer layers for fine-tuning.
            if trainable_layers > 0 and hasattr(self.backbone, "encoder"):
                for layer in self.backbone.encoder.layers[-trainable_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        hidden_dim = self.backbone.config.hidden_size
        self.pool = AttentiveStatsPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(self.training):
            features = self.backbone(wav, attention_mask=attention_mask).last_hidden_state

        x = self.pool(features)
        logit = self.head(x).squeeze(1)
        return {"logit": logit, "score": torch.sigmoid(logit)}

    def extract_embedding(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(wav, attention_mask=attention_mask).last_hidden_state
            return self.pool(features)
