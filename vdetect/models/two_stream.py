from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import WavLMModel

from .wavlm_baseline import AttentiveStatsPool
from .aasist_encoder import AASISTEncoder, pad_or_crop_for_aasist
from .blocks import FiLM


WAVLM_EMBED_DIM = 1536           # WavLM (768) after attentive stats pooling.
AASIST_EMBED_DIM = 160           # AASIST last_hidden.
FUSED_EMBED_DIM = WAVLM_EMBED_DIM + AASIST_EMBED_DIM  # 1696
AASIST_MAX_SAMPLES = 64600       # AASIST is hard-coded to this length.


class TwoStreamDetector(nn.Module):
    # Late-fusion detector: WavLM (1536-d) + AASIST (160-d) -> MLP, with optional FiLM on the fused vector.

    def __init__(
        self,
        wavlm_model: str = "microsoft/wavlm-base-plus",
        wavlm_checkpoint: Optional[Union[str, Path]] = None,
        aasist_checkpoint: Optional[Union[str, Path]] = None,
        freeze_wavlm: bool = True,
        freeze_aasist: bool = True,
        trainable_wavlm_layers: int = 0,
    ):
        super().__init__()

        # WavLM stream
        self.wavlm_backbone = WavLMModel.from_pretrained(wavlm_model)
        hidden_dim = self.wavlm_backbone.config.hidden_size
        self.wavlm_pool = AttentiveStatsPool(hidden_dim)

        if freeze_wavlm:
            for param in self.wavlm_backbone.parameters():
                param.requires_grad = False
            if trainable_wavlm_layers > 0 and hasattr(self.wavlm_backbone, "encoder"):
                for layer in self.wavlm_backbone.encoder.layers[-trainable_wavlm_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        if wavlm_checkpoint is not None:
            self._load_wavlm_weights(wavlm_checkpoint)

        # AASIST stream
        self.aasist = AASISTEncoder(pretrained_path=aasist_checkpoint, freeze=freeze_aasist)

        # FiLM modulates the fused vector using a speaker prototype.
        self.film = FiLM(feature_dim=FUSED_EMBED_DIM, condition_dim=FUSED_EMBED_DIM)

        self.fusion_head = nn.Sequential(
            nn.Linear(FUSED_EMBED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _load_wavlm_weights(self, checkpoint_path: Union[str, Path]) -> None:
        # Reuse the attentive-pool weights from a trained WavLM baseline checkpoint.
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"WavLM checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu").get("model", {})
        pool_state_dict = {
            key[len("pool."):]: value
            for key, value in state_dict.items()
            if key.startswith("pool.")
        }
        if pool_state_dict:
            self.wavlm_pool.load_state_dict(pool_state_dict)
            print(f"Loaded WavLM pooling weights from {checkpoint_path}")
        else:
            print(f"Warning: No pooling weights found in {checkpoint_path}")

    def _prepare_aasist_input(self, wav: torch.Tensor) -> torch.Tensor:
        return pad_or_crop_for_aasist(wav, target_length=AASIST_MAX_SAMPLES, train=self.training)

    def forward(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_proto: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(self.training):
            wavlm_features = self.wavlm_backbone(wav, attention_mask=attention_mask).last_hidden_state

        wavlm_emb = self.wavlm_pool(wavlm_features)
        aasist_emb = self.aasist(self._prepare_aasist_input(wav))
        fused = torch.cat([wavlm_emb, aasist_emb], dim=-1)

        if speaker_proto is not None:
            fused = self.film(fused, speaker_proto)

        logit = self.fusion_head(fused).squeeze(-1)
        return {"logit": logit, "score": torch.sigmoid(logit)}

    def extract_embedding(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            wavlm_features = self.wavlm_backbone(wav, attention_mask=attention_mask).last_hidden_state
            wavlm_emb = self.wavlm_pool(wavlm_features)
            aasist_emb = self.aasist(self._prepare_aasist_input(wav))
            return torch.cat([wavlm_emb, aasist_emb], dim=-1)

    def extract_branch_embeddings(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            wavlm_features = self.wavlm_backbone(wav, attention_mask=attention_mask).last_hidden_state
            wavlm_emb = self.wavlm_pool(wavlm_features)
            aasist_emb = self.aasist(self._prepare_aasist_input(wav))
            fused = torch.cat([wavlm_emb, aasist_emb], dim=-1)
        return {"wavlm": wavlm_emb, "aasist": aasist_emb, "fused": fused}

    def get_trainable_params(self) -> list:
        # Splits trainable params into groups so the optimiser can use different LR scales.
        fusion_params = list(self.fusion_head.parameters()) + list(self.film.parameters())
        pool_params = list(self.wavlm_pool.parameters())

        other_params = [
            param for name, param in self.named_parameters()
            if param.requires_grad
            and "fusion_head" not in name
            and "film" not in name
            and "wavlm_pool" not in name
        ]

        return [
            {"params": fusion_params, "lr_scale": 1.0},
            {"params": pool_params, "lr_scale": 0.5},
            {"params": other_params, "lr_scale": 0.1},
        ]


def create_two_stream_model(
    wavlm_checkpoint: Optional[str] = "assets/checkpoints/wavlm_baseline.pt",
    aasist_checkpoint: Optional[str] = None,
    freeze_backbones: bool = True,
    device: str = "cpu",
) -> TwoStreamDetector:
    model = TwoStreamDetector(
        wavlm_checkpoint=wavlm_checkpoint,
        aasist_checkpoint=aasist_checkpoint,
        freeze_wavlm=freeze_backbones,
        freeze_aasist=freeze_backbones,
    )
    return model.to(device)

