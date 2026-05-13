import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

# Make the vendored AASIST repo importable so we can reuse the official model class.
AASIST_REPO_PATH = Path(__file__).parent.parent.parent / "aasist"
if str(AASIST_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(AASIST_REPO_PATH))

from models.AASIST import Model as AASISTModel  # noqa: E402


# Default config copied verbatim from aasist/config/AASIST.conf.
DEFAULT_AASIST_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# Embedding dim = 5 * gat_dims[1] = 160 (T_max, T_avg, S_max, S_avg, master node).
AASIST_EMBED_DIM = 160


class AASISTEncoder(nn.Module):
    # Wraps the official AASIST model and exposes its 160-d embedding for fusion.

    def __init__(
        self,
        pretrained_path: Optional[Union[str, Path]] = None,
        config: Optional[dict] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.config = config or DEFAULT_AASIST_CONFIG.copy()
        self.embed_dim = AASIST_EMBED_DIM
        self.model = AASISTModel(self.config)

        if pretrained_path is None:
            pretrained_path = AASIST_REPO_PATH / "models" / "weights" / "AASIST.pth"
        self._load_pretrained(pretrained_path)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _load_pretrained(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AASIST pretrained weights not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded AASIST pretrained weights from {path}")

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # AASIST returns (last_hidden, logits); we want just the embedding.
        last_hidden, _ = self.model(wav)
        return last_hidden

    def forward_with_logits(self, wav: torch.Tensor) -> tuple:
        return self.model(wav)

    def get_score(self, wav: torch.Tensor) -> torch.Tensor:
        _, output = self.model(wav)
        # AASIST was trained with bonafide=1, spoof=0, so column 0 is the spoof probability.
        return torch.softmax(output, dim=-1)[:, 0]


def pad_or_crop_for_aasist(
    wav: torch.Tensor,
    target_length: int = 64600,
    train: bool = False,
) -> torch.Tensor:
    # AASIST expects exactly 64600 samples. Random crop during training, centre crop otherwise.
    import random

    squeeze_output = False
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze_output = True

    _, current_length = wav.shape

    if current_length > target_length:
        if train:
            start = random.randint(0, current_length - target_length)
        else:
            start = (current_length - target_length) // 2
        wav = wav[:, start:start + target_length]
    elif current_length < target_length:
        wav = torch.nn.functional.pad(wav, (0, target_length - current_length))

    if squeeze_output:
        wav = wav.squeeze(0)
    return wav

