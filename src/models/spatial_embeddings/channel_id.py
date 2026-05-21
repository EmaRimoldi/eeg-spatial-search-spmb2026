"""
Variant B: Channel-ID embedding.

Learned embedding per canonical channel identity (fixed vocabulary).
This is the common approach in models like LaBraM.

Maps channel names → integer indices → learned embedding vectors.
Channels not in the vocabulary receive a zero embedding (with warning).
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase

# Standard 10-20/10-10 channel vocabulary (73 channels)
# This should match the canonical channel names in the coordinate lookup table.
_DEFAULT_CHANNEL_VOCAB = [
    # Frontal polar
    "Fp1", "Fp2", "FPz",
    # Anterior frontal
    "AF7", "AF3", "AFz", "AF4", "AF8",
    # Frontal
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    # Fronto-temporal / fronto-central
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    # Central / temporal
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    # Temporo-parietal / centro-parietal
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    # Parietal
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    # Parieto-occipital
    "PO7", "PO3", "POz", "PO4", "PO8",
    # Occipital
    "O1", "Oz", "O2", "I1", "Iz", "I2",
    # References
    "A1", "A2", "M1", "M2", "Nz",
    # Unknown token
    "<UNK>",
]


class ChannelIDEmbedding(SpatialEmbeddingBase):
    """
    Learned embedding per canonical channel identity.

    Every canonical channel name has its own embedding vector.
    Channels outside the vocabulary are mapped to an <UNK> embedding.

    This is the standard approach in fixed-vocabulary models like LaBraM.
    Its key limitation: it cannot handle truly novel electrode positions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: Optional[int] = None,
        channel_vocab: Optional[list[str]] = None,
    ):
        """
        Args:
            embed_dim: Embedding output dimension.
            num_channels: Ignored if channel_vocab is provided.
            channel_vocab: Ordered list of canonical channel names.
                          Default: standard 10-20/10-10 vocabulary.
        """
        super().__init__(embed_dim=embed_dim, name="channel_id")

        if channel_vocab is None:
            channel_vocab = _DEFAULT_CHANNEL_VOCAB

        self.channel_vocab = channel_vocab
        self.vocab_size = len(channel_vocab)
        self._name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(channel_vocab)
        }
        self._unk_idx = self._name_to_idx.get("<UNK>", self.vocab_size - 1)

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def _names_to_indices(
        self,
        channel_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Map channel names to embedding indices."""
        indices = []
        for name in channel_names:
            idx = self._name_to_idx.get(name, self._unk_idx)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, C, embed_dim).
        """
        C = len(channel_names)
        indices = self._names_to_indices(channel_names, device)  # (C,)
        emb = self.embedding(indices)  # (C, embed_dim)

        # Expand to batch dimension: (B, C, embed_dim)
        emb = emb.unsqueeze(0).expand(batch_size, -1, -1)
        return emb

    def known_channels(self) -> set[str]:
        """Return the set of channel names in the vocabulary."""
        return set(self._name_to_idx.keys()) - {"<UNK>"}
