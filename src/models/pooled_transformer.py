"""Skeleton interfaces for dual-level pooled Transformer variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from models.transformer import TransformerOutput, VectorFieldTransformer


@dataclass
class DualLevelCodes:
    """Container for pooled early/late representations."""

    early_tokens: torch.Tensor
    late_tokens: torch.Tensor
    early_slots: torch.Tensor
    late_slots: torch.Tensor
    early_code: torch.Tensor
    late_code: torch.Tensor


@dataclass
class DualLevelOutput:
    """Structured output for a self-guided pooled Transformer wrapper."""

    sample: torch.Tensor
    hidden_states: list[torch.Tensor]
    pooled: DualLevelCodes
    early_shared: torch.Tensor
    late_shared: torch.Tensor
    early_pred: torch.Tensor


class MAB(nn.Module):
    """Skeleton for a multihead attention block used by PMA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_mult = ff_mult

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return attended query states with shape `[B, S, D]`."""
        raise NotImplementedError("Implement MAB attention + FFN update.")


class PMA(nn.Module):
    """Skeleton for pooling by multihead attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token set `x` into seed slots with shape `[B, S, D]`."""
        raise NotImplementedError("Implement PMA using learnable seeds and MAB.")


class DualLevelPoolingHead(nn.Module):
    """Skeleton for grouped early/late layer pooling."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        pool_heads: int,
        early_indices: Sequence[int],
        late_indices: Sequence[int],
        early_num_seeds: int = 4,
        late_num_seeds: int = 1,
        code_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.pool_heads = pool_heads
        self.early_indices = tuple(early_indices)
        self.late_indices = tuple(late_indices)
        self.early_num_seeds = early_num_seeds
        self.late_num_seeds = late_num_seeds
        self.code_dim = code_dim

    def merge_group(
        self,
        hidden_states: list[torch.Tensor],
        indices: Sequence[int],
    ) -> torch.Tensor:
        """Merge selected hidden states into one grouped token set."""
        raise NotImplementedError("Implement layer grouping and optional layer embeddings.")

    def pool_group(
        self,
        group_tokens: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        """Pool one layer group into slots."""
        raise NotImplementedError("Implement PMA call for the requested level.")

    def project_group(
        self,
        slots: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        """Project pooled slots into a compact code vector."""
        raise NotImplementedError("Implement slot flattening/norm/projection.")

    def forward(
        self,
        hidden_states: list[torch.Tensor],
    ) -> DualLevelCodes:
        raise NotImplementedError("Implement dual-level early/late pooling.")


class SemanticConsistencyHead(nn.Module):
    """Skeleton for projection and predictor heads used in semantic alignment."""

    def __init__(
        self,
        code_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim or code_dim

    def forward(
        self,
        early_code: torch.Tensor,
        late_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return `(early_shared, late_shared, early_pred)`.
        """
        raise NotImplementedError("Implement projector and predictor heads.")


class DualLevelSelfGuidedTransformer(nn.Module):
    """Skeleton wrapper around the backbone and dual-level heads."""

    def __init__(
        self,
        backbone: VectorFieldTransformer,
        pooling_head: DualLevelPoolingHead,
        consistency_head: SemanticConsistencyHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_head = pooling_head
        self.consistency_head = consistency_head

    def extract_hidden_states(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
    ) -> TransformerOutput:
        """Call the backbone with hidden-state extraction enabled."""
        output = self.backbone(
            x_t,
            t_start,
            t_now,
            return_hidden_states=True,
            return_tokens=True,
            return_dict=True,
        )
        if not isinstance(output, TransformerOutput):
            raise TypeError("Backbone must return TransformerOutput when return_dict=True.")
        return output

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
    ) -> DualLevelOutput:
        raise NotImplementedError("Implement pooled extraction, projection, and structured return.")


def semantic_consistency_loss(
    early_pred: torch.Tensor,
    late_shared: torch.Tensor,
) -> torch.Tensor:
    """Skeleton for the feature-level semantic consistency loss."""
    raise NotImplementedError("Implement stop-grad semantic matching loss.")


def collapse_regularization_loss(
    early_shared: torch.Tensor,
    late_shared: torch.Tensor,
) -> torch.Tensor:
    """Skeleton for collapse-prevention regularization."""
    raise NotImplementedError("Implement variance/covariance or related regularization.")
