from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.transformer import TransformerOutput, VectorFieldTransformer


@dataclass
class DualLevelCodes:
    early_tokens: torch.Tensor
    late_tokens: torch.Tensor
    early_slots: torch.Tensor
    late_slots: torch.Tensor
    early_code: torch.Tensor
    late_code: torch.Tensor


@dataclass
class DualLevelOutput:
    sample: torch.Tensor
    hidden_states: list[torch.Tensor]
    pooled: DualLevelCodes
    early_shared: torch.Tensor
    late_shared: torch.Tensor
    early_pred: torch.Tensor


class MAB(nn.Module):
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
        self.ff_mult = ff_mult
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_o = nn.Linear(dim, dim)
        self.mid = nn.Linear(dim, dim * ff_mult)
        self.end = nn.Linear(dim * ff_mult, dim) 
        self.dropout = nn.Dropout(dropout)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        
        assert self.dim % self.num_heads == 0

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        Q = self.fc_q(q)
        K, V = self.fc_k(x), self.fc_v(x)
        

        dim_split = self.dim // self.num_heads
        
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))


        identity = O
        O = self.dropout(O)
        O = self.ln1(O)
        O = self.mid(O)
        O = F.relu(O)
        O = self.end(O)
        O = O + identity        
        return O


class PMA(nn.Module):
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
        raise NotImplementedError("Implement PMA using learnable seeds and MAB.")


class DualLevelPoolingHead(nn.Module):
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
        raise NotImplementedError("Implement layer grouping and optional layer embeddings.")

    def pool_group(
        self,
        group_tokens: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement PMA call for the requested level.")

    def project_group(
        self,
        slots: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement slot flattening/norm/projection.")

    def forward(
        self,
        hidden_states: list[torch.Tensor],
    ) -> DualLevelCodes:
        raise NotImplementedError("Implement dual-level early/late pooling.")


class SemanticConsistencyHead(nn.Module):
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
    raise NotImplementedError("Implement stop-grad semantic matching loss.")


def collapse_regularization_loss(
    early_shared: torch.Tensor,
    late_shared: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("Implement variance/covariance or related regularization.")
