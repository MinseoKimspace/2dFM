from __future__ import annotations

from dataclasses import dataclass
from turtle import hideturtle
from types import NotImplementedType
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.seed = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        learnable_s = self.seed.repeat(batch_size, 1, 1)
        return self.mab(learnable_s, x)

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
        self.early_pma = PMA(dim, pool_heads, early_num_seeds)
        self.late_pma = PMA(dim, pool_heads, late_num_seeds)
        self.early_norm = nn.LayerNorm(code_dim)
        self.late_norm = nn.LayerNorm(code_dim)
        self.early_proj = nn.Linear(early_num_seeds * dim, code_dim)
        self.late_proj = nn.Linear(late_num_seeds * dim, code_dim)

    def merge_group(
        self,
        hidden_states: list[torch.Tensor],
        indices: Sequence[int],
    ) -> torch.Tensor:
        result = []
        for i in indices:
            result.append(hidden_states[i])
        
        return torch.concat(result, 1)

    def pool_group(
        self,
        group_tokens: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        if level == "early":
            return self.early_pma(group_tokens)
        elif level == "late":
            return self.late_pma(group_tokens)
        else:
            raise ValueError

    def project_group(
        self,
        slots: torch.Tensor,
        level: str,
    ) -> torch.Tensor:
        if level == "early":
            return self.early_norm(self.early_proj(torch.flatten(slots, 1)))
        elif level == "late":
            return self.late_norm(self.late_proj(torch.flatten(slots, 1)))
        else:
            raise ValueError

    def forward(
        self,
        hidden_states: list[torch.Tensor],
    ) -> DualLevelCodes:
        early_tokens = self.merge_group(hidden_states, self.early_indices)
        late_tokens = self.merge_group(hidden_states, self.late_indices)
        early_slots = self.pool_group(early_tokens, "early")
        late_slots = self.pool_group(late_tokens, "late")
        early_code = self.project_group(early_slots, "early")
        late_code = self.project_group(late_slots, "late")
        dual_level_codes = DualLevelCodes(early_tokens, late_tokens, early_slots, late_slots, early_code, late_code)
        return dual_level_codes


class SemanticConsistencyHead(nn.Module):
    def __init__(
        self,
        code_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.early_proj1 = nn.Linear(code_dim, hidden_dim)
        self.early_proj2 = nn.Linear(hidden_dim, code_dim)
        self.late_proj1 = nn.Linear(code_dim, hidden_dim)
        self.late_proj2 = nn.Linear(hidden_dim, code_dim)
        self.early_pred1 = nn.Linear(code_dim, hidden_dim)
        self.early_pred2 = nn.Linear(hidden_dim, code_dim)

    def forward(
        self,
        early_code: torch.Tensor,
        late_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        early_shared = self.early_proj1(early_code)
        early_shared = F.relu(early_shared)
        early_shared = self.early_proj2(early_shared)
        
        late_shared = self.late_proj1(late_code)
        late_shared = F.relu(late_shared)
        late_shared = self.late_proj2(late_shared)

        early_pred = self.early_pred1(early_shared)
        early_pred = F.relu(early_pred)
        early_pred = self.early_pred2(early_pred)

        return early_shared, late_shared, early_pred

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

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
    ) -> DualLevelOutput:
        split_layer = max(self.pooling_head.early_indices) + 1

        tokens = self.backbone.prepare_tokens(x_t, t_start, t_now)
        
        early_out = self.backbone.encode_tokens(
            tokens,
            return_hidden_states= True,
            start_layer=0,
            end_layer=split_layer,
            cond=None,
        )
        early_hidden = early_out.hidden_states
        if early_hidden is None:
            raise ValueError()
        
        early_tokens = self.pooling_head.merge_group(early_hidden, self.pooling_head.early_indices)
        early_slots = self.pooling_head.pool_group(early_tokens, "early")
        early_code = self.pooling_head.project_group(early_slots, "early")

        late_out = self.backbone.encode_tokens(
            early_out.tokens,
            return_hidden_states=True,
            start_layer=split_layer,
            end_layer=None,
            cond=early_code,
        )
        late_hidden = late_out.hidden_states
        if late_hidden is None:
            raise ValueError
        
        late_local_indices = [i - split_layer for i in self.pooling_head.late_indices]
        late_tokens = self.pooling_head.merge_group(late_hidden, late_local_indices)
        late_slots = self.pooling_head.pool_group(late_tokens, "late")
        late_code = self.pooling_head.project_group(late_slots, "late")

        pooled = DualLevelCodes(
            early_tokens=early_tokens,
            late_tokens=late_tokens,
            early_slots=early_slots,
            late_slots=late_slots,
            early_code=early_code,
            late_code=late_code
        )
        early_shared, late_shared, early_pred = self.consistency_head(
            early_code,
            late_code,
        )
        sample = self.backbone.decode_tokens(late_out.tokens)
        full_hidden_states = early_hidden + late_hidden
        dual_level_output = DualLevelOutput(sample, full_hidden_states, pooled, early_shared, late_shared, early_pred)
        
        return dual_level_output

def semantic_consistency_loss(
    early_pred: torch.Tensor,
    late_shared: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(early_pred, late_shared.detach())

def collapse_regularization_loss(
    early_shared: torch.Tensor,
    late_shared: torch.Tensor,
) -> torch.Tensor:
    early = torch.max(torch.zeros_like(early_shared[-1]), 1 - torch.sqrt(torch.var(early_shared, 0) + 1e-5)).mean()
    late = torch.max(torch.zeros_like(early_shared[-1]), 1 - torch.sqrt(torch.var(late_shared, 0) + 1e-5)).mean()
    return early + late