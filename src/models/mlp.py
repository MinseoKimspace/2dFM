"""FM/iMF 벡터 필드 예측을 위한 공용 MLP 백본."""

from __future__ import annotations

# torch는 Tensor 연산을 담당한다.
import torch
# nn은 Linear, Sequential 같은 학습 가능한 모듈을 제공한다.
import torch.nn as nn

# TimeEmbeddingMLP는 스칼라 시간을 학습 가능한 임베딩으로 바꾼다.
from models.time_embed import TimeEmbeddingMLP


def _init_linear_conservative(module: nn.Module) -> None:
    """Linear 층을 작은 가우시안으로 보수적으로 초기화한다."""
    # Linear 층만 건드리고 나머지 모듈은 그대로 둔다.
    if isinstance(module, nn.Linear):
        # fan_in은 입력 차원 수다.
        fan_in = module.weight.size(1)
        # 초기 activation을 작게 유지하기 위한 표준편차다.
        std = (0.1 / float(fan_in)) ** 0.5
        # 평균 0의 작은 가우시안으로 weight를 초기화한다.
        nn.init.normal_(module.weight, mean=0.0, std=std)
        # bias는 0에서 시작시킨다.
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VectorFieldMLP(nn.Module):
    """FM의 v 또는 iMF의 u를 예측하는 MLP."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        variant: str = "fm",
        position_embed_dim: int | None = None,
    ) -> None:
        # 학습 가능한 벡터 필드 모델로 등록한다.
        super().__init__()
        # 출력 head 타입은 fm / imf 중 하나여야 한다.
        if variant not in {"fm", "imf"}:
            raise ValueError(f"Unknown variant: {variant}")

        # 현재 모델이 예측할 필드 종류를 저장한다.
        self.variant = variant
        # 위치 임베딩 차원을 따로 주지 않으면 입력 차원과 같게 둔다.
        pos_dim = int(position_embed_dim) if position_embed_dim is not None else int(input_dim)

        # 현재 상태 x_t를 한 번 더 표현 공간으로 옮기는 위치 임베딩 MLP.
        self.position_embed = nn.Sequential(
            nn.Linear(input_dim, pos_dim),
            nn.SiLU(),
            nn.Linear(pos_dim, pos_dim),
        )
        # 시작 시간 r을 임베딩하는 MLP.
        self.start_time_embed = TimeEmbeddingMLP(time_embed_dim)
        # 현재 시간 t를 임베딩하는 MLP.
        self.now_time_embed = TimeEmbeddingMLP(time_embed_dim)

        # MLP 본체에 들어갈 입력 차원은 위치 + 시작시간 + 현재시간 임베딩의 합이다.
        layers = []
        in_dim = pos_dim + time_embed_dim + time_embed_dim
        # 요청한 층 수만큼 Linear -> SiLU -> Dropout 블록을 반복한다.
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            # 첫 층 이후에는 hidden_dim을 계속 유지한다.
            in_dim = hidden_dim
        # 반복해서 쌓은 블록을 하나의 Sequential backbone으로 묶는다.
        self.backbone = nn.Sequential(*layers)

        # FM에서는 순간 속도 v를 출력한다.
        self.v_head = nn.Linear(hidden_dim, input_dim)
        # iMF에서는 평균 속도 u를 출력한다.
        self.u_head = nn.Linear(hidden_dim, input_dim)
        # 모델 안의 모든 Linear 층에 보수적 초기화를 적용한다.
        self.apply(_init_linear_conservative)
        # iMF는 초반 학습이 흔들릴 수 있어 u-head를 아주 작게 시작시킨다.
        if self.variant == "imf":
            nn.init.normal_(self.u_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.u_head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        # 예전 호출 방식 forward(x_t, t_now)도 지원하기 위해 t_start를 0으로 보정한다.
        if t_now is None:
            t_now = t_start
            t_start = torch.zeros_like(t_now)

        # 현재 상태 x_t를 위치 임베딩 공간으로 보낸다.
        x_emb = self.position_embed(x_t)
        # 시작 시간 r을 임베딩한다.
        t_start_emb = self.start_time_embed(t_start)
        # 현재 시간 t를 임베딩한다.
        t_now_emb = self.now_time_embed(t_now)
        # 위치와 두 시간 임베딩을 이어 붙여 backbone 입력을 만든다.
        h = torch.cat([x_emb, t_start_emb, t_now_emb], dim=1)
        # 연결된 특징을 MLP backbone에 통과시킨다.
        h = self.backbone(h)

        # 오래된 call site와 인터페이스를 맞추기 위해 인자는 유지한다.
        _ = return_dict

        # FM이면 v-head를 사용한다.
        if self.variant == "fm":
            return self.v_head(h)
        # iMF이면 u-head를 사용한다.
        return self.u_head(h)
