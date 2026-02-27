"""시간 t를 신경망이 쓰기 좋은 벡터로 바꾸는 임베딩 모듈."""

# 최신 타입 힌트 문법 호환을 위해 어노테이션 지연 평가를 사용한다.
from __future__ import annotations

# 주파수 스케일 계산에 로그 함수를 사용하기 위해 math를 불러온다.
import math

# 텐서 연산과 모듈 정의를 위해 PyTorch를 사용한다.
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for scalar t in [0, 1]."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        # nn.Module 초기화를 먼저 호출해 내부 상태 등록을 준비한다.
        super().__init__()
        # 출력 임베딩 차원을 저장해 forward에서 사용한다.
        self.dim = dim
        # 가장 긴 주기를 제한해 수치 안정적인 주파수 범위를 만든다.
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 입력이 [B,1]이면 [B]로 줄여 주파수 브로드캐스팅을 단순화한다.
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        # sin/cos 쌍을 만들기 위해 절반 차원을 기본 주파수 개수로 둔다.
        half = self.dim // 2
        # 주파수 텐서는 입력 t와 같은 디바이스에 둬야 불필요한 복사가 없다.
        device = t.device
        # 지수적으로 감소하는 주파수 벡터를 만들어 다양한 시간 스케일을 포착한다.
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        # [B] 시간값과 [half] 주파수를 외적 형태로 결합한다.
        args = t[:, None] * freqs[None]
        # sin/cos를 이어붙여 위상 정보가 다른 임베딩을 함께 제공한다.
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # 홀수 차원 요청 시 마지막 1차원은 0으로 채워 차원을 맞춘다.
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        # 시간 정보를 담은 최종 임베딩을 반환한다.
        return emb


class TimeEmbeddingMLP(nn.Module):
    """Sinusoidal embedding followed by a small MLP projection."""

    def __init__(self, embed_dim: int) -> None:
        # nn.Module 초기화를 수행한다.
        super().__init__()
        # 사인파 임베딩 뒤에 MLP를 붙여 모델이 쓰기 쉬운 표현으로 투영한다.
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 시간 입력을 임베딩 네트워크에 통과시켜 반환한다.
        return self.net(t)
