"""모델 파라미터의 EMA(지수이동평균)를 관리하는 유틸리티."""

# 최신 타입 힌트 문법을 안정적으로 쓰기 위해 지연 평가를 켠다.
from __future__ import annotations

# 내부 상태(dict)와 선택적 백업 상태 타입을 명시한다.
from typing import Dict, Optional

# state_dict 텐서 복사/로딩을 위해 PyTorch를 사용한다.
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        # EMA 감쇠 계수를 저장해 update 때 동일 규칙을 적용한다.
        self.decay = decay
        # 현재 모델 파라미터를 shadow 복사본으로 보관해 EMA 기준점을 만든다.
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        # 평가 시 원본 파라미터 임시 보관용 슬롯을 초기화한다.
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # 모든 파라미터에 대해 shadow = decay*shadow + (1-decay)*current 를 적용한다.
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def store(self, model: torch.nn.Module) -> None:
        # EMA 가중치 적용 전 현재 모델 상태를 복제해 복구 가능하게 만든다.
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        # shadow EMA 파라미터를 실제 모델에 로드해 평가/샘플링 품질을 높인다.
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        # store로 저장한 원본 상태가 있으면 EMA 적용 전 상태로 되돌린다.
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # 체크포인트 저장을 위해 shadow 상태를 외부에 제공한다.
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # 체크포인트에서 읽은 EMA 상태로 내부 shadow를 교체한다.
        self.shadow = state_dict
