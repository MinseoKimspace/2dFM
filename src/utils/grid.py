"""평탄화된 MNIST 텐서를 이미지/그리드로 변환하는 유틸리티."""

# 최신 타입 힌트 문법 호환을 위해 어노테이션 지연 평가를 사용한다.
from __future__ import annotations

# 출력 경로를 운영체제 독립적으로 다루기 위해 Path를 사용한다.
from pathlib import Path

# 텐서 reshape/보정을 위해 torch를 사용한다.
import torch
# torchvision의 그리드 생성/이미지 저장 함수를 재사용한다.
from torchvision.utils import make_grid, save_image


def flat_to_image(x_flat: torch.Tensor) -> torch.Tensor:
    """Convert [B, 784] in [-1, 1] to [B, 1, 28, 28] in [0, 1]."""
    # 학습용 평탄 텐서를 실제 MNIST 이미지 형태로 복원한다.
    x = x_flat.view(-1, 1, 28, 28)
    # 학습 스케일 [-1,1]을 시각화 스케일 [0,1]로 변환한다.
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def make_grid_from_flat(x_flat: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    # 먼저 평탄 텐서를 이미지 배치로 바꾼다.
    images = flat_to_image(x_flat)
    # 이미지 배치를 지정한 열 개수의 하나의 그리드 텐서로 합친다.
    return make_grid(images, nrow=nrow, padding=2)


def save_grid_from_flat(x_flat: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    # 문자열/Path 입력을 Path 객체로 통일해 후속 처리를 단순화한다.
    path = Path(path)
    # 출력 디렉터리가 없으면 자동 생성해 저장 실패를 방지한다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # 저장 직전에도 이미지 형태로 변환해 일관된 결과를 보장한다.
    images = flat_to_image(x_flat)
    # 배치 이미지를 그리드로 저장한다.
    save_image(images, str(path), nrow=nrow, padding=2)
