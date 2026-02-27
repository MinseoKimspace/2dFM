"""MNIST 데이터셋 로딩과 전처리를 담당하는 유틸리티."""

# 최신 타입 힌트 문법을 지연 평가 방식으로 사용한다.
from __future__ import annotations

# 함수 반환 타입(학습/평가 로더 쌍)을 명시한다.
from typing import Tuple

# 미니배치 로더 생성을 위해 DataLoader를 사용한다.
from torch.utils.data import DataLoader
# MNIST 데이터셋과 전처리 파이프라인을 위해 torchvision을 사용한다.
from torchvision import datasets, transforms


def _flatten_to_vector(x):
    """[1, 28, 28] 이미지를 [784] 벡터로 펼친다."""
    # Windows 멀티프로세싱 DataLoader에서도 안전하게 피클링되도록
    # lambda 대신 모듈 최상위 함수를 사용한다.
    return x.view(-1)


def _mnist_transform(flatten: bool = True) -> transforms.Compose:
    # 텐서 변환과 정규화는 모든 모드에서 공통으로 적용한다.
    tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
    ]
    # MLP 입력 차원(784)에 맞추기 위해 필요 시 이미지를 1차원으로 펼친다.
    if flatten:
        tfms.append(transforms.Lambda(_flatten_to_vector))
    # 리스트로 구성한 전처리를 하나의 Compose로 묶어 반환한다.
    return transforms.Compose(tfms)


def get_mnist_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test dataloaders for MNIST flattened to 784."""
    # 학습 코드와 맞춘 기본 전처리(평탄화 포함)를 가져온다.
    transform = _mnist_transform(flatten=True)
    # 학습 분할을 로드한다.
    train_ds = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    # 테스트 분할을 로드한다.
    test_ds = datasets.MNIST(root=root, train=False, transform=transform, download=download)

    # 학습 로더는 셔플과 drop_last를 켜서 배치 통계를 안정화한다.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    # 테스트 로더는 순서를 유지하고 모든 샘플을 평가하기 위해 drop_last를 끈다.
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    # 학습/테스트 로더를 함께 반환한다.
    return train_loader, test_loader
