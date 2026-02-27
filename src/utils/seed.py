"""실험 재현성을 높이기 위한 시드 설정 유틸리티."""

# 최신 타입 힌트 문법을 안전하게 사용하기 위해 지연 평가를 활성화한다.
from __future__ import annotations

# 파이썬 해시 시드와 환경 변수를 제어하기 위해 os를 사용한다.
import os
# 파이썬 표준 난수 시드를 맞추기 위해 random을 사용한다.
import random

# NumPy 난수 시드를 맞추기 위해 numpy를 사용한다.
import numpy as np
# PyTorch 난수 및 결정론 옵션을 제어하기 위해 torch를 사용한다.
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds and deterministic flags where practical."""
    # 파이썬 기본 난수 생성기 시드를 고정한다.
    random.seed(seed)
    # NumPy 난수 생성기 시드를 고정한다.
    np.random.seed(seed)
    # 해시 기반 연산 재현성을 위해 PYTHONHASHSEED를 고정한다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch CPU 난수 시드를 고정한다.
    torch.manual_seed(seed)
    # 사용 가능한 모든 CUDA 디바이스 난수 시드를 고정한다.
    torch.cuda.manual_seed_all(seed)

    # cuDNN 결정론 모드를 설정해 실행마다 동일한 결과를 유도한다.
    torch.backends.cudnn.deterministic = deterministic
    # 결정론 모드일 때는 benchmark를 꺼서 알고리즘 선택 변동을 줄인다.
    torch.backends.cudnn.benchmark = not deterministic
    # 가능한 연산에서 PyTorch 결정론 알고리즘을 사용하도록 요청한다.
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # 일부 환경에서 지원되지 않을 수 있어 전체 실행이 멈추지 않게 무시한다.
            pass
