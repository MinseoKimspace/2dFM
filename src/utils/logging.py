"""실험 디렉터리/로깅/체크포인트 관리를 위한 공통 유틸리티."""

# 최신 타입 힌트 문법을 런타임 충돌 없이 쓰기 위해 지연 평가를 켠다.
from __future__ import annotations

# 파이썬 표준 로거를 구성하기 위해 logging을 사용한다.
import logging
# 설정 파일 복사를 위해 shutil을 사용한다.
import shutil
# 런 디렉터리 이름에 시각 정보를 넣기 위해 datetime을 사용한다.
from datetime import datetime
# 파일 경로를 안전하게 다루기 위해 Path를 사용한다.
from pathlib import Path
# 함수 인자/반환 타입을 명확히 하기 위한 타입 힌트들이다.
from typing import Any, Dict, Optional

# 체크포인트 직렬화를 위해 torch를 사용한다.
import torch
# YAML 설정 로딩을 위해 PyYAML을 사용한다.
import yaml


def create_run_dir(base_dir: str, experiment_name: str) -> Path:
    # 실행 시각을 문자열로 만들어 실험 디렉터리 이름에 포함한다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir 아래에 실험명+타임스탬프 형태의 고유 디렉터리를 만든다.
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    # 동일 이름이 있으면 실패하게 만들어 실험 결과 덮어쓰기를 막는다.
    run_dir.mkdir(parents=True, exist_ok=False)
    # 체크포인트 저장 폴더를 미리 만든다.
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    # 샘플 이미지 저장 폴더를 미리 만든다.
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    # 생성한 런 디렉터리 경로를 반환한다.
    return run_dir


def save_config_copy(config_path: str, run_dir: Path) -> None:
    # 학습 당시 설정을 run_dir에 복사해 재현성을 확보한다.
    shutil.copy2(config_path, run_dir / "config.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    # UTF-8로 설정 파일을 읽어 한글/특수문자 깨짐을 방지한다.
    with open(path, "r", encoding="utf-8") as f:
        # 안전 로더로 YAML을 파싱해 파이썬 dict로 반환한다.
        return yaml.safe_load(f)


def setup_python_logger(run_dir: Path) -> logging.Logger:
    # 학습에서 공통으로 사용할 로거 이름을 고정한다.
    logger = logging.getLogger("train")
    # INFO 이상 로그를 기록한다.
    logger.setLevel(logging.INFO)
    # 중복 핸들러 누적을 막기 위해 기존 핸들러를 정리한다.
    logger.handlers.clear()

    # 파일/콘솔 공통 포맷을 정의한다.
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # 파일로 남길 핸들러를 만든다.
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    # 터미널 출력용 핸들러를 만든다.
    sh = logging.StreamHandler()
    # 파일 핸들러 포맷을 지정한다.
    fh.setFormatter(fmt)
    # 콘솔 핸들러 포맷을 지정한다.
    sh.setFormatter(fmt)

    # 파일 핸들러를 로거에 등록한다.
    logger.addHandler(fh)
    # 콘솔 핸들러를 로거에 등록한다.
    logger.addHandler(sh)
    # 설정 완료된 로거를 반환한다.
    return logger


def maybe_init_wandb(cfg: Dict[str, Any], run_dir: Path, config_dict: Dict[str, Any]):
    # wandb 관련 하위 설정을 가져온다.
    wb_cfg = cfg.get("wandb", {})
    # 비활성화된 경우 바로 None을 반환해 호출부 분기를 단순화한다.
    if not wb_cfg.get("enabled", True):
        return None

    # wandb 설치가 없는 환경에서도 학습이 계속되도록 선택적으로 import한다.
    try:
        import wandb
    except ImportError:
        return None

    # 설정 정보를 넘겨 원격 대시보드에서 실험 메타데이터를 추적하게 한다.
    run = wandb.init(
        project=wb_cfg.get("project", "mnist-flow-matching"),
        entity=wb_cfg.get("entity", None),
        name=cfg["experiment"]["name"],
        dir=str(run_dir),
        config=config_dict,
    )
    # 초기화된 wandb run 객체를 반환한다.
    return run


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_metric: float,
    cfg: Dict[str, Any],
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    # 학습 재개에 필요한 상태를 하나의 payload 딕셔너리로 묶는다.
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": cfg,
        "ema": ema_state,
    }
    # 지정 경로에 체크포인트를 직렬화 저장한다.
    torch.save(payload, path)
