from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def create_run_dir(base_dir: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_copy(config_path: str, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_python_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def maybe_init_wandb(cfg: Dict[str, Any], run_dir: Path, config_dict: Dict[str, Any]):
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", True):
        return None

    try:
        import wandb
    except ImportError:
        return None

    run = wandb.init(
        project=wb_cfg.get("project", "mnist-flow-matching"),
        entity=wb_cfg.get("entity", None),
        name=cfg["experiment"]["name"],
        dir=str(run_dir),
        config=config_dict,
    )
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
    torch.save(payload, path)
