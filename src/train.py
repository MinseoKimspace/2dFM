"""FM/iMF 방식으로 MNIST(784 벡터) 생성 모델을 학습하는 메인 스크립트."""

# 최신 타입 힌트 문법을 지연 평가 방식으로 사용한다.
from __future__ import annotations

# CLI 인자 파싱을 위해 argparse를 사용한다.
import argparse
# step당 소요 시간을 재기 위해 고해상도 타이머를 사용한다.
import time
# AMP 비활성 시 대체 컨텍스트로 nullcontext를 사용한다.
from contextlib import nullcontext
# 실행 경로/체크포인트 경로 관리를 위해 Path를 사용한다.
from pathlib import Path
# 설정 타입 힌트를 위해 Any/Dict를 사용한다.
from typing import Any, Dict

# 텐서 연산, 모델 학습, 체크포인트 저장을 위해 torch를 사용한다.
import torch
# 모델 타입 힌트를 위해 nn을 사용한다.
import torch.nn as nn
# 옵티마이저 구성을 위해 optim을 사용한다.
import torch.optim as optim

# MNIST 학습 로더를 가져온다.
from data import get_mnist_dataloaders
# FM 학습 손실을 가져온다.
from fm.losses import fm_velocity_loss
# FM 샘플링 함수를 가져온다.
from fm.sampler import euler_sample_fm
# iMF 학습 손실을 가져온다.
from imf.losses import imf_velocity_loss
# iMF 샘플링 함수를 가져온다.
from imf.sampler import euler_sample_imf
# 공통 백본 모델을 가져온다.
from models.mlp import VectorFieldMLP
from models.transformer import VectorFieldTransformer
# EMA 유틸리티를 가져온다.
from utils.ema import EMA
# 샘플 그리드 생성/저장을 위해 유틸리티를 가져온다.
from utils.grid import make_grid_from_flat, save_grid_from_flat
# 런 디렉터리, 로거, 체크포인트, wandb 초기화를 위한 유틸리티 묶음이다.
from utils.logging import (
    create_run_dir,
    load_yaml,
    maybe_init_wandb,
    save_checkpoint,
    save_config_copy,
    setup_python_logger,
)
# 실험 재현성을 위한 시드 설정 함수를 가져온다.
from utils.seed import set_seed

# 샘플 로깅 시 검증된 NFE 집합만 허용해 비정상 설정을 조기에 차단한다.
SUPPORTED_NFE = {1, 2, 4, 8, 16, 20, 50}


def _compute_grad_norm(parameters) -> float:
    # 모든 파라미터 gradient의 L2 norm을 하나로 합쳐 반환한다.
    total_sq_norm = 0.0
    for param in parameters:
        # gradient가 없는 파라미터는 건너뛴다.
        if param.grad is None:
            continue
        # 각 gradient의 L2 norm을 구한다.
        grad_norm = float(param.grad.detach().float().norm(2).item())
        # 전체 norm 계산을 위해 제곱합을 누적한다.
        total_sq_norm += grad_norm * grad_norm
    # 마지막에 sqrt를 취해 전체 L2 norm으로 만든다.
    return total_sq_norm ** 0.5


def parse_args() -> argparse.Namespace:
    # 학습 스크립트용 CLI 파서를 만든다.
    parser = argparse.ArgumentParser()
    # 실험 설정 YAML 경로를 필수 인자로 받는다.
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    # 파싱 결과를 반환한다.
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    # CUDA 선호 + 사용 가능이면 GPU를 사용한다.
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # 그 외에는 CPU로 폴백한다.
    return torch.device("cpu")


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    # 모델 관련 설정 블록을 읽는다.
    model_cfg = cfg["model"]
    # 실험 모드(fm/imf)를 읽어 출력 헤드를 결정한다.
    mode = cfg["experiment"]["mode"]
    arch = str(model_cfg.get("arch", "mlp")).lower()

    if arch == "mlp":
        # 설정값을 반영해 벡터장 MLP를 생성한다.
        return VectorFieldMLP(
            input_dim=int(model_cfg.get("input_dim", 784)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            variant=mode,
        )
    if arch == "transformer":
        model_dim = int(model_cfg.get("model_dim", model_cfg.get("hidden_dim", 512)))
        return VectorFieldTransformer(
            input_dim=int(model_cfg.get("input_dim", 784)),
            model_dim=model_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ff_dim=int(model_cfg.get("ff_dim", 4 * model_dim)),
            patch_size=int(model_cfg.get("patch_size", 2)),
            image_size=int(model_cfg.get("image_size", 28)),
            in_channels=int(model_cfg.get("in_channels", 1)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            # standard / mhc encoder 스위치를 설정 파일에서 그대로 넘긴다.
            encoder_variant=str(model_cfg.get("encoder_variant", "standard")).lower(),
            # 아래 인자들은 mHC를 사용할 때 residual routing 세부 동작을 정한다.
            num_streams=int(model_cfg.get("num_streams", 4)),
            sinkhorn_iters=int(model_cfg.get("sinkhorn_iters", 10)),
            sinkhorn_tau=float(model_cfg.get("sinkhorn_tau", 0.05)),
            residual_identity_mix=bool(model_cfg.get("residual_identity_mix", False)),
            residual_alpha=float(model_cfg.get("residual_alpha", 0.01)),
            variant=mode,
        )
    raise ValueError(f"Unknown model.arch: {arch}")


@torch.no_grad()
def sample_for_nfe(
    model: nn.Module,
    mode: str,
    cfg: Dict[str, Any],
    device: torch.device,
    nfe: int,
    x_init: torch.Tensor,
) -> torch.Tensor:
    # 샘플 차원을 설정에서 읽어 샘플러에 전달한다.
    input_dim = int(cfg["model"].get("input_dim", 784))
    # FM 모드면 FM 샘플러로 고정 초기노이즈를 적분한다.
    if mode == "fm":
        return euler_sample_fm(
            model=model,
            num_samples=x_init.size(0),
            dim=input_dim,
            device=device,
            nfe=nfe,
            x_init=x_init,
        )

    return euler_sample_imf(
        model=model,
        num_samples=x_init.size(0),
        dim=input_dim,
        device=device,
        nfe=nfe,
        x_init=x_init,
    )


def main() -> None:
    # CLI 인자를 파싱한다.
    args = parse_args()
    # YAML 설정을 로드한다.
    cfg = load_yaml(args.config)

    # 자주 쓰는 하위 설정을 미리 꺼내 코드 가독성을 높인다.
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    samp_cfg = cfg["sampling"]

    # 재현성 확보를 위해 시드를 고정한다.
    set_seed(int(exp_cfg.get("seed", 42)), deterministic=bool(exp_cfg.get("deterministic", True)))
    # 실제 실행 가능한 디바이스를 결정한다.
    device = resolve_device(str(exp_cfg.get("device", "cuda")))

    # 새 실험 런 디렉터리를 만들고 설정 사본을 저장한다.
    run_dir = create_run_dir(str(exp_cfg.get("output_dir", "runs")), str(exp_cfg["name"]))
    save_config_copy(args.config, run_dir)
    # 파일/콘솔 동시 로깅을 위한 로거를 초기화한다.
    logger = setup_python_logger(run_dir)
    # wandb 사용 설정이면 런을 초기화한다.
    wandb_run = maybe_init_wandb(cfg, run_dir, cfg)

    # 실험 시작 시 핵심 환경 정보를 로그로 남긴다.
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    # 학습 데이터 로더를 준비한다.
    train_loader, _ = get_mnist_dataloaders(
        root=str(data_cfg.get("root", "data")),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        download=True,
    )

    # 설정 기반 모델을 생성해 디바이스에 올린다.
    model = build_model(cfg).to(device)
    # AdamW 옵티마이저를 설정한다.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=tuple(float(b) for b in train_cfg.get("betas", [0.9, 0.999])),
    )
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    warmup_steps = max(0, warmup_epochs * len(train_loader))
    scheduler = None
    if warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
        )

    # AMP는 CUDA 환경에서만 활성화한다.
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    # gradient scaling으로 AMP 학습의 underflow를 완화한다.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    # AMP on/off에 따라 autocast 컨텍스트를 통일해서 쓰기 위한 핸들이다.
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    # 설정이 켜져 있으면 EMA 추적 객체를 만든다.
    ema_cfg = cfg.get("ema", {})
    ema = EMA(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("use", False) else None

    # 학습 루프 제어 하이퍼파라미터들을 읽는다.
    mode = str(exp_cfg["mode"])
    epochs = int(train_cfg.get("epochs", 20))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_every = int(train_cfg.get("log_every", 50))
    sample_every = int(train_cfg.get("sample_every", 500))
    max_steps = train_cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    # 샘플 비교를 공정하게 하기 위해 고정 초기노이즈를 한 번 생성한다.
    fixed_noise = torch.randn(int(samp_cfg.get("sample_batch_size", 64)), int(cfg["model"]["input_dim"]), device=device)

    # 전체 스텝과 최고 성능(낮은 loss)을 추적한다.
    global_step = 0
    best_metric = float("inf")

    # 에폭 단위 학습을 시작한다.
    for epoch in range(1, epochs + 1):
        # 학습 모드로 전환한다.
        model.train()
        # 에폭 평균 손실 계산용 누적 변수다.
        epoch_loss_sum = 0.0
        epoch_batches = 0

        # 미니배치 반복 학습을 수행한다.
        for x1, _ in train_loader:
            # 한 optimization step 전체 시간을 재기 시작한다.
            step_start_time = time.perf_counter()
            # 입력 배치를 디바이스로 이동한다.
            x1 = x1.to(device, non_blocking=True)
            # 이전 step gradient를 초기화한다.
            optimizer.zero_grad(set_to_none=True)

            # AMP 설정에 맞춰 손실 계산 구간을 autocast로 감싼다.
            with autocast_ctx(enabled=amp_enabled) if amp_enabled else autocast_ctx():
                # 모드별 손실 함수를 호출한다.
                if mode == "fm":
                    loss, metrics = fm_velocity_loss(model, x1)
                elif mode == "imf":
                    loss, metrics = imf_velocity_loss(model, x1)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            # 스케일된 손실로 역전파한다.
            scaler.scale(loss).backward()
            if amp_enabled:
                # AMP 사용 시 실제 gradient 값을 보기 위해 먼저 unscale한다.
                scaler.unscale_(optimizer)
            # clipping 전 gradient 크기를 기록해 학습 안정성을 모니터링한다.
            grad_norm = _compute_grad_norm(model.parameters())
            # 필요 시 gradient clipping으로 학습 안정성을 높인다.
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # 옵티마이저 스텝을 적용한다.
            scaler.step(optimizer)
            # 스케일러 상태를 업데이트한다.
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            # 현재 step의 실제 wall-clock 시간을 기록한다.
            step_time_sec = time.perf_counter() - step_start_time
            # 기존 metrics dict에 비교용 진단 지표를 추가한다.
            metrics["grad_norm"] = grad_norm
            metrics["step_time_sec"] = float(step_time_sec)

            # EMA 사용 시 현재 파라미터를 shadow에 반영한다.
            if ema is not None:
                ema.update(model)

            # 로깅/종료 제어를 위해 스텝/손실 누적을 갱신한다.
            global_step += 1
            epoch_batches += 1
            epoch_loss_sum += float(loss.item())

            # 지정 주기마다 학습 로그를 남긴다.
            if global_step % log_every == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.6f grad_norm=%.4f step_time=%.4fs",
                    epoch,
                    global_step,
                    metrics["loss"],
                    metrics["grad_norm"],
                    metrics["step_time_sec"],
                )
                # wandb 사용 시 동일 지표를 원격 대시보드에 기록한다.
                if wandb_run is not None:
                    import wandb

                    payload = {f"train/{k}": v for k, v in metrics.items()}
                    payload["train/epoch"] = epoch
                    wandb.log(payload, step=global_step)

            # 지정 스텝마다 중간 샘플을 생성해 학습 진행을 시각적으로 확인한다.
            if sample_every > 0 and global_step % sample_every == 0:
                # 샘플 품질 측정은 EMA 가중치가 일반적으로 안정적이므로 임시 적용한다.
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)
                # 샘플링 전 추론 모드로 전환한다.
                model.eval()

                # 여러 NFE에서의 샘플을 동시에 저장해 속도-품질 트레이드오프를 본다.
                nfe_list = [int(n) for n in samp_cfg.get("nfe_log", [1, 4, 20])]
                for nfe in nfe_list:
                    # 지원하지 않는 NFE는 경고만 남기고 건너뛴다.
                    if nfe not in SUPPORTED_NFE:
                        logger.warning("Skipping unsupported NFE=%d", nfe)
                        continue
                    # 고정 노이즈에서 시작해 해당 NFE로 샘플을 생성한다.
                    samples = sample_for_nfe(
                        model=model,
                        mode=mode,
                        cfg=cfg,
                        device=device,
                        nfe=nfe,
                        x_init=fixed_noise,
                    )
                    # 스텝/ NFE 정보를 파일명에 넣어 비교 가능하게 저장한다.
                    sample_path = run_dir / "samples" / f"step_{global_step:07d}_nfe_{nfe}.png"
                    save_grid_from_flat(samples[:64].detach().cpu(), sample_path, nrow=8)
                    # wandb 사용 시 샘플 그리드도 같이 업로드한다.
                    if wandb_run is not None:
                        import wandb

                        grid = make_grid_from_flat(samples[:64].detach().cpu(), nrow=8)
                        wandb.log({f"samples/nfe_{nfe}": wandb.Image(grid)}, step=global_step)

                # 샘플링이 끝나면 다시 학습 모드로 복귀한다.
                model.train()
                # EMA를 임시 적용했다면 원본 파라미터로 복구한다.
                if ema is not None:
                    ema.restore(model)

            # max_steps가 설정된 경우 목표 스텝에 도달하면 배치 루프를 종료한다.
            if max_steps is not None and global_step >= max_steps:
                break

        # 에폭 평균 손실을 계산해 로그로 남긴다.
        epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
        logger.info("epoch=%d avg_loss=%.6f", epoch, epoch_loss)
        # wandb에도 에폭 평균 손실을 기록한다.
        if wandb_run is not None:
            import wandb

            wandb.log({"train/epoch_avg_loss": epoch_loss, "train/epoch": epoch}, step=global_step)

        # 체크포인트 경로를 준비한다.
        ckpt_dir = run_dir / "checkpoints"
        latest_path = ckpt_dir / "latest.pt"
        best_path = ckpt_dir / "best.pt"
        # EMA 상태가 있으면 함께 저장한다.
        ema_state = ema.state_dict() if ema is not None else None

        # 최신 상태 체크포인트는 매 에폭 저장한다.
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            cfg=cfg,
            ema_state=ema_state,
        )

        # 성능이 개선되면 best 체크포인트를 갱신한다.
        if epoch_loss < best_metric:
            best_metric = epoch_loss
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                cfg=cfg,
                ema_state=ema_state,
            )

        # max_steps 조건을 만족하면 에폭 루프도 종료한다.
        if max_steps is not None and global_step >= max_steps:
            logger.info("Reached max_steps=%d, stopping.", max_steps)
            break

    # wandb 런이 있으면 종료 처리한다.
    if wandb_run is not None:
        wandb_run.finish()
    # 학습 종료 메시지를 남긴다.
    logger.info("Training complete.")


if __name__ == "__main__":
    # 스크립트를 직접 실행한 경우 학습 루틴을 시작한다.
    main()
