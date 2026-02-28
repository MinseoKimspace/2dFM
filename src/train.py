"""Train FM/iMF MNIST models."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_mnist_dataloaders
from fm.losses import fm_velocity_loss
from fm.sampler import euler_sample_fm
from imf.losses import imf_velocity_loss
from imf.sampler import euler_sample_imf
from models.mlp import VectorFieldMLP
from models.transformer import VectorFieldTransformer
from utils.ema import EMA
from utils.grid import make_grid_from_flat, save_grid_from_flat
from utils.logging import (
    create_run_dir,
    load_yaml,
    maybe_init_wandb,
    save_checkpoint,
    save_config_copy,
    setup_python_logger,
)
from utils.seed import set_seed

SUPPORTED_NFE = {1, 2, 4, 8, 16, 20, 50}


def _compute_grad_norm(parameters) -> float:
    total_sq_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().float().norm(2).item())
        total_sq_norm += grad_norm * grad_norm
    return total_sq_norm ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    mode = cfg["experiment"]["mode"]
    arch = str(model_cfg.get("arch", "mlp")).lower()

    if arch == "mlp":
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
    input_dim = int(cfg["model"].get("input_dim", 784))
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
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    samp_cfg = cfg["sampling"]

    set_seed(int(exp_cfg.get("seed", 42)), deterministic=bool(exp_cfg.get("deterministic", True)))
    device = resolve_device(str(exp_cfg.get("device", "cuda")))

    run_dir = create_run_dir(str(exp_cfg.get("output_dir", "runs")), str(exp_cfg["name"]))
    save_config_copy(args.config, run_dir)
    logger = setup_python_logger(run_dir)
    wandb_run = maybe_init_wandb(cfg, run_dir, cfg)

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    train_loader, _ = get_mnist_dataloaders(
        root=str(data_cfg.get("root", "data")),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        download=True,
    )

    model = build_model(cfg).to(device)
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

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    ema_cfg = cfg.get("ema", {})
    ema = EMA(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("use", False) else None

    mode = str(exp_cfg["mode"])
    epochs = int(train_cfg.get("epochs", 20))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_every = int(train_cfg.get("log_every", 50))
    sample_every = int(train_cfg.get("sample_every", 500))
    max_steps = train_cfg.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    fixed_noise = torch.randn(
        int(samp_cfg.get("sample_batch_size", 64)),
        int(cfg["model"]["input_dim"]),
        device=device,
    )

    global_step = 0
    best_metric = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for x1, _ in train_loader:
            step_start_time = time.perf_counter()
            x1 = x1.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled) if amp_enabled else nullcontext():
                if mode == "fm":
                    loss, metrics = fm_velocity_loss(model, x1)
                elif mode == "imf":
                    loss, metrics = imf_velocity_loss(model, x1)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            scaler.scale(loss).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)

            grad_norm = _compute_grad_norm(model.parameters())
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            step_time_sec = time.perf_counter() - step_start_time
            metrics["grad_norm"] = grad_norm
            metrics["step_time_sec"] = float(step_time_sec)

            if ema is not None:
                ema.update(model)

            global_step += 1
            epoch_batches += 1
            epoch_loss_sum += float(loss.item())

            if global_step % log_every == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.6f grad_norm=%.4f step_time=%.4fs",
                    epoch,
                    global_step,
                    metrics["loss"],
                    metrics["grad_norm"],
                    metrics["step_time_sec"],
                )
                if wandb_run is not None:
                    import wandb

                    payload = {f"train/{k}": v for k, v in metrics.items()}
                    payload["train/epoch"] = epoch
                    wandb.log(payload, step=global_step)

            if sample_every > 0 and global_step % sample_every == 0:
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)
                model.eval()

                nfe_list = [int(n) for n in samp_cfg.get("nfe_log", [1, 4, 20])]
                for nfe in nfe_list:
                    if nfe not in SUPPORTED_NFE:
                        logger.warning("Skipping unsupported NFE=%d", nfe)
                        continue

                    samples = sample_for_nfe(
                        model=model,
                        mode=mode,
                        cfg=cfg,
                        device=device,
                        nfe=nfe,
                        x_init=fixed_noise,
                    )
                    sample_path = run_dir / "samples" / f"step_{global_step:07d}_nfe_{nfe}.png"
                    save_grid_from_flat(samples[:64].detach().cpu(), sample_path, nrow=8)

                    if wandb_run is not None:
                        import wandb

                        grid = make_grid_from_flat(samples[:64].detach().cpu(), nrow=8)
                        wandb.log({f"samples/nfe_{nfe}": wandb.Image(grid)}, step=global_step)

                model.train()
                if ema is not None:
                    ema.restore(model)

            if max_steps is not None and global_step >= max_steps:
                break

        epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
        logger.info("epoch=%d avg_loss=%.6f", epoch, epoch_loss)
        if wandb_run is not None:
            import wandb

            wandb.log({"train/epoch_avg_loss": epoch_loss, "train/epoch": epoch}, step=global_step)

        ckpt_dir = run_dir / "checkpoints"
        latest_path = ckpt_dir / "latest.pt"
        best_path = ckpt_dir / "best.pt"
        ema_state = ema.state_dict() if ema is not None else None

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

        if max_steps is not None and global_step >= max_steps:
            logger.info("Reached max_steps=%d, stopping.", max_steps)
            break

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
