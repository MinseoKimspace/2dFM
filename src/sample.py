"""Generate samples from a trained FM/iMF checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from fm.sampler import euler_sample_fm
from imf.sampler import euler_sample_imf
from models.mlp import VectorFieldMLP
from models.pooled_transformer import (
    DualLevelPoolingHead,
    DualLevelSelfGuidedTransformer,
    SemanticConsistencyHead,
)
from models.transformer import VectorFieldTransformer
from utils.grid import save_grid_from_flat
from utils.logging import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint.")
    parser.add_argument("--nfe", type=int, default=None, help="Euler NFE. Default comes from config.")
    parser.add_argument("--num_samples", type=int, default=256, help="Total samples to generate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Sampling batch size.")
    parser.add_argument("--out_dir", type=str, default="sample_outputs", help="Output directory.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights if available in checkpoint.",
    )
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
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
    if arch == "pooled_transformer":
        model_dim = int(model_cfg.get("model_dim", model_cfg.get("hidden_dim", 512)))
        num_layers = int(model_cfg.get("num_layers", 6))
        num_heads = int(model_cfg.get("num_heads", 8))
        code_dim = int(model_cfg.get("code_dim", model_dim))
        pool_heads = int(model_cfg.get("pool_heads", num_heads))
        early_indices = model_cfg.get("early_indices", list(range(num_layers // 2)))
        late_indices = model_cfg.get("late_indices", list(range(num_layers // 2, num_layers)))

        backbone = VectorFieldTransformer(
            input_dim=int(model_cfg.get("input_dim", 784)),
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=int(model_cfg.get("ff_dim", 4 * model_dim)),
            patch_size=int(model_cfg.get("patch_size", 2)),
            image_size=int(model_cfg.get("image_size", 28)),
            in_channels=int(model_cfg.get("in_channels", 1)),
            time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            variant=mode,
            cond_dim=code_dim,
        )
        pooling_head = DualLevelPoolingHead(
            num_layers=num_layers,
            dim=model_dim,
            pool_heads=pool_heads,
            early_indices=[int(i) for i in early_indices],
            late_indices=[int(i) for i in late_indices],
            early_num_seeds=int(model_cfg.get("early_num_seeds", 4)),
            late_num_seeds=int(model_cfg.get("late_num_seeds", 1)),
            code_dim=code_dim,
        )
        consistency_head = SemanticConsistencyHead(
            code_dim=code_dim,
            hidden_dim=int(model_cfg.get("consistency_hidden_dim", code_dim)),
        )
        return DualLevelSelfGuidedTransformer(backbone, pooling_head, consistency_head)
    raise ValueError(f"Unknown model.arch: {arch}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    nfe: int,
) -> torch.Tensor:
    mode = cfg["experiment"]["mode"]
    dim = int(cfg["model"].get("input_dim", 784))

    if mode == "fm":
        return euler_sample_fm(model, num_samples=batch_size, dim=dim, device=device, nfe=nfe)
    return euler_sample_imf(
        model,
        num_samples=batch_size,
        dim=dim,
        device=device,
        nfe=nfe,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = resolve_device(str(cfg["experiment"].get("device", "cuda")))

    nfe = args.nfe if args.nfe is not None else int(cfg["sampling"].get("nfe_default", 20))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.use_ema and ckpt.get("ema", None) is not None:
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    chunks = []
    remaining = int(args.num_samples)
    while remaining > 0:
        bs = min(int(args.batch_size), remaining)
        chunks.append(sample_batch(model=model, cfg=cfg, device=device, batch_size=bs, nfe=nfe).cpu())
        remaining -= bs

    samples = torch.cat(chunks, dim=0)

    tensor_path = out_dir / f"samples_n{args.num_samples}_nfe{nfe}.pt"
    image_path = out_dir / f"grid_n{args.num_samples}_nfe{nfe}.png"
    torch.save(samples, tensor_path)
    save_grid_from_flat(samples[:64], image_path, nrow=8)

    print(f"Saved tensor: {tensor_path}")
    print(f"Saved grid:   {image_path}")


if __name__ == "__main__":
    main()
