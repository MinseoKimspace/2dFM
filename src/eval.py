"""Evaluate FM/iMF MNIST generators with a proxy classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import get_mnist_dataloaders
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
from utils.seed import set_seed


class MNISTClassifier(nn.Module):
    """Small classifier used as a proxy evaluator."""

    def __init__(self, input_dim: int = 784) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--generator_ckpt", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, default="runs/mnist_classifier.pt")
    parser.add_argument("--classifier_epochs", type=int, default=2)
    parser.add_argument("--num_gen_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nfe", type=int, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
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


def evaluate_classifier_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.numel()
            correct += (pred == y).sum().item()
    return correct / max(total, 1)


def train_or_load_classifier(
    ckpt_path: Path,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 2,
) -> Tuple[nn.Module, float]:
    clf = MNISTClassifier().to(device)
    if ckpt_path.exists():
        clf.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        acc = evaluate_classifier_accuracy(clf, test_loader, device)
        return clf, acc

    optimizer = optim.AdamW(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(clf(x), y)
            loss.backward()
            optimizer.step()

    acc = evaluate_classifier_accuracy(clf, test_loader, device)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), ckpt_path)
    return clf, acc


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    total_samples: int,
    batch_size: int,
    nfe: int,
) -> torch.Tensor:
    mode = cfg["experiment"]["mode"]
    dim = int(cfg["model"].get("input_dim", 784))
    chunks = []
    remaining = total_samples

    while remaining > 0:
        bs = min(batch_size, remaining)
        if mode == "fm":
            x = euler_sample_fm(model, num_samples=bs, dim=dim, device=device, nfe=nfe)
        else:
            x = euler_sample_imf(
                model,
                num_samples=bs,
                dim=dim,
                device=device,
                nfe=nfe,
            )
        chunks.append(x.cpu())
        remaining -= bs
    return torch.cat(chunks, dim=0)


def generated_proxy_metrics(clf: nn.Module, x_gen: torch.Tensor, device: torch.device) -> Dict[str, float]:
    clf.eval()
    with torch.no_grad():
        logits = clf(x_gen.to(device))
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values.mean().item()
        preds = probs.argmax(dim=1)
        hist = torch.bincount(preds, minlength=10).float()
        hist = hist / hist.sum().clamp_min(1.0)
        entropy = float(-(hist * torch.log(hist.clamp_min(1e-8))).sum().item())
    return {"gen_confidence": confidence, "gen_class_entropy": entropy}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(args.seed, deterministic=True)

    device = resolve_device(str(cfg["experiment"].get("device", "cuda")))
    nfe = args.nfe if args.nfe is not None else int(cfg["sampling"].get("nfe_default", 20))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(
        root=str(cfg["data"].get("root", "data")),
        batch_size=int(cfg["data"].get("batch_size", 128)),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        download=True,
    )

    classifier, real_acc = train_or_load_classifier(
        ckpt_path=Path(args.classifier_ckpt),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.classifier_epochs,
    )

    generator = build_generator(cfg).to(device)
    ckpt = torch.load(args.generator_ckpt, map_location=device)
    if args.use_ema and ckpt.get("ema", None) is not None:
        generator.load_state_dict(ckpt["ema"], strict=True)
    else:
        generator.load_state_dict(ckpt["model"], strict=True)
    generator.eval()

    x_gen = generate_samples(
        model=generator,
        cfg=cfg,
        device=device,
        total_samples=args.num_gen_samples,
        batch_size=args.batch_size,
        nfe=nfe,
    )

    metrics = generated_proxy_metrics(classifier, x_gen, device)
    metrics["real_test_acc"] = real_acc
    metrics["nfe"] = float(nfe)
    save_grid_from_flat(x_gen[:64], out_dir / f"eval_grid_nfe{nfe}.png", nrow=8)

    print("Evaluation summary:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
