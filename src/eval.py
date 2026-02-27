"""FM/iMF MNIST 생성기 성능을 가볍게 점검하는 평가 스크립트."""

# 최신 타입 힌트 문법을 지연 평가로 안전하게 사용한다.
from __future__ import annotations

# CLI 인자 처리를 위해 argparse를 사용한다.
import argparse
# 체크포인트/출력 경로 처리를 위해 Path를 사용한다.
from pathlib import Path
# 설정과 반환값 타입 힌트를 위해 Any/Dict/Tuple을 사용한다.
from typing import Any, Dict, Tuple

# 텐서 연산, 모델 실행, 체크포인트 로딩을 위해 torch를 사용한다.
import torch
# 분류기 정의를 위해 nn 모듈을 사용한다.
import torch.nn as nn
# 분류기 학습 옵티마이저를 위해 optim을 사용한다.
import torch.optim as optim
# 분류 손실 계산을 위해 F API를 사용한다.
import torch.nn.functional as F

# 실제 데이터 로더를 재사용해 분류기 평가 기준을 만든다.
from data import get_mnist_dataloaders
# FM 모드 생성 샘플러를 가져온다.
from fm.sampler import euler_sample_fm
# iMF 모드 생성 샘플러를 가져온다.
from imf.sampler import euler_sample_imf
# 생성기 아키텍처 복원을 위해 모델 클래스를 가져온다.
from models.mlp import VectorFieldMLP
from models.transformer import VectorFieldTransformer
# 생성 결과를 시각적으로 확인하기 위한 그리드 저장 함수를 가져온다.
from utils.grid import save_grid_from_flat
# YAML 설정 로더를 가져온다.
from utils.logging import load_yaml
# 재현 가능한 평가를 위해 시드 설정 함수를 가져온다.
from utils.seed import set_seed


class MNISTClassifier(nn.Module):
    """Small classifier used as a proxy evaluator."""

    def __init__(self, input_dim: int = 784) -> None:
        # 부모 초기화로 파라미터 등록을 준비한다.
        super().__init__()
        # 간단한 MLP 분류기를 구성해 생성 샘플의 proxy 품질을 측정한다.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 벡터를 클래스 로짓으로 변환한다.
        return self.net(x)


def parse_args() -> argparse.Namespace:
    # 평가 스크립트용 CLI 파서를 만든다.
    parser = argparse.ArgumentParser()
    # 공통 실험 설정 파일 경로를 받는다.
    parser.add_argument("--config", type=str, required=True)
    # 평가 대상 생성기 체크포인트 경로를 받는다.
    parser.add_argument("--generator_ckpt", type=str, required=True)
    # 분류기 체크포인트 경로를 받는다(없으면 새로 학습).
    parser.add_argument("--classifier_ckpt", type=str, default="runs/mnist_classifier.pt")
    # 분류기 체크포인트가 없을 때 학습 epoch 수를 지정한다.
    parser.add_argument("--classifier_epochs", type=int, default=2)
    # 생성해 평가할 샘플 총 개수를 지정한다.
    parser.add_argument("--num_gen_samples", type=int, default=2000)
    # 생성 배치 크기를 지정한다.
    parser.add_argument("--batch_size", type=int, default=256)
    # NFE를 CLI에서 덮어쓸 수 있게 한다.
    parser.add_argument("--nfe", type=int, default=None)
    # 생성기 EMA 가중치 사용 여부를 지정한다.
    parser.add_argument("--use_ema", action="store_true")
    # 재현성용 시드를 받는다.
    parser.add_argument("--seed", type=int, default=42)
    # 평가 출력물 저장 디렉터리를 지정한다.
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    # 파싱된 인자 객체를 반환한다.
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    # CUDA 사용 가능 시 GPU를 선택한다.
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # 그렇지 않으면 CPU를 사용한다.
    return torch.device("cpu")


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
    # 생성기 모델 하위 설정을 읽는다.
    model_cfg = cfg["model"]
    # fm/imf 모드를 읽어 출력 헤드를 결정한다.
    mode = cfg["experiment"]["mode"]
    arch = str(model_cfg.get("arch", "mlp")).lower()

    if arch == "mlp":
        # 학습 시 구조와 동일한 생성기 모델을 생성한다.
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
            # 학습 때 쓴 encoder 종류를 그대로 복원한다.
            encoder_variant=str(model_cfg.get("encoder_variant", "standard")).lower(),
            # mHC 실험이라면 같은 residual routing 하이퍼파라미터도 함께 복원한다.
            num_streams=int(model_cfg.get("num_streams", 4)),
            sinkhorn_iters=int(model_cfg.get("sinkhorn_iters", 10)),
            sinkhorn_tau=float(model_cfg.get("sinkhorn_tau", 0.05)),
            residual_identity_mix=bool(model_cfg.get("residual_identity_mix", False)),
            residual_alpha=float(model_cfg.get("residual_alpha", 0.01)),
            variant=mode,
        )
    raise ValueError(f"Unknown model.arch: {arch}")


def evaluate_classifier_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    # 평가 모드로 전환해 드롭아웃/배치통계를 고정한다.
    model.eval()
    # 전체 샘플 수 누적 변수다.
    total = 0
    # 정답 수 누적 변수다.
    correct = 0
    # 정확도 계산은 역전파가 필요 없으므로 no_grad를 사용한다.
    with torch.no_grad():
        for x, y in loader:
            # 입력/라벨을 평가 디바이스로 보낸다.
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 분류기 로짓을 계산한다.
            logits = model(x)
            # 최대 로짓 인덱스를 예측 클래스로 사용한다.
            pred = logits.argmax(dim=1)
            # 배치 샘플 수를 누적한다.
            total += y.numel()
            # 맞춘 샘플 수를 누적한다.
            correct += (pred == y).sum().item()
    # 0으로 나누기를 방지하며 최종 정확도를 반환한다.
    return correct / max(total, 1)


def train_or_load_classifier(
    ckpt_path: Path,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 2,
) -> Tuple[nn.Module, float]:
    # 평가용 보조 분류기를 생성해 디바이스에 올린다.
    clf = MNISTClassifier().to(device)
    # 체크포인트가 있으면 재학습 없이 로드해 시간을 절약한다.
    if ckpt_path.exists():
        clf.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
        acc = evaluate_classifier_accuracy(clf, test_loader, device)
        return clf, acc

    # 체크포인트가 없을 때만 간단히 학습한다.
    optimizer = optim.AdamW(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        clf.train()
        for x, y in train_loader:
            # 배치를 디바이스로 이동한다.
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 이전 step gradient를 초기화한다.
            optimizer.zero_grad(set_to_none=True)
            # 교차엔트로피로 분류 손실을 계산한다.
            loss = F.cross_entropy(clf(x), y)
            # 손실 역전파를 수행한다.
            loss.backward()
            # 파라미터를 업데이트한다.
            optimizer.step()

    # 학습 후 테스트 정확도를 계산한다.
    acc = evaluate_classifier_accuracy(clf, test_loader, device)
    # 분류기 체크포인트 저장 디렉터리를 보장한다.
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 추후 재사용을 위해 분류기 가중치를 저장한다.
    torch.save(clf.state_dict(), ckpt_path)
    # 분류기와 정확도를 반환한다.
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
    # fm/imf 모드에 맞는 샘플러를 선택하기 위해 모드를 읽는다.
    mode = cfg["experiment"]["mode"]
    # 생성 벡터 차원을 설정에서 읽는다.
    dim = int(cfg["model"].get("input_dim", 784))
    # 메모리 관리를 위해 배치별 결과를 누적할 리스트다.
    chunks = []
    # 생성해야 할 남은 샘플 수다.
    remaining = total_samples

    # 총 샘플 수를 배치 단위로 나눠 반복 생성한다.
    while remaining > 0:
        bs = min(batch_size, remaining)
        # 모드에 맞는 샘플러로 현재 배치를 생성한다.
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
        # 후처리를 위해 CPU로 옮겨 보관한다.
        chunks.append(x.cpu())
        # 남은 샘플 수를 감소시킨다.
        remaining -= bs
    # 배치 조각을 하나의 텐서로 합쳐 반환한다.
    return torch.cat(chunks, dim=0)


def generated_proxy_metrics(clf: nn.Module, x_gen: torch.Tensor, device: torch.device) -> Dict[str, float]:
    # 분류기를 평가 모드로 둔다.
    clf.eval()
    # proxy metric 계산은 gradient가 필요 없다.
    with torch.no_grad():
        # 생성 샘플에 대한 분류기 로짓을 계산한다.
        logits = clf(x_gen.to(device))
        # 클래스별 확률로 변환한다.
        probs = torch.softmax(logits, dim=1)
        # 최대 클래스 확률 평균을 confidence proxy로 사용한다.
        confidence = probs.max(dim=1).values.mean().item()
        # 샘플별 예측 클래스를 구한다.
        preds = probs.argmax(dim=1)
        # 예측 클래스 분포 히스토그램을 만든다.
        hist = torch.bincount(preds, minlength=10).float()
        # 확률 분포로 정규화한다.
        hist = hist / hist.sum().clamp_min(1.0)
        # 클래스 분포 엔트로피를 계산해 다양성을 근사 측정한다.
        entropy = float(-(hist * torch.log(hist.clamp_min(1e-8))).sum().item())
    # 평가 로그에 쓸 지표 딕셔너리를 반환한다.
    return {"gen_confidence": confidence, "gen_class_entropy": entropy}


def main() -> None:
    # CLI 인자를 읽는다.
    args = parse_args()
    # YAML 설정을 로드한다.
    cfg = load_yaml(args.config)
    # 평가 재현성을 위해 시드를 고정한다.
    set_seed(args.seed, deterministic=True)

    # 실행 디바이스를 결정한다.
    device = resolve_device(str(cfg["experiment"].get("device", "cuda")))
    # NFE는 CLI 우선, 없으면 config 기본값을 사용한다.
    nfe = args.nfe if args.nfe is not None else int(cfg["sampling"].get("nfe_default", 20))
    # 평가 출력 디렉터리를 만든다.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 분류기 학습/평가에 필요한 실제 MNIST 로더를 준비한다.
    train_loader, test_loader = get_mnist_dataloaders(
        root=str(cfg["data"].get("root", "data")),
        batch_size=int(cfg["data"].get("batch_size", 128)),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        download=True,
    )

    # 평가용 분류기를 로드하거나 새로 학습한다.
    classifier, real_acc = train_or_load_classifier(
        ckpt_path=Path(args.classifier_ckpt),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.classifier_epochs,
    )

    # 생성기 모델을 만들고 체크포인트를 로드한다.
    generator = build_generator(cfg).to(device)
    ckpt = torch.load(args.generator_ckpt, map_location=device)
    # 요청 시 EMA 가중치를 우선 적용한다.
    if args.use_ema and ckpt.get("ema", None) is not None:
        generator.load_state_dict(ckpt["ema"], strict=True)
    else:
        generator.load_state_dict(ckpt["model"], strict=True)
    # 샘플 생성 전 추론 모드로 전환한다.
    generator.eval()

    # 지정 개수만큼 샘플을 생성한다.
    x_gen = generate_samples(
        model=generator,
        cfg=cfg,
        device=device,
        total_samples=args.num_gen_samples,
        batch_size=args.batch_size,
        nfe=nfe,
    )

    # 생성 샘플 기반 proxy 지표를 계산한다.
    metrics = generated_proxy_metrics(classifier, x_gen, device)
    # 분류기의 실제 테스트 정확도도 함께 보고한다.
    metrics["real_test_acc"] = real_acc
    # 비교 편의를 위해 사용한 NFE를 기록한다.
    metrics["nfe"] = float(nfe)
    # 생성 샘플 일부를 그리드 이미지로 저장한다.
    save_grid_from_flat(x_gen[:64], out_dir / f"eval_grid_nfe{nfe}.png", nrow=8)

    # 최종 평가 요약을 콘솔에 출력한다.
    print("Evaluation summary:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    # 스크립트 직접 실행 시 평가 루틴을 시작한다.
    main()
