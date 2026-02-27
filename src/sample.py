"""학습된 FM/iMF 체크포인트에서 Euler NFE를 지정해 샘플을 생성한다."""

# 최신 타입 힌트 문법을 지연 평가로 안전하게 사용한다.
from __future__ import annotations

# CLI 인자 파싱을 위해 argparse를 사용한다.
import argparse
# 출력 경로 조작을 위해 Path를 사용한다.
from pathlib import Path
# 설정 dict 타입 힌트를 위해 Any/Dict를 사용한다.
from typing import Any, Dict

# 모델 로딩/추론/텐서 저장을 위해 PyTorch를 사용한다.
import torch

# FM 모드용 샘플러를 가져온다.
from fm.sampler import euler_sample_fm
# iMF 모드용 샘플러를 가져온다.
from imf.sampler import euler_sample_imf
# 학습 때와 동일한 생성기 아키텍처를 재구성하기 위해 모델 클래스를 가져온다.
from models.mlp import VectorFieldMLP
from models.transformer import VectorFieldTransformer
# 샘플 시각화를 파일로 저장하기 위한 유틸리티를 가져온다.
from utils.grid import save_grid_from_flat
# YAML 설정 파일을 읽기 위한 유틸리티를 가져온다.
from utils.logging import load_yaml


def parse_args() -> argparse.Namespace:
    # 샘플링 스크립트용 CLI 파서를 만든다.
    parser = argparse.ArgumentParser()
    # 학습/모델 하이퍼파라미터가 담긴 설정 파일 경로를 받는다.
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    # 생성기 체크포인트 경로를 받는다.
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint.")
    # NFE를 CLI에서 덮어쓸 수 있게 하되, 미지정 시 config 값을 쓰도록 None 기본값을 둔다.
    parser.add_argument("--nfe", type=int, default=None, help="Euler NFE. Default comes from config.")
    # 생성할 총 샘플 수를 지정한다.
    parser.add_argument("--num_samples", type=int, default=256, help="Total samples to generate.")
    # 한 번에 생성할 배치 크기를 지정해 메모리 사용량을 제어한다.
    parser.add_argument("--batch_size", type=int, default=256, help="Sampling batch size.")
    # 결과 저장 디렉터리를 지정한다.
    parser.add_argument("--out_dir", type=str, default="sample_outputs", help="Output directory.")
    # 체크포인트에 EMA가 있으면 EMA 가중치를 사용할지 선택한다.
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights if available in checkpoint.",
    )
    # 파싱된 인자 객체를 반환한다.
    return parser.parse_args()


def resolve_device(device_pref: str) -> torch.device:
    # 사용자가 cuda를 원하고 실제 CUDA가 가능하면 GPU를 사용한다.
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # 그 외에는 CPU로 안전하게 폴백한다.
    return torch.device("cpu")


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    # 모델 관련 하위 설정을 추출한다.
    model_cfg = cfg["model"]
    # 실험 모드(fm/imf)에 따라 출력 헤드가 달라지므로 모드를 읽는다.
    mode = cfg["experiment"]["mode"]
    arch = str(model_cfg.get("arch", "mlp")).lower()

    if arch == "mlp":
        # 학습 시 사용한 구조와 동일한 모델을 복원한다.
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
            # 체크포인트와 같은 encoder 종류를 사용해야 state_dict가 맞는다.
            encoder_variant=str(model_cfg.get("encoder_variant", "standard")).lower(),
            # mHC를 쓴 경우 stream 수와 Sinkhorn 설정도 동일하게 복원한다.
            num_streams=int(model_cfg.get("num_streams", 4)),
            sinkhorn_iters=int(model_cfg.get("sinkhorn_iters", 10)),
            sinkhorn_tau=float(model_cfg.get("sinkhorn_tau", 0.05)),
            residual_identity_mix=bool(model_cfg.get("residual_identity_mix", False)),
            residual_alpha=float(model_cfg.get("residual_alpha", 0.01)),
            variant=mode,
        )
    raise ValueError(f"Unknown model.arch: {arch}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    nfe: int,
) -> torch.Tensor:
    # 샘플러 선택을 위해 현재 실험 모드를 읽는다.
    mode = cfg["experiment"]["mode"]
    # 생성 벡터 차원(기본 MNIST 784)을 설정에서 가져온다.
    dim = int(cfg["model"].get("input_dim", 784))

    # FM 모드이면 FM 전용 오일러 샘플러를 사용한다.
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
    # CLI 인자를 읽는다.
    args = parse_args()
    # YAML 설정을 로드한다.
    cfg = load_yaml(args.config)
    # 설정의 device 선호값을 실제 가능한 torch.device로 변환한다.
    device = resolve_device(str(cfg["experiment"].get("device", "cuda")))

    # NFE는 CLI 우선, 없으면 설정 기본값을 사용한다.
    nfe = args.nfe if args.nfe is not None else int(cfg["sampling"].get("nfe_default", 20))
    # 출력 경로를 Path로 만들고 디렉터리를 준비한다.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 인스턴스를 만들고 지정 디바이스로 올린다.
    model = build_model(cfg).to(device)
    # 체크포인트를 동일 디바이스로 로드한다.
    ckpt = torch.load(args.checkpoint, map_location=device)
    # 요청 시 EMA 가중치를 우선 적용하고, 없으면 일반 모델 가중치를 사용한다.
    if args.use_ema and ckpt.get("ema", None) is not None:
        model.load_state_dict(ckpt["ema"], strict=True)
    else:
        model.load_state_dict(ckpt["model"], strict=True)
    # 샘플링 추론 모드로 전환한다.
    model.eval()

    # 메모리 사용량을 제어하기 위해 총 샘플 수를 배치 단위로 나눠 생성한다.
    chunks = []
    remaining = int(args.num_samples)
    while remaining > 0:
        bs = min(int(args.batch_size), remaining)
        chunks.append(sample_batch(model=model, cfg=cfg, device=device, batch_size=bs, nfe=nfe).cpu())
        remaining -= bs
    # 분할 생성한 결과를 하나의 텐서로 합친다.
    samples = torch.cat(chunks, dim=0)

    # 재사용 가능한 텐서 파일과 빠른 확인용 이미지 파일 경로를 만든다.
    tensor_path = out_dir / f"samples_n{args.num_samples}_nfe{nfe}.pt"
    image_path = out_dir / f"grid_n{args.num_samples}_nfe{nfe}.png"
    # 샘플 전체를 텐서로 저장한다.
    torch.save(samples, tensor_path)
    # 앞 64개를 8x8 그리드 이미지로 저장한다.
    save_grid_from_flat(samples[:64], image_path, nrow=8)

    # 사용자 확인을 위해 저장 경로를 출력한다.
    print(f"Saved tensor: {tensor_path}")
    print(f"Saved grid:   {image_path}")


if __name__ == "__main__":
    # 스크립트를 직접 실행한 경우 메인 루틴을 시작한다.
    main()
