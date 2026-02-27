# Minimal MNIST Flow-Matching Research Codebase (PyTorch)

This project provides a clean baseline for MNIST `flatten(784)` experiments with:

- Baseline Flow Matching (FM): direct regression of instantaneous velocity `v_theta(x_t, t)`
- iMF-style variant: network predicts `u_theta(x_t, t)` with a deterministic `u -> v` mapping, while the loss stays on instantaneous velocity
- Euler ODE sampling with configurable NFE (`1, 2, 4, 8, 16, 20`)
- W&B logging (losses + sample grids), YAML config-driven runs, reproducible seeds, and optional AMP

설명:
- 이 저장소는 FM 기본선과 iMF 변형을 같은 구조에서 비교하기 위해 만든 최소 실험 코드입니다.
- 입력을 MNIST `flatten(784)`로 고정해 모델/샘플링 차이의 영향만 분리해서 보기 쉽습니다.
- NFE를 바꿔가며 속도-품질 트레이드오프를 빠르게 확인하도록 구성했습니다.

## Project Layout

```text
project/
  configs/
    mnist_fm_mlp.yaml
    mnist_imf_mlp.yaml
  src/
    data.py
    utils/
      seed.py
      logging.py
      ema.py
      grid.py
    models/
      time_embed.py
      mlp.py
    fm/
      paths.py
      losses.py
      sampler.py
    imf/
      losses.py
      sampler.py
    train.py
    sample.py
    eval.py
  README.md
```

설명:
- `src/fm/*`는 FM 경로/손실/샘플러, `src/imf/*`는 iMF 매핑/손실/샘플러를 담당합니다.
- `src/train.py`, `src/sample.py`, `src/eval.py`는 각각 학습/샘플링/평가 진입점입니다.
- `configs/*.yaml` 중심으로 실험을 제어하도록 분리되어 재현 실험이 쉽습니다.

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml wandb
```

If you want CPU-only PyTorch, install the matching CPU wheels from the official PyTorch selector.

설명:
- 위 명령은 CUDA wheel 기준입니다.
- GPU가 없으면 PyTorch 공식 selector에서 CPU wheel로 바꿔 설치하면 됩니다.
- `pyyaml`은 설정 로딩, `wandb`는 실험 추적용이며 필요 없으면 비활성화 가능합니다.

## W&B Login

```bash
wandb login
```

You can disable W&B by setting `wandb.enabled: false` in config.

설명:
- 온라인 로깅이 필요 없으면 YAML에서 `wandb.enabled: false`로 두고 로컬 로그만 사용하면 됩니다.

## Train Baseline FM

```bash
python src/train.py --config configs/mnist_fm_mlp.yaml
```

설명:
- FM 모드는 모델이 순간 속도 `v`를 직접 회귀하는 기준선 실험입니다.

## Train iMF-Style Variant

```bash
python src/train.py --config configs/mnist_imf_mlp.yaml
```

Notes:
- FM uses model output `v`.
- iMF uses model output `u` and applies deterministic mapping in `src/imf/losses.py`:
  - `identity`: `v = u`
  - `affine_xt`: `v = u + kappa * (2t - 1) * x_t`
- Edit `map_average_to_instantaneous(...)` to plug in your own mapping.

설명:
- iMF 실험의 핵심은 "출력 파라미터화(u)"와 "최종 손실 기준(v)"을 분리하는 것입니다.
- 매핑 함수를 바꿔도 손실은 동일한 속도 기준으로 유지되어 비교가 명확합니다.

## Sample With Specific NFE (1 / 4 / 20)

```bash
python src/sample.py --config configs/mnist_fm_mlp.yaml --checkpoint runs/<run>/checkpoints/best.pt --nfe 1
python src/sample.py --config configs/mnist_fm_mlp.yaml --checkpoint runs/<run>/checkpoints/best.pt --nfe 4
python src/sample.py --config configs/mnist_fm_mlp.yaml --checkpoint runs/<run>/checkpoints/best.pt --nfe 20
```

iMF example:

```bash
python src/sample.py --config configs/mnist_imf_mlp.yaml --checkpoint runs/<run>/checkpoints/best.pt --nfe 1
```

설명:
- 같은 체크포인트에서 NFE만 바꾸면 적분 스텝 수 변화의 영향만 따로 관찰할 수 있습니다.
- 샘플은 텐서(`.pt`)와 그리드 이미지(`.png`)로 저장되어 정량/정성 비교를 같이 할 수 있습니다.

## Evaluate

`eval.py` trains (or loads) a small MNIST classifier and reports:
- `real_test_acc`: classifier sanity on true MNIST test set
- `gen_confidence`: average max softmax confidence on generated samples
- `gen_class_entropy`: class usage entropy on generated samples

```bash
python src/eval.py --config configs/mnist_fm_mlp.yaml --generator_ckpt runs/<run>/checkpoints/best.pt --nfe 20
```

설명:
- 이 평가는 빠른 반복 실험을 위한 경량 proxy 지표입니다.
- `real_test_acc`는 분류기 정상 동작 확인용이고, 생성 샘플 품질 신호는 `gen_confidence`, 다양성 신호는 `gen_class_entropy`입니다.

## Reproducibility

- Seed is set for Python, NumPy, and PyTorch
- Deterministic flags are enabled via config (`experiment.deterministic`)

설명:
- 실험 비교의 신뢰도를 위해 시드/결정론 옵션을 설정 레벨에서 고정합니다.

## Future Hook Locations

- Guidance code / feature hook stubs are in `src/models/mlp.py`:
  - early hidden feature extraction (`h_early`)
  - guidance code `s = g(h_early)`
  - FiLM/AdaLN injection point

These are intentionally disabled by default so FM and iMF baselines stay clean.

설명:
- 가이드 코드/FiLM 주입 포인트는 향후 확장 실험을 위한 자리입니다.
- 기본선 비교를 깨지 않기 위해 현재는 의도적으로 비활성화되어 있습니다.
