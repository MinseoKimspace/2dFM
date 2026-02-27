"""FM/iMF 벡터 필드 예측을 위한 Transformer 백본."""

from __future__ import annotations

# math는 log-domain 상수 계산과 sigmoid-logit 변환에 사용한다.
import math

# torch는 Tensor 연산과 모듈 정의를 제공한다.
import torch
# nn은 Linear, MultiheadAttention 같은 학습 가능한 블록을 담고 있다.
import torch.nn as nn
# functional은 patch 추출/복원과 softmax 같은 함수형 연산을 제공한다.
import torch.nn.functional as F

# TimeEmbeddingMLP는 스칼라 시간을 학습 가능한 임베딩으로 바꾼다.
from models.time_embed import TimeEmbeddingMLP


def _init_linear_conservative(module: nn.Module) -> None:
    """Linear 층을 작은 가우시안으로 보수적으로 초기화한다."""
    # Linear 층만 건드리고 나머지 모듈은 그대로 둔다.
    if isinstance(module, nn.Linear):
        # fan_in은 입력 차원 수다.
        fan_in = module.weight.size(1)
        # 초기 activation이 너무 커지지 않도록 표준편차를 작게 잡는다.
        std = (0.1 / float(fan_in)) ** 0.5
        # 평균 0의 작은 가우시안으로 weight를 초기화한다.
        nn.init.normal_(module.weight, mean=0.0, std=std)
        # bias는 0에서 시작시켜 출력을 중앙에 맞춘다.
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SinkhornProjector(nn.Module):
    """정사각 logits를 거의 doubly-stochastic한 행렬로 투영한다."""

    def __init__(self, num_iters: int = 10, tau: float = 0.05, eps: float = 1e-6) -> None:
        # 일반적인 PyTorch 모듈로 등록한다.
        super().__init__()
        # Sinkhorn은 최소 1번 이상 반복해야 한다.
        if num_iters <= 0:
            raise ValueError("num_iters must be positive.")
        # temperature는 양수여야 한다.
        if tau <= 0:
            raise ValueError("tau must be positive.")
        # 안정성 상수도 양수여야 한다.
        if eps <= 0:
            raise ValueError("eps must be positive.")

        # 행/열 정규화 반복 횟수.
        self.num_iters = int(num_iters)
        # logits를 얼마나 날카롭게 볼지 결정하는 temperature.
        self.tau = float(tau)
        # 향후 안정성 용도로 보관하는 작은 상수.
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # stream mixing 행렬은 [S, S] 모양의 정사각 텐서여야 한다.
        if logits.ndim != 2 or logits.size(0) != logits.size(1):
            raise ValueError("SinkhornProjector expects square [S, S] logits.")

        # temperature로 스케일링해 soft/hard 정도를 조절한다.
        scaled = logits / self.tau
        # exp 전에 최댓값을 빼서 수치적으로 더 안정하게 만든다.
        scaled = scaled - scaled.amax(dim=(-2, -1), keepdim=True)
        # S는 현재 stream 개수다.
        num_streams = scaled.size(0)
        # 목표 marginal은 균등 분포이며 log-space에서 표현한다.
        log_marginal = torch.full(
            (num_streams,),
            -math.log(float(num_streams)),
            dtype=scaled.dtype,
            device=scaled.device,
        )
        # u는 row scaling 항.
        u = torch.zeros_like(log_marginal)
        # v는 column scaling 항.
        v = torch.zeros_like(log_marginal)

        # log-space에서 row / column 정규화를 번갈아 수행한다.
        for _ in range(self.num_iters):
            # 각 row가 목표 marginal을 갖도록 보정한다.
            u = log_marginal - torch.logsumexp(scaled + v.unsqueeze(0), dim=1)
            # 각 column이 목표 marginal을 갖도록 보정한다.
            v = log_marginal - torch.logsumexp(scaled + u.unsqueeze(1), dim=0)

        # 다시 normal space로 돌아와 양수 행렬을 만든다.
        # 여기서는 행/열 합이 1 근처가 되도록 S를 곱하는 형태를 쓴다.
        return torch.exp(scaled + u.unsqueeze(1) + v.unsqueeze(0)) * float(num_streams)


class MHCResidualWrapper(nn.Module):
    """mHC residual 업데이트 x <- H_res x + H_post^T F(H_pre x)."""

    def __init__(
        self,
        num_streams: int,
        selected_stream: int,
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        residual_identity_mix: bool = False,
        residual_alpha: float = 0.01,
        init_logit_scale: float = 8.0,
    ) -> None:
        # 학습 가능한 모듈로 등록한다.
        super().__init__()
        # stream은 최소 1개 이상이어야 한다.
        if num_streams <= 0:
            raise ValueError("num_streams must be positive.")
        # 초기 read/write 기준 stream은 유효한 인덱스여야 한다.
        if not 0 <= selected_stream < num_streams:
            raise ValueError("selected_stream must index a valid stream.")
        # 초기 logit scale은 양수여야 한다.
        if init_logit_scale <= 0:
            raise ValueError("init_logit_scale must be positive.")

        # 나중 reshape와 mixing에서 사용할 stream 개수.
        self.num_streams = int(num_streams)
        # raw residual logits를 Sinkhorn으로 투영하는 모듈.
        self.project = SinkhornProjector(num_iters=sinkhorn_iters, tau=sinkhorn_tau)
        # 초기에 identity 근처에서 시작할지 여부.
        self.residual_identity_mix = bool(residual_identity_mix)

        # 큰 음수 off-diagonal로 시작시켜 초기에 거의 identity routing이 되게 한다.
        base = float(init_logit_scale)
        # H_res는 기존 stream들이 다음 residual stream으로 어떻게 섞일지 학습한다.
        h_res_logits = torch.full((num_streams, num_streams), -base)
        # 대각선은 0으로 두어 자기 자신을 우선 통과시키게 만든다.
        h_res_logits.fill_diagonal_(0.0)
        # H_res logits를 학습 파라미터로 등록한다.
        self.h_res_logits = nn.Parameter(h_res_logits)

        # H_pre는 branch 입력으로 어떤 stream 조합을 읽을지 정한다.
        h_pre_logits = torch.full((num_streams,), -base)
        # 초반에는 선택된 stream 하나만 주로 읽게 만든다.
        h_pre_logits[selected_stream] = 0.0
        # H_pre logits를 학습 파라미터로 등록한다.
        self.h_pre_logits = nn.Parameter(h_pre_logits)

        # H_post는 branch 출력을 어떤 stream들에 써 넣을지 정한다.
        h_post_logits = torch.full((num_streams,), -base)
        # 초반에는 선택된 stream 하나에 주로 써 넣게 만든다.
        h_post_logits[selected_stream] = 0.0
        # H_post logits를 학습 파라미터로 등록한다.
        self.h_post_logits = nn.Parameter(h_post_logits)

        # alpha를 0과 1에서 조금 떼어 logit 변환이 무한대로 가지 않게 한다.
        clipped_alpha = max(1e-4, min(1.0 - 1e-4, float(residual_alpha)))
        # unconstrained하게 학습할 수 있도록 alpha를 logit 형태로 저장한다.
        alpha_logit = math.log(clipped_alpha / (1.0 - clipped_alpha))
        # identity interpolation 강도를 학습 파라미터로 등록한다.
        self.h_res_alpha_logit = nn.Parameter(torch.tensor(alpha_logit))

    def _reshape_streams(self, streams: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        # 내부에서는 flatten된 [B*S, T, D] 형태를 입력으로 받는다.
        if streams.ndim != 3:
            raise ValueError("Expected flattened streams with shape [B*S, T, D].")
        # flatten된 batch, token 수, hidden 차원을 읽는다.
        batch_times_streams, num_tokens, model_dim = streams.shape
        # B*S가 정확히 S로 나누어져야 원래 batch를 복원할 수 있다.
        if batch_times_streams % self.num_streams != 0:
            raise ValueError(
                "Batch dimension must be divisible by num_streams "
                f"(got {batch_times_streams} and {self.num_streams})."
            )

        # 원래 batch 크기 B를 복원한다.
        batch_size = batch_times_streams // self.num_streams
        # 먼저 [B, S, T, D]로 reshape한다.
        streams_4d = streams.contiguous().view(batch_size, self.num_streams, num_tokens, model_dim)
        # stream mixing이 편하도록 [B, T, S, D]로 축 순서를 바꾼다.
        streams_4d = streams_4d.permute(0, 2, 1, 3).contiguous()
        # 변환된 텐서와 복원된 차원 정보를 함께 돌려준다.
        return streams_4d, batch_size, num_tokens, model_dim

    @staticmethod
    def _flatten_streams(streams: torch.Tensor) -> torch.Tensor:
        # 입력은 [B, T, S, D]라고 가정한다.
        batch_size, num_tokens, num_streams, model_dim = streams.shape
        # 다시 [B*S, T, D]로 평탄화해 이후 branch 모듈과 인터페이스를 맞춘다.
        return streams.permute(0, 2, 1, 3).contiguous().view(
            batch_size * num_streams,
            num_tokens,
            model_dim,
        )

    def _compute_h_res(self) -> torch.Tensor:
        # raw residual logits를 거의 doubly-stochastic한 mixing 행렬로 만든다.
        h_res = self.project(self.h_res_logits)
        # identity interpolation을 쓰지 않으면 그대로 반환한다.
        if not self.residual_identity_mix:
            return h_res

        # alpha logit을 [0, 1] 구간의 interpolation weight로 바꾼다.
        alpha = torch.sigmoid(self.h_res_alpha_logit)
        # 같은 dtype / device의 identity 행렬을 만든다.
        eye = torch.eye(self.num_streams, dtype=h_res.dtype, device=h_res.device)
        # identity와 학습된 mixing을 섞어 초반 학습을 더 안정적으로 만든다.
        return (1.0 - alpha) * eye + alpha * h_res

    def width_connection(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # flatten된 [B*S, T, D]를 명시적 stream 축이 있는 [B, T, S, D]로 바꾼다.
        streams_4d, _, _, _ = self._reshape_streams(streams)
        # residual 경로에서 사용할 H_res를 만든다.
        h_res = self._compute_h_res()
        # H_pre logits를 softmax해 읽기 가중치로 바꾼다.
        h_pre = F.softmax(self.h_pre_logits, dim=-1)
        # H_post logits를 softmax해 쓰기 가중치로 바꾼다.
        h_post = F.softmax(self.h_post_logits, dim=-1)

        # residual stream들을 H_res로 섞어 skip path를 만든다.
        residual_mixed = torch.einsum("ij,btjd->btid", h_res, streams_4d)
        # branch는 하나의 [B, T, D] 시퀀스를 받으므로 H_pre로 stream 축을 접는다.
        branch_input = torch.einsum("s,btsd->btd", h_pre, streams_4d)
        # branch 입력, flatten된 residual path, writeback 가중치를 반환한다.
        return branch_input, self._flatten_streams(residual_mixed), h_post

    def depth_connection(
        self,
        branch_output: torch.Tensor,
        residual_streams: torch.Tensor,
        h_post: torch.Tensor,
    ) -> torch.Tensor:
        # flatten된 residual path를 다시 [B, T, S, D]로 복원한다.
        residuals_4d, _, _, _ = self._reshape_streams(residual_streams)
        # branch 출력을 모든 stream에 broadcast한 뒤 H_post로 크기를 조절한다.
        branch_to_streams = branch_output.unsqueeze(2) * h_post.view(1, 1, self.num_streams, 1)
        # residual path와 branch 출력을 합친 뒤 다시 flatten해서 반환한다.
        return self._flatten_streams(residuals_4d + branch_to_streams)

    def forward(self, streams: torch.Tensor, branch_fn) -> torch.Tensor:
        # width_connection은 branch 입력을 만들고 residual path를 준비한다.
        branch_input, residual_streams, h_post = self.width_connection(streams)
        # attention 또는 FFN branch를 실행한다.
        branch_output = branch_fn(branch_input)
        # branch 출력을 stream 축에 다시 써 넣는다.
        return self.depth_connection(branch_output, residual_streams, h_post)


class MHCTransformerEncoderLayer(nn.Module):
    """attention과 FFN residual을 mHC로 감싼 pre-LN encoder layer."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
        num_streams: int,
        selected_stream: int,
        dropout: float = 0.0,
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        residual_identity_mix: bool = False,
        residual_alpha: float = 0.01,
    ) -> None:
        # 일반적인 PyTorch 모듈로 등록한다.
        super().__init__()
        # attention branch 앞의 pre-norm.
        self.norm1 = nn.LayerNorm(model_dim)
        # FFN branch 앞의 pre-norm.
        self.norm2 = nn.LayerNorm(model_dim)
        # branch 내부는 표준 self-attention을 그대로 사용한다.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # attention 출력 뒤 dropout.
        self.dropout1 = nn.Dropout(dropout)
        # FFN의 첫 번째 projection은 hidden width를 넓힌다.
        self.linear1 = nn.Linear(model_dim, ff_dim)
        # FFN의 두 번째 projection은 다시 model_dim으로 줄인다.
        self.linear2 = nn.Linear(ff_dim, model_dim)
        # FFN 중간 dropout.
        self.dropout = nn.Dropout(dropout)
        # FFN 출력 dropout.
        self.dropout2 = nn.Dropout(dropout)
        # 표준 encoder와 맞추기 위해 GELU를 사용한다.
        self.activation = nn.GELU()

        # attention branch를 mHC residual wrapper로 감싼다.
        self.attn_residual = MHCResidualWrapper(
            num_streams=num_streams,
            selected_stream=selected_stream,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_tau=sinkhorn_tau,
            residual_identity_mix=residual_identity_mix,
            residual_alpha=residual_alpha,
        )
        # FFN branch도 별도의 mHC residual wrapper로 감싼다.
        self.ff_residual = MHCResidualWrapper(
            num_streams=num_streams,
            selected_stream=selected_stream,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_tau=sinkhorn_tau,
            residual_identity_mix=residual_identity_mix,
            residual_alpha=residual_alpha,
        )

    def _attention_branch(self, x: torch.Tensor) -> torch.Tensor:
        # branch 입력 [B, T, D]에 pre-norm을 적용한다.
        x_norm = self.norm1(x)
        # 표준 self-attention을 수행한다.
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        # residual writeback 전에 dropout을 적용한다.
        return self.dropout1(attn_out)

    def _ff_branch(self, x: torch.Tensor) -> torch.Tensor:
        # FFN branch 앞에서 pre-norm을 적용한다.
        x_norm = self.norm2(x)
        # hidden width를 넓힌다.
        x_ff = self.linear1(x_norm)
        # 비선형성을 넣는다.
        x_ff = self.activation(x_ff)
        # 중간 dropout을 적용한다.
        x_ff = self.dropout(x_ff)
        # 다시 model_dim으로 줄인다.
        x_ff = self.linear2(x_ff)
        # 출력 dropout을 적용한다.
        return self.dropout2(x_ff)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        # attention branch를 mHC routing으로 통과시킨다.
        streams = self.attn_residual(streams, self._attention_branch)
        # 이어서 FFN branch도 mHC routing으로 통과시킨다.
        return self.ff_residual(streams, self._ff_branch)


class MHCTransformerEncoder(nn.Module):
    """mHC residual routing을 사용하는 Transformer encoder 스택."""

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        num_streams: int = 4,
        dropout: float = 0.0,
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        residual_identity_mix: bool = False,
        residual_alpha: float = 0.01,
    ) -> None:
        # PyTorch 모듈로 등록한다.
        super().__init__()
        # encoder layer는 최소 1개 이상이어야 한다.
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        # stream도 최소 1개 이상이어야 한다.
        if num_streams <= 0:
            raise ValueError("num_streams must be positive.")

        # reshape와 stream 확장에 사용할 stream 개수.
        self.num_streams = int(num_streams)
        # 같은 토큰을 여러 stream으로 복제할 때 작은 차이를 주는 embedding.
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, model_dim))
        # mHC encoder layer를 num_layers개 쌓는다.
        self.layers = nn.ModuleList(
            [
                MHCTransformerEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    num_streams=num_streams,
                    selected_stream=layer_index % num_streams,
                    dropout=dropout,
                    sinkhorn_iters=sinkhorn_iters,
                    sinkhorn_tau=sinkhorn_tau,
                    residual_identity_mix=residual_identity_mix,
                    residual_alpha=residual_alpha,
                )
                for layer_index in range(num_layers)
            ]
        )
        # stream embedding은 서로 아주 조금만 다르게 시작시킨다.
        nn.init.normal_(self.stream_embed, mean=0.0, std=0.02)

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        # 입력은 일반 encoder와 같은 [B, T, D] 형태다.
        batch_size, num_tokens, model_dim = x.shape
        # 각 토큰을 S개 stream으로 복제하고 stream별 embedding을 더한다.
        streams = x.unsqueeze(2) + self.stream_embed.view(1, 1, self.num_streams, model_dim)
        # branch 모듈 재사용을 위해 [B*S, T, D]로 flatten한다.
        return streams.permute(0, 2, 1, 3).contiguous().view(
            batch_size * self.num_streams,
            num_tokens,
            model_dim,
        )

    def _reduce_streams(self, streams: torch.Tensor) -> torch.Tensor:
        # 입력은 flatten된 [B*S, T, D] 형태다.
        batch_times_streams, num_tokens, model_dim = streams.shape
        # 다시 [B, S, T, D]로 복원할 수 있어야 한다.
        if batch_times_streams % self.num_streams != 0:
            raise ValueError(
                "Batch dimension must be divisible by num_streams "
                f"(got {batch_times_streams} and {self.num_streams})."
            )
        # 원래 batch 크기 B를 복원한다.
        batch_size = batch_times_streams // self.num_streams
        # 명시적 stream 축을 가진 형태로 바꾼다.
        streams_4d = streams.contiguous().view(batch_size, self.num_streams, num_tokens, model_dim)
        # 마지막에는 stream 평균으로 [B, T, D]로 되돌린다.
        return streams_4d.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 일반 token 시퀀스를 multi-stream 표현으로 확장한다.
        streams = self._expand_streams(x)
        # 모든 mHC encoder layer를 순서대로 통과시킨다.
        for layer in self.layers:
            streams = layer(streams)
        # 최종적으로 다시 일반 [B, T, D] 표현으로 줄인다.
        return self._reduce_streams(streams)


class VectorFieldTransformer(nn.Module):
    """FM/iMF용 벡터 필드를 예측하는 Transformer encoder.

    입력:
    - x_t: 현재 상태 텐서 [B, D]
    - t_start: 시작 시간 텐서 [B, 1]
    - t_now: 현재 시간 텐서 [B, 1]

    하위 호환:
    - forward(x_t, t_now) 형태로 호출되면 t_start는 0으로 간주한다.
    """

    def __init__(
        self,
        input_dim: int = 784,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        patch_size: int = 2,
        image_size: int = 28,
        in_channels: int = 1,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        encoder_variant: str = "standard",
        num_streams: int = 4,
        sinkhorn_iters: int = 10,
        sinkhorn_tau: float = 0.05,
        residual_identity_mix: bool = False,
        residual_alpha: float = 0.01,
        variant: str = "fm",
    ) -> None:
        # 학습 가능한 벡터 필드 모델로 등록한다.
        super().__init__()
        # 출력 head 타입은 fm / imf 중 하나여야 한다.
        if variant not in {"fm", "imf"}:
            raise ValueError(f"Unknown variant: {variant}")
        # encoder는 기존 표준 버전과 mHC 버전만 지원한다.
        if encoder_variant not in {"standard", "mhc"}:
            raise ValueError(f"Unknown encoder_variant: {encoder_variant}")
        # multi-head attention은 model_dim이 num_heads로 나누어져야 한다.
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        # patch 크기는 양수여야 한다.
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        # image 크기는 양수여야 한다.
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        # non-overlapping patch를 쓰므로 image_size는 patch_size로 나누어져야 한다.
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        # 채널 수는 양수여야 한다.
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        # flatten된 input_dim은 실제 이미지 기하와 정확히 일치해야 한다.
        if in_channels * image_size * image_size != input_dim:
            raise ValueError(
                "input_dim must match in_channels * image_size * image_size "
                f"(got {input_dim} vs {in_channels}*{image_size}*{image_size})."
            )

        # flatten된 상태 차원 D를 저장한다.
        self.input_dim = int(input_dim)
        # 이미지 한 변의 길이 H = W를 저장한다.
        self.image_size = int(image_size)
        # 채널 수 C를 저장한다.
        self.in_channels = int(in_channels)
        # patch 한 변의 길이 P를 저장한다.
        self.patch_size = int(patch_size)
        # patch 하나가 가진 원소 수는 C * P * P다.
        self.patch_dim = int(in_channels * patch_size * patch_size)
        # 한 축에 들어가는 patch 개수.
        patches_per_side = image_size // patch_size
        # 전체 patch token 개수 T.
        self.num_tokens = int(patches_per_side * patches_per_side)
        # 현재 모델이 fm / imf 중 무엇을 예측하는지 저장한다.
        self.variant = variant
        # standard encoder인지 mhc encoder인지 저장한다.
        self.encoder_variant = encoder_variant

        # flatten된 patch를 model_dim token으로 투영한다.
        self.patch_in = nn.Linear(self.patch_dim, model_dim)
        # 각 patch token 위치마다 학습 가능한 positional embedding을 둔다.
        self.token_pos = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))

        # 시작 시간 r을 임베딩하는 MLP.
        self.start_time_embed = TimeEmbeddingMLP(time_embed_dim)
        # 현재 시간 t를 임베딩하는 MLP.
        self.now_time_embed = TimeEmbeddingMLP(time_embed_dim)
        # 두 시간 임베딩을 token 차원으로 융합하는 작은 MLP.
        self.time_fuse = nn.Sequential(
            nn.Linear(2 * time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # standard 모드는 PyTorch 기본 pre-norm TransformerEncoder를 쓴다.
        if self.encoder_variant == "standard":
            # 원하는 폭과 head 수를 가진 encoder layer 하나를 만든다.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # 같은 layer를 num_layers개 쌓아 encoder를 만든다.
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        else:
            # mhc 모드는 custom multi-stream encoder를 사용한다.
            self.encoder = MHCTransformerEncoder(
                model_dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_streams=num_streams,
                dropout=dropout,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_tau=sinkhorn_tau,
                residual_identity_mix=residual_identity_mix,
                residual_alpha=residual_alpha,
            )

        # FM용 head는 patch-wise velocity를 예측한다.
        self.v_head = nn.Linear(model_dim, self.patch_dim)
        # iMF용 head는 compute_V에서 쓰는 u field를 예측한다.
        self.u_head = nn.Linear(model_dim, self.patch_dim)

        # 모듈 트리 안의 모든 Linear 층에 보수적 초기화를 적용한다.
        self.apply(_init_linear_conservative)
        # token positional embedding은 작은 랜덤값으로 시작한다.
        nn.init.normal_(self.token_pos, mean=0.0, std=0.02)
        # iMF는 초반에 더 불안정할 수 있어 u-head를 아주 작게 시작시킨다.
        if self.variant == "imf":
            nn.init.normal_(self.u_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.u_head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        # 예전 호출 방식 forward(x_t, t_now)도 지원하기 위해 t_start를 0으로 보정한다.
        if t_now is None:
            t_now = t_start
            t_start = torch.zeros_like(t_now)

        # B는 flatten된 이미지 상태의 batch 크기다.
        bsz = x_t.size(0)
        # flat state [D]를 다시 이미지 [C, H, W]로 복원한다.
        x_2d = x_t.view(bsz, self.in_channels, self.image_size, self.image_size)
        # non-overlapping patch를 뽑아 [B, T, patch_dim] token 시퀀스로 만든다.
        x_tokens = F.unfold(
            x_2d,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        # raw patch 벡터를 model_dim token embedding으로 바꾼다.
        h = self.patch_in(x_tokens)

        # 시작 시간 r을 임베딩한다.
        t_start_emb = self.start_time_embed(t_start)
        # 현재 시간 t를 임베딩한다.
        t_now_emb = self.now_time_embed(t_now)
        # 두 시간 임베딩을 하나의 conditioning 벡터 [B, model_dim]로 합친다.
        time_cond = self.time_fuse(torch.cat([t_start_emb, t_now_emb], dim=1))
        # patch 내용, 위치 정보, 시간 조건을 더해 최종 encoder 입력을 만든다.
        h = h + self.token_pos + time_cond.unsqueeze(1)

        # standard 또는 mhc encoder를 통과시킨다.
        h = self.encoder(h)

        # 오래된 call site와 인터페이스를 맞추기 위해 인자는 유지한다.
        _ = return_dict

        # FM일 때는 v-head를 사용한다.
        if self.variant == "fm":
            out = self.v_head(h)
        # iMF일 때는 u-head를 사용한다.
        else:
            out = self.u_head(h)
        # fold는 [B, patch_dim, T] 형태를 기대하므로 축을 바꾼다.
        out_cols = out.transpose(1, 2)
        # patch 출력을 다시 이미지 공간 [B, C, H, W]로 접어 넣는다.
        out_2d = F.fold(
            out_cols,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # 나머지 코드와 인터페이스를 맞추기 위해 최종 출력을 다시 [B, D]로 flatten한다.
        return out_2d.reshape(bsz, self.input_dim)
