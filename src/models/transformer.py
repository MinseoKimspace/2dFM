"""Transformer backbone for FM/iMF vector field prediction."""

from __future__ import annotations

# `torch` provides tensors, autograd, and parameter initialization utilities.
import torch
# `nn` contains model building blocks such as `Linear` and `TransformerEncoder`.
import torch.nn as nn
# `functional` provides tensor ops such as `unfold` and `fold`.
import torch.nn.functional as F

# This MLP turns scalar times into learned embeddings.
from models.time_embed import TimeEmbeddingMLP


def _init_linear_conservative(module: nn.Module) -> None:
    """Initialize linear layers with a small standard deviation."""
    # Only touch linear layers and leave all other modules unchanged.
    if isinstance(module, nn.Linear):
        # `fan_in` is the number of input features for the layer.
        fan_in = module.weight.size(1)
        # Use a small variance so early training stays stable.
        std = (0.1 / float(fan_in)) ** 0.5
        # Sample weights from a zero-mean Gaussian.
        nn.init.normal_(module.weight, mean=0.0, std=std)
        # Start biases at zero so the layer is centered initially.
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VectorFieldTransformer(nn.Module):
    """Patch-based Transformer that predicts FM/iMF vector fields."""

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
        variant: str = "fm",
    ) -> None:
        # Register all submodules and parameters with PyTorch.
        super().__init__()

        # The model only supports the two output heads used in this repo.
        if variant not in {"fm", "imf"}:
            raise ValueError(f"Unknown variant: {variant}")
        # Multi-head attention requires an even split across heads.
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        # Patch size must be a positive integer.
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        # Image size must be a positive integer.
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        # We use non-overlapping patches, so the image must divide cleanly.
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        # Channel count must be valid.
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        # The flattened input dimension must match the image geometry.
        if in_channels * image_size * image_size != input_dim:
            raise ValueError(
                "input_dim must match in_channels * image_size * image_size "
                f"(got {input_dim} vs {in_channels}*{image_size}*{image_size})."
            )

        # Save the flattened state dimension so the output can match the input shape.
        self.input_dim = int(input_dim)
        # Save the image side length for `view` and `fold`.
        self.image_size = int(image_size)
        # Save the number of channels for image reshape logic.
        self.in_channels = int(in_channels)
        # Save the patch side length.
        self.patch_size = int(patch_size)
        # Each patch contains `C * P * P` scalars.
        self.patch_dim = int(in_channels * patch_size * patch_size)
        # Count how many patches fit on one image side.
        patches_per_side = image_size // patch_size
        # Total number of patch tokens is the 2D grid area.
        self.num_tokens = int(patches_per_side * patches_per_side)
        # Remember which output head to use in `forward`.
        self.variant = variant

        # Project each raw patch vector into the Transformer hidden width.
        self.patch_in = nn.Linear(self.patch_dim, model_dim)
        # Learn a positional embedding for every patch token.
        self.token_pos = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))

        # Embed the start time `r`.
        self.start_time_embed = TimeEmbeddingMLP(time_embed_dim)
        # Embed the current time `t`.
        self.now_time_embed = TimeEmbeddingMLP(time_embed_dim)
        # Fuse the two time embeddings into one conditioning vector.
        self.time_fuse = nn.Sequential(
            # First project concatenated time features into model width.
            nn.Linear(2 * time_embed_dim, model_dim),
            # Use SiLU for a smooth nonlinearity.
            nn.SiLU(),
            # Project again so the fused time code matches token width.
            nn.Linear(model_dim, model_dim),
        )

        # Build a standard pre-norm Transformer encoder block.
        encoder_layer = nn.TransformerEncoderLayer(
            # Hidden width for every token.
            d_model=model_dim,
            # Number of self-attention heads.
            nhead=num_heads,
            # Width of the feed-forward sublayer.
            dim_feedforward=ff_dim,
            # Dropout used inside attention and FFN blocks.
            dropout=dropout,
            # Use GELU to match common vision Transformer setups.
            activation="gelu",
            # Tokens are stored as `[batch, tokens, dim]`.
            batch_first=True,
            # Keep pre-norm behavior because it is usually more stable.
            norm_first=True,
        )
        # Disable nested tensor optimization to silence the PyTorch warning for pre-norm layers.
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # FM predicts a velocity field in patch space.
        self.v_head = nn.Linear(model_dim, self.patch_dim)
        # iMF predicts the alternative `u` field in patch space.
        self.u_head = nn.Linear(model_dim, self.patch_dim)

        # Apply the conservative initialization to every linear layer in the module.
        self.apply(_init_linear_conservative)
        # Initialize token positions with a small random scale.
        nn.init.normal_(self.token_pos, mean=0.0, std=0.02)
        # The iMF head starts especially small to reduce unstable early updates.
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
        # Support the legacy call style `forward(x_t, t_now)` by treating `t_start` as zero.
        if t_now is None:
            t_now = t_start
            t_start = torch.zeros_like(t_now)

        # Read the batch size from the flattened input tensor `[B, D]`.
        bsz = x_t.size(0)
        # Reshape flat vectors back into image tensors `[B, C, H, W]`.
        x_2d = x_t.view(bsz, self.in_channels, self.image_size, self.image_size)
        # Extract non-overlapping patches and move them to `[B, T, patch_dim]`.
        x_tokens = F.unfold(
            x_2d,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        # Project raw patch vectors into Transformer token embeddings.
        h = self.patch_in(x_tokens)

        # Embed the start time for every batch element.
        t_start_emb = self.start_time_embed(t_start)
        # Embed the current time for every batch element.
        t_now_emb = self.now_time_embed(t_now)
        # Combine both time embeddings into one conditioning vector `[B, model_dim]`.
        time_cond = self.time_fuse(torch.cat([t_start_emb, t_now_emb], dim=1))
        # Add positional encoding and broadcasted time conditioning to every token.
        h = h + self.token_pos + time_cond.unsqueeze(1)

        # Run the full token sequence through the Transformer stack.
        h = self.encoder(h)

        # Keep this argument for call-site compatibility even though the model returns a tensor.
        _ = return_dict
        # Pick the FM or iMF output head.
        out = self.v_head(h) if self.variant == "fm" else self.u_head(h)
        # Transpose to `[B, patch_dim, T]` because `fold` expects channel-first patches.
        out_cols = out.transpose(1, 2)
        # Reconstruct the patch outputs back into image layout `[B, C, H, W]`.
        out_2d = F.fold(
            out_cols,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # Flatten the image-shaped output back to `[B, D]` for the rest of the codebase.
        return out_2d.reshape(bsz, self.input_dim)
