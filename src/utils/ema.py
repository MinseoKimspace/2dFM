from __future__ import annotations

from typing import Dict, Optional

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = state_dict
