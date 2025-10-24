from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassAccuracy
from typing import Optional, Sequence


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = None if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        out = self.act(out)
        out = self.drop(out)
        skip = x if self.proj is None else self.proj(x)
        return out + skip


class MLPClassifier(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        class_names: Optional[Sequence[str]] = None,
        class_weights: Optional[Sequence[float]] = None,
        lr_min: float = 1e-5,
        wd_min: Optional[float] = None,
        wd_max: Optional[float] = None,
    ):
        super().__init__()

        hidden_sizes = tuple(hidden_sizes)
        if class_names is None:
            raise ValueError("class_names must be provided for multiclass classification.")
        class_names = tuple(class_names)
        if len(class_names) < 2:
            raise ValueError("class_names must contain at least two entries.")
        if class_weights is not None:
            class_weights = tuple(float(w) for w in class_weights)

        self.save_hyperparameters()

        self.class_names = list(self.hparams.class_names)
        self.num_classes = len(self.class_names)
        self.register_buffer("_classes_idx", torch.arange(self.num_classes), persistent=False)

        blocks: list[nn.Module] = []
        prev_dim = self.hparams.input_dim
        for h in self.hparams.hidden_sizes:
            blocks.append(ResidualBlock(prev_dim, h, dropout=self.hparams.dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()

        self.head = nn.Linear(prev_dim, self.num_classes)

        if self.hparams.class_weights is not None:
            weight_tensor = torch.tensor(self.hparams.class_weights, dtype=torch.float32)
            self.register_buffer("class_weights_tensor", weight_tensor)
            loss_weight = self.class_weights_tensor
        else:
            self.class_weights_tensor = None
            loss_weight = None
        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

        self.metrics = nn.ModuleDict({
            f"m_{stage}": nn.ModuleDict({
                "acc": MulticlassAccuracy(num_classes=self.num_classes),
            })
            for stage in ("train", "val", "test")
        })
        self._stage_metrics_key = {stage: f"m_{stage}" for stage in ("train", "val", "test")}

        self._lr0 = float(self.hparams.lr)
        self._lr_min = float(self.hparams.lr_min)
        wd0 = self.hparams.weight_decay
        self._wd_max = float(wd0 if self.hparams.wd_max is None else self.hparams.wd_max)
        self._wd_min = float(wd0 if self.hparams.wd_min is None else self.hparams.wd_min)

    @property
    def classes_(self):
        return self.class_names

    def decode(self, idx_tensor_or_ndarray):
        idx = idx_tensor_or_ndarray.detach().cpu().numpy() if torch.is_tensor(idx_tensor_or_ndarray) else np.asarray(idx_tensor_or_ndarray)
        return np.asarray(self.class_names, dtype=object)[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        on_step = (stage == "train")
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=(stage == "train"), batch_size=x.size(0))

        acc = self.metrics[self._stage_metrics_key[stage]]["acc"](logits, y)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): self._shared_step(batch, "val")
    def test_step(self, batch, batch_idx): self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        t_max = max(int(getattr(self.trainer, "max_epochs", 1)), 1)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=float(self.hparams.lr_min))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "epoch", "frequency": 1, "name": "cosine_lr", "monitor": None}}

    def on_fit_start(self): self._apply_weight_decay(self._wd_max)
    def on_train_epoch_start(self):
        max_epochs = max(int(self.trainer.max_epochs), 1)
        e = self.current_epoch
        progress = 0.0 if max_epochs <= 1 else float(e) / float(max_epochs - 1)
        wd = self._cosine_decay(self._wd_max, self._wd_min, progress)
        self._apply_weight_decay(wd)
        self.log("wd", wd, prog_bar=False, on_step=False, on_epoch=True)

    @staticmethod
    def _cosine_decay(start: float, end: float, progress01: float) -> float:
        cos_term = 0.5 * (1.0 + math.cos(math.pi * progress01))
        return end + (start - end) * cos_term

    def _apply_weight_decay(self, wd: float):
        opt = self.optimizers(use_pl_optimizer=False)
        if opt is None:
            return
        for pg in opt.param_groups:
            pg["weight_decay"] = float(wd)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_indices(self, x: torch.Tensor):
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1), probs

    @torch.no_grad()
    def predict_labels(self, x: torch.Tensor, return_probs: bool = False):
        pred_idx, probs = self.predict_indices(x)
        labels = self.decode(pred_idx)
        if return_probs:
            return labels, probs.detach().cpu().numpy()
        return labels

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        pred_idx, probs = self.predict_indices(x)
        labels = self.decode(pred_idx)
        return {
            "pred_labels": labels,
            "pred_indices": pred_idx.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
            "classes": np.array(self.class_names, dtype=object),
        }
