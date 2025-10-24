from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy, AUROC
from typing import Optional, Sequence


class MLPClassifier(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None,
        class_names: Sequence[str] = ("TME", "Malignant"),
        threshold: float = 0.5,
        lr_min: float = 1e-5,
        wd_min: Optional[float] = None,
        wd_max: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        if len(class_names) != 2:
            raise ValueError("class_names must have length 2 for binary classification.")
        self.class_names = list(class_names)
        self.register_buffer("_classes_idx", torch.arange(2))

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.acc = BinaryAccuracy()
        self.auroc = AUROC(task="binary")

        # cache thresholds
        self._threshold = float(threshold)

        self._lr0 = float(lr)
        self._lr_min = float(lr_min)
        wd0 = weight_decay
        self._wd_max = float(wd0 if wd_max is None else wd_max)
        self._wd_min = float(wd0 if wd_min is None else wd_min)

    @property
    def classes_(self):
        return self.class_names

    def decode(self, idx_tensor_or_ndarray):
        idx = idx_tensor_or_ndarray.detach().cpu().numpy() if torch.is_tensor(idx_tensor_or_ndarray) else np.asarray(idx_tensor_or_ndarray)
        return np.asarray(self.class_names, dtype=object)[idx]

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc",  self.acc(probs, y), prog_bar=True, on_epoch=True)
        try:
            self.log(f"{stage}_auroc", self.auroc(probs, y), prog_bar=False, on_epoch=True)
        except ValueError:
            # AUROC undefined if a batch has a single class only
            pass
        return loss

    def training_step(self, b, i):
        return self._shared_step(b, "train")

    def validation_step(self, b, i):
        self._shared_step(b, "val")

    def test_step(self, b, i):
        self._shared_step(b, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        t_max = max(int(getattr(self.trainer, "max_epochs", 1)), 1)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t_max, eta_min=float(self.hparams.lr_min)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_lr",
                "monitor": None,
            },
        }

    def on_fit_start(self):
        self._apply_weight_decay(self._wd_max)

    def on_train_epoch_start(self):
        max_epochs = max(int(self.trainer.max_epochs), 1)
        e = self.current_epoch
        progress = 0.0 if max_epochs <= 1 else float(e) / float(max_epochs - 1)
        wd = self._cosine_decay(self._wd_max, self._wd_min, progress)
        self._apply_weight_decay(wd)
        self.log("wd", wd, prog_bar=False, on_epoch=True)

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
    def predict_proba(self, x):
        logits = self(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict01(self, x, threshold: Optional[float] = None):
        thr = self._threshold if threshold is None else float(threshold)
        probs = self.predict_proba(x)
        return (probs > thr).long(), probs

    @torch.no_grad()
    def predict_labels(self, x, threshold: Optional[float] = None, return_probs: bool = False):
        preds01, probs = self.predict01(x, threshold=threshold)
        labels = self.decode(preds01)
        if return_probs:
            return labels, probs.detach().cpu().numpy()
        return labels

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        preds01, probs = self.predict01(x)
        labels = self.decode(preds01)
        return {
            "pred_labels": labels,
            "pred01": preds01.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
            "classes": np.array(self.class_names, dtype=object),
        }


