"""Training routines for classifiers.

Provides train_classifier(...) that wires dataset, model and lightning Trainer.
"""
from __future__ import annotations
import numpy as np
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from models import MLPClassifier
from dataset import make_dataloader_streaming


def train_classifier(
    adata,
    layer: str | None = None,
    batch_size: int = 256,
    max_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_sizes=(128, 64, 32),
    dropout: float = 0.2,
    val_size: float = 0.1,
    test_size: float = 0.05,
    seed: int = 42,
    num_workers: int = 2,
    accelerator: str = "auto",
    devices: int | str = "auto",
):
    seed_everything(seed)

    Xref = adata.layers[layer] if layer is not None else adata.X
    n_features = Xref.shape[1]
    y_all = (adata.obs["Malignant"].values == "Malignant").astype(np.int64)
    idx_all = np.arange(adata.n_obs, dtype=np.int64)

    from sklearn.model_selection import train_test_split
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, y_all, test_size=val_size + test_size, stratify=y_all, random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed
    )

    train_loader, _ = make_dataloader_streaming(
        adata, layer, idx_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        class_to_int={"TME": 0, "Malignant": 1},  # <- add this
    )
    val_loader, _ = make_dataloader_streaming(
        adata, layer, idx_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        class_to_int={"TME": 0, "Malignant": 1},
    )
    test_loader, _ = make_dataloader_streaming(
        adata, layer, idx_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        class_to_int={"TME": 0, "Malignant": 1},
    )

    n_pos = int(y_train.sum()); n_neg = int(y_train.shape[0] - n_pos)
    pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

    model = MLPClassifier(
        input_dim=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
    )

    ckpt_cb = ModelCheckpoint(monitor="val_auroc", mode="max", save_top_k=1, filename="mlp-{epoch:02d}-{val_auroc:.4f}")
    es_cb = EarlyStopping(monitor="val_auroc", mode="max", patience=5)
    logger = CSVLogger(save_dir="logs", name="malignant_mlp")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = ckpt_cb.best_model_path
    if best_path:
        model = MLPClassifier.load_from_checkpoint(best_path)

    trainer.test(model, dataloaders=test_loader)
    return model, (train_loader, val_loader, test_loader), trainer


__all__ = ["train_classifier"]
