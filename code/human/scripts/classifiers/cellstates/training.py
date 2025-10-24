from __future__ import annotations
import numpy as np

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from models import MLPClassifier
from dataset import make_dataloader_streaming


def train_classifier(
    adata,
    layer: str | None = None,
    batch_size: int = 256,
    max_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_sizes=(1024, 512, 256),
    dropout: float = 0.2,
    val_size: float = 0.1,
    test_size: float = 0.05,
    seed: int = 42,
    num_workers: int = 2,
    accelerator: str = "auto",
    devices: int | str = "auto",
    obs_key: str = "CellState",
    balance_classes: bool = True,
):
    seed_everything(seed)

    Xref = adata.layers[layer] if layer is not None else adata.X
    n_features = Xref.shape[1]
    labels_all = adata.obs[obs_key].astype(str).values
    idx_all = np.arange(adata.n_obs, dtype=np.int64)

    idx_train, idx_temp, y_train_labels, y_temp_labels = train_test_split(
        idx_all, labels_all,
        test_size=val_size + test_size,
        stratify=labels_all,
        random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp_labels,
        test_size=rel_test,
        stratify=y_temp_labels,
        random_state=seed,
    )

    train_loader, train_dataset = make_dataloader_streaming(
        adata, layer, idx_train,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, obs_key=obs_key,
        sampling_strategy='class_weighted'
    )
    class_to_int = train_dataset.class_to_int
    class_names = train_dataset.class_names

    val_loader, _ = make_dataloader_streaming(
        adata, layer, idx_val,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, obs_key=obs_key, class_to_int=class_to_int,
        sampling_strategy='uniform'
    )
    test_loader, _ = make_dataloader_streaming(
        adata, layer, idx_test,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, obs_key=obs_key, class_to_int=class_to_int,
        sampling_strategy='uniform'
    )

    class_weights = None
    if balance_classes:
        classes_idx = np.arange(len(class_names))
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes_idx,
            y=train_dataset.y
        ).astype(np.float32)

    model = MLPClassifier(
        input_dim=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        class_names=class_names,
        class_weights=class_weights.tolist() if class_weights is not None else None,
    )

    ckpt_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="mlp-{epoch:02d}-{val_acc:.4f}")
    es_cb = EarlyStopping(monitor="val_acc", mode="max", patience=30)
    logger = CSVLogger(save_dir="logs", name="mlp_flat")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1 if devices == "auto" else devices,
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
