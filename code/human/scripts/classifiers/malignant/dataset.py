from __future__ import annotations
from typing import Optional, Sequence, Mapping
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AnnDataDatasetStreaming(Dataset):
    def __init__(self, adata, layer: Optional[str] = None, indices: Optional[Sequence[int]] = None,
                 obs_key: str = "Malignant", class_to_int: Optional[Mapping] = None):
        self.adata = adata
        self.layer = layer
        self.X = adata.layers[layer] if layer is not None else adata.X
        if indices is None:
            self.indices = np.arange(self.X.shape[0], dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        if obs_key not in adata.obs:
            raise KeyError(f"obs_key '{obs_key}' not found in adata.obs")

        y_raw = np.asarray(adata.obs[obs_key].values, dtype=object)
        unique = np.unique(y_raw).astype(object)
        unique_sorted = list(unique)

        if class_to_int is None:
            self.class_names = unique_sorted
            self.class_to_int = {c: i for i, c in enumerate(self.class_names)}
        else:
            self.class_to_int = dict(class_to_int)
            self.class_names = [None] * len(self.class_to_int)
            for k, v in self.class_to_int.items():
                if v < 0:
                    raise ValueError("class_to_int mapping values must be >= 0")
                if v < len(self.class_names):
                    self.class_names[v] = k
            self.class_names = [c for c in self.class_names if c is not None]

        y_all = np.asarray([self.class_to_int[v] for v in y_raw], dtype=np.int64)
        self.y = y_all[self.indices]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.X[idx]
        if hasattr(row, "toarray"):
            x = row.toarray().ravel().astype(np.float32, copy=False)
        else:
            x = np.asarray(row, dtype=np.float32)
        y = np.int64(self.y[i])
        return x, y


def _collate_dense_float32(batch):
    xs, ys = zip(*batch)
    X = torch.from_numpy(np.stack(xs, axis=0))
    y = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    return X, y


def make_dataloader_streaming(adata, layer: Optional[str], indices, batch_size: int = 256, shuffle: bool = True,
                              num_workers: int = 2, pin_memory: bool = True, prefetch_factor: int = 2, persistent_workers: bool = True,
                              obs_key: str = "Malignant", class_to_int: Optional[Mapping] = None):
    ds = AnnDataDatasetStreaming(adata, layer=layer, indices=indices, obs_key=obs_key, class_to_int=class_to_int)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_dense_float32,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    return loader, ds


__all__ = ["AnnDataDatasetStreaming", "make_dataloader_streaming"]
