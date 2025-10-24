from __future__ import annotations
from typing import Optional, Sequence, Mapping, Tuple, List, Dict, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from collections import defaultdict
import math


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
            inv = sorted(self.class_to_int.items(), key=lambda kv: kv[1])
            self.class_names = [k for k, _ in inv]

        y_all = np.asarray([self.class_to_int[v] for v in y_raw], dtype=np.int64)
        self.y = y_all[self.indices]

        self.num_classes = len(self.class_names)
        self.class_counts = np.bincount(self.y, minlength=self.num_classes)
        self.class_probs = self.class_counts / max(1, self.class_counts.sum())

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


def _make_class_weighted_sampler(ds: AnnDataDatasetStreaming, replacement: bool = True) -> Sampler[int]:
    counts = ds.class_counts
    counts = np.maximum(counts, 1)
    inv_freq = 1.0 / counts
    w_per_class = inv_freq / inv_freq.sum()
    sample_weights = w_per_class[ds.y]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(ds), replacement=replacement)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels: np.ndarray, num_classes: int, samples_per_class: int, generator: Optional[torch.Generator] = None):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_classes = int(num_classes)
        self.k = int(samples_per_class)
        self.batch_size = self.k * self.num_classes
        self.generator = generator

        self._by_class: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for idx, y in enumerate(self.labels):
            self._by_class[int(y)].append(idx)

        for c in range(self.num_classes):
            if c not in self._by_class:
                self._by_class[c] = []

        self._cursors = {c: 0 for c in range(self.num_classes)}
        self._shuffles: Dict[int, List[int]] = {}
        self._reshuffle_all()

        max_class_len = max(1, max(len(v) for v in self._by_class.values()))
        self._num_batches = math.ceil(max_class_len / self.k)

    def _rng_perm(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.int64)
        if self.generator is None:
            return np.random.permutation(n).astype(np.int64)
        perm = torch.randperm(n, generator=self.generator).cpu().numpy()
        return perm.astype(np.int64)

    def _reshuffle_all(self):
        self._shuffles.clear()
        for c, idxs in self._by_class.items():
            if len(idxs) == 0:
                self._shuffles[c] = []
            else:
                perm = self._rng_perm(len(idxs))
                self._shuffles[c] = [idxs[i] for i in perm]
            self._cursors[c] = 0

    def __iter__(self):
        self._reshuffle_all()
        for _ in range(self._num_batches):
            batch: List[int] = []
            for c in range(self.num_classes):
                pool = self._shuffles[c]
                if len(pool) == 0:
                    continue
                cur = self._cursors[c]
                need = self.k
                take = []
                while need > 0:
                    left = len(pool) - cur
                    if left >= need:
                        take.extend(pool[cur:cur+need])
                        cur += need
                        need = 0
                    else:
                        if left > 0:
                            take.extend(pool[cur:])
                        perm = self._rng_perm(len(self._by_class[c]))
                        pool = [self._by_class[c][i] for i in perm]
                        cur = 0
                        self._shuffles[c] = pool
                self._cursors[c] = cur
                batch.extend(take)
            if len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        return self._num_batches
# ---------------------------
def make_dataloader_streaming(
    adata,
    layer: Optional[str],
    indices,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    obs_key: str = "Malignant",
    class_to_int: Optional[Mapping] = None,

    sampling_strategy: str = "class_weighted",
    samples_per_class: int = 8,
    replacement: bool = True,
    generator: Optional[torch.Generator] = None,
):
    ds = AnnDataDatasetStreaming(adata, layer=layer, indices=indices, obs_key=obs_key, class_to_int=class_to_int)

    sampler = None
    batch_sampler = None
    effective_batch_size = batch_size

    if sampling_strategy == "uniform":
        pass

    elif sampling_strategy == "class_weighted":
        sampler = _make_class_weighted_sampler(ds, replacement=replacement)
        shuffle = False

    elif sampling_strategy == "balanced_batch":
        sampler = None
        shuffle = False
        k = max(1, int(samples_per_class))
        batch_sampler = BalancedBatchSampler(labels=ds.y, num_classes=ds.num_classes, samples_per_class=k, generator=generator)
        effective_batch_size = k * ds.num_classes

    else:
        raise ValueError("sampling_strategy must be one of: 'uniform', 'class_weighted', 'balanced_batch'")

    loader = DataLoader(
        ds,
        batch_size=None if batch_sampler is not None else effective_batch_size,
        shuffle=(shuffle and sampler is None and batch_sampler is None),
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_dense_float32,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        generator=generator,
    )
    return loader, ds
