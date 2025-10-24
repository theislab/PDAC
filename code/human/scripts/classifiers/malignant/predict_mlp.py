import argparse
import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

from dataset import make_dataloader_streaming 

try:
    import anndata as ad
except Exception as e:
    print("ERROR: anndata is required. pip install anndata scanpy zarr", file=sys.stderr)
    raise

def import_object(dotted_path: str):
    import importlib
    if "." not in dotted_path:
        raise ValueError(f"Expected dotted path like 'module.ClassName', got: {dotted_path}")
    module_name, obj_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    if not hasattr(mod, obj_name):
        raise AttributeError(f"Module '{module_name}' has no attribute '{obj_name}'.")
    return getattr(mod, obj_name)


def load_adata(path: str):
    if path.endswith(".zarr"):
        return ad.read_zarr(path)
    try:
        return ad.read_zarr(path)
    except Exception:
        return ad.read_h5ad(path)


def prepare_adata_for_inference(adata, layer: Optional[str] = None):
    if "Level_4" in adata.obs.columns and "Malignant" not in adata.obs.columns:
        adata.obs["Malignant"] = [
            "Malignant" if isinstance(cell, str) and "Malignant" in cell else "TME"
            for cell in adata.obs["Level_4"]
        ]

    if "Manual_Genes" in adata.var.columns:
        mask = adata.var["Manual_Genes"]
        if mask.dtype == bool:
            mask = mask.values
        else:
            mask = (adata.var["Manual_Genes"].astype(str) == "True").values
        adata = adata[:, mask].copy()

    return adata


def build_dataloader(adata, layer: Optional[str], batch_size: int):
    class_to_int = {"TME": 0, "Malignant": 1}
    try:
        dl, _ = make_dataloader_streaming(
            adata,
            layer=layer,
            indices=None,
            batch_size=batch_size,
            shuffle=False,
            class_to_int=class_to_int,
        )
    except TypeError:
        dl, _ = make_dataloader_streaming(
            adata,
            layer=layer,
            indices=None,
            batch_size=batch_size,
            shuffle=False,
        )
    return dl


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def ensure_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_load_model(args) -> torch.nn.Module:
    ModelClass = import_object(args.model_class)

    if args.model_path.endswith(".ckpt") and hasattr(ModelClass, "load_from_checkpoint"):
        try:
            model = ModelClass.load_from_checkpoint(args.model_path, map_location="cpu")
            return model
        except Exception as e:
            print(f"Warning: load_from_checkpoint failed: {e}. Will try state_dict route...", file=sys.stderr)

    return ModelClass


def predict_with_model(model: torch.nn.Module, dl, device: torch.device, include_probs: bool):
    model.eval()
    model.to(device)

    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for xb, *_ in dl:
            xb = xb.to(device)
            if hasattr(model, "predict_labels"):
                labels, probs = model.predict_labels(xb, return_probs=True)  # type: ignore
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                if isinstance(probs, torch.Tensor):
                    probs = probs.detach().cpu().numpy()
            else:
                logits = model(xb)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs_t = softmax(logits)
                probs = probs_t.detach().cpu().numpy()
                labels = probs.argmax(axis=-1)

            all_preds.append(np.asarray(labels))
            if include_probs:
                all_probs.append(np.asarray(probs, dtype=float))

    pred_arr = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=object)
    probs_arr = np.concatenate(all_probs, axis=0) if include_probs and all_probs else None
    return pred_arr, probs_arr


def main():
    p = argparse.ArgumentParser(description="Run inference for MLP malignant classifier on an AnnData file.")
    p.add_argument("adata_path", help="Path to AnnData (.zarr or .h5ad)")
    p.add_argument("--layer", default=None, help="Layer name to use (default: adata.X if None)")
    p.add_argument("--model-path", required=True, help="Path to trained model checkpoint (.ckpt) or state dict (.pt)")
    p.add_argument("--model-class", default="training.MLPClassifier",
                   help="Dotted path to model class for loading (default: training.MLPClassifier)")
    p.add_argument("--out", default="predictions.csv", help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--include-probs", action="store_true", help="Also export per-class probabilities")
    args = p.parse_args()

    # 1) Load AnnData and mirror preprocessing
    adata = load_adata(args.adata_path)
    adata = prepare_adata_for_inference(adata, layer=args.layer)

    device = ensure_device()
    print(f"Using device: {device}", file=sys.stderr)

    # 2) Build dataloader (no shuffle to keep order == obs_names order)
    dl = build_dataloader(adata, layer=args.layer, batch_size=args.batch_size)

    # 3) Load model
    maybe_model_or_class = try_load_model(args)
    if isinstance(maybe_model_or_class, type):
        # Need to instantiate with correct input size if the constructor requires it.
        # Try to infer input dimension from the first batch.
        first_batch = next(iter(dl))[0]
        input_dim = int(first_batch.shape[-1])
        try:
            model = maybe_model_or_class(input_dim=input_dim)  # common pattern
        except TypeError:
            # Try no-arg construction
            model = maybe_model_or_class()
        # Load weights
        state = torch.load(args.model_path, map_location="cpu")
        # Lightning .ckpt typically stores under 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            # if keys are 'model.xxx', try to strip a common prefix
            model_state = model.state_dict()
            adapted_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model.") and not any(s.startswith("model.") for s in model_state.keys()):
                    nk = nk[len("model."):]
                adapted_state[nk] = v
            missing, unexpected = model.load_state_dict(adapted_state, strict=False)
            if missing:
                print(f"Warning: missing keys when loading: {missing}", file=sys.stderr)
            if unexpected:
                print(f"Warning: unexpected keys when loading: {unexpected}", file=sys.stderr)
        else:
            # Plain state_dict.pt
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys when loading: {missing}", file=sys.stderr)
            if unexpected:
                print(f"Warning: unexpected keys when loading: {unexpected}", file=sys.stderr)
    else:
        model = maybe_model_or_class  # already loaded via load_from_checkpoint

    # 4) Run predictions
    pred_labels, probs = predict_with_model(model, dl, device=device, include_probs=args.include_probs)

    # 5) Build output DataFrame
    obs_names = np.asarray(adata.obs_names)
    if len(obs_names) != len(pred_labels):
        # As a fallback, try to truncate to the min length
        n = min(len(obs_names), len(pred_labels))
        print(
            f"Warning: obs_names length ({len(obs_names)}) != predictions length ({len(pred_labels)}). "
            f"Truncating to {n}.",
            file=sys.stderr,
        )
        obs_names = obs_names[:n]
        pred_labels = pred_labels[:n]
        if probs is not None:
            probs = probs[:n, :]

    # Normalize predictions to strings.
    pred_labels = np.asarray(pred_labels)
    if np.issubdtype(pred_labels.dtype, np.integer):
        id_to_class = {0: "TME", 1: "Malignant"}
        pred_str = [id_to_class.get(int(i), str(i)) for i in pred_labels]
    else:
        pred_str = pred_labels.astype(str).tolist()

    out_df = pd.DataFrame({
        "obs_name": obs_names,
        "prediction": pred_str,
    })

    # Build probability columns if requested
    if args.include_probs and probs is not None:
        # Try to use class names from the model if available
        class_names = None
        for attr in ("class_names_", "classes_", "label_names_", "classes"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                try:
                    class_names = list(val)
                except Exception:
                    class_names = None
                if class_names:
                    break

        if class_names is not None and probs.shape[1] == len(class_names):
            for j, name in enumerate(class_names):
                out_df[f"prob_{str(name)}"] = probs[:, j]
        elif probs.shape[1] == 2:
            out_df["prob_TME"] = probs[:, 0]
            out_df["prob_Malignant"] = probs[:, 1]
        else:
            for j in range(probs.shape[1]):
                out_df[f"prob_class{j}"] = probs[:, j]

    # 6) Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = os.path.abspath(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()