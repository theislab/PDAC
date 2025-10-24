import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from dataset import make_dataloader_streaming
from training import train_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _save_confusion_matrices(cm_df, out_dir: str, prefix: str = ""):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _plot_and_save(plot_df, title, filename, vmin=None, vmax=None, fmt=".0f"):
        fig, ax = plt.subplots(
            figsize=(max(6, len(plot_df) * 0.6), max(5, len(plot_df) * 0.5))
        )
        im = ax.imshow(plot_df.values, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(plot_df.columns)))
        ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_yticklabels(plot_df.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                ax.text(j, i, f"{plot_df.values[i, j]:{fmt}}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = os.path.join(out_dir, filename if not prefix else f"{prefix}_{filename}")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved {filename} -> {path}")

    _plot_and_save(
        cm_df.astype(float),
        f"{prefix} Confusion Matrix (counts)" if prefix else "Confusion Matrix (counts)",
        "confusion_matrix_counts.png",
    )
    norm = cm_df.div(cm_df.sum(axis=1).replace(0, 1), axis=0)
    _plot_and_save(
        norm,
        f"{prefix} Confusion Matrix (row-normalized)" if prefix else "Confusion Matrix (row-normalized)",
        "confusion_matrix_norm.png",
        vmin=0.0, vmax=1.0, fmt=".2f",
    )

def _save_overall_metrics_bar(metrics_dict: dict, out_dir: str, filename: str = "overall_metrics_bar.png", title: str = "Overall metrics (full predictions)"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    keys = list(metrics_dict.keys())
    vals = [float(metrics_dict[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2.0, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved overall metrics barplot -> {out_path}")


PALETTE_L3: Dict[str, str] = {
    "Macrophage":                   "#8b4513",
    "Acinar (REG+) Cell":           "#8a2be2",
    "Malignant Cell - Mesenchymal": "#483d8b",
    "B Cell":                       "#b03060",
    "Cancer Associated Fibroblast": "#dc143c",
    "Malignant Cell - Epithelial":  "#008b8b",
    "Monocyte":                     "#808000",
    "Epsilon Cell":                 "#ff8c00",
    "CD4+ T Cell":                  "#8fbc8f",
    "Ductal Cell":                  "#ffd700",
    "Ductal Cell (atypical)":       "#00ff00",
    "Fibroblast":                   "#dc143c",
    "NK Cell":                      "#ff4500",
    "Mixed T Cell":                 "#800080",
    "Gamma Cell":                   "#add8e6",
    "CD8+ T Cell":                  "#00008b",
    "Neutrophil":                   "#006400",
    "Malignant Cell - EMT":         "#bc8f8f",
    "Endothelial Cell":             "#dda0dd",
    "Alpha Cell":                   "#1e90ff",
    "ADM Cell":                     "#00ff00",
    "Dendritic Cell":               "#00fa9a",
    "Beta Cell":                    "#ff00ff",
    "Acinar Idling Cell":           "#dc143c",
    "Delta Cell":                   "#00bfff",
    "Acinar Cell":                  "#00ffff",
    "Plasma Cell":                  "#0000ff",
    "Adipocyte":                    "#adff2f",
    "Mast Cell":                    "#ffe4c4",
    "Other Endocrine":              "#000000",
    "Pericyte":                     "#ff1493",
    "Schwann Cell":                 "#a5a5a5",
    "Smooth Muscle Cell":           "#b22222",
}


def _apply_palette_to_key(adata, key: str, palette_map: Dict[str, str]):
    if key not in adata.obs.columns:
        return
    cats = adata.obs[key].astype("category").cat.categories
    colors: List[str] = []
    for c in cats:
        colors.append(palette_map.get(c, "#808080"))
    adata.uns[f"{key}_colors"] = colors


def _plot_umap(adata, color_key: str, out_path: str, title: str):
    import scanpy as sc
    fig = sc.pl.umap(
        adata,
        color=color_key,
        legend_loc=None,
        show=False,
        na_color="white",
        outline_width=(0.1, 0.05),
        add_outline=True,
        title=title,
        frameon=False,
        return_fig=True,
    )
    fig.savefig(out_path, dpi=300)
    print(f"Saved UMAP -> {out_path}")

def _assert_mapping_consistency(model, class_to_int_from_loader):
    assert set(model.classes_) == set(class_to_int_from_loader.keys()), \
        "Model classes and dataset class_to_int keys differ."
    n = len(model.classes_)
    ds_int2class = [None] * n
    for cls, idx in class_to_int_from_loader.items():
        ds_int2class[idx] = cls
    assert list(model.classes_) == ds_int2class, \
        "Model.classes_ order != dataset class_to_int order."
        

def main():
    p = argparse.ArgumentParser(
        description="Train separate MLPs for malignant vs healthy cells and predict the whole AnnData."
    )
    p.add_argument("adata_path", help="Path to AnnData (zarr, h5ad, etc.)")
    p.add_argument("--layer", default=None, help="Layer name to use (default: adata.X if None)")
    p.add_argument("--out", default="models", help="Directory to save model checkpoints and artifacts")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.9, help="Fraction of cells used for model fitting (rest held out for evaluation)")
    p.add_argument("--eval-batch-size", type=int, default=128, help="Batch size for evaluation on the held-out set")

    p.add_argument("--obs-key", default="Level_3", help="obs column containing the class labels")
    p.add_argument("--malignant-pattern", default="Malignant", help="Substring to identify malignant cells in obs-key")

    p.add_argument("--no-balance-classes", action="store_true", help="Disable balanced class weights during training")

    args = p.parse_args()

    import anndata as ad

    adata_path = args.adata_path
    if adata_path.endswith(".zarr"):
        adata = ad.read_zarr(adata_path)
    else:
        try:
            adata = ad.read_zarr(adata_path)
        except Exception:
            adata = ad.read_h5ad(adata_path)

    os.makedirs(args.out, exist_ok=True)

    if "Manual_Genes" not in adata.var.columns:
        raise KeyError("Manual_Genes column not found in adata.var; cannot select features.")
    feature_mask = adata.var["Manual_Genes"].astype(str) == "True"
    if feature_mask.sum() == 0:
        raise ValueError("Manual_Genes column must contain at least one 'True' entry for feature selection.")

    obs_key = args.obs_key
    if obs_key not in adata.obs.columns:
        raise KeyError(f"{obs_key} not found in adata.obs")
    malignant_mask_full = adata.obs[obs_key].astype(str).str.contains(args.malignant_pattern)

    adata = adata[:, feature_mask].copy()

    adata_mal = adata[malignant_mask_full].copy()
    adata_healthy = adata[~malignant_mask_full].copy()

    print(f"Malignant cells: {adata_mal.n_obs} | Healthy cells: {adata_healthy.n_obs}")

    from sklearn.model_selection import train_test_split
    rng_seed = args.seed

    def _split_by_labels(a):
        labels_all = a.obs[obs_key].astype(str).values
        obs_names = np.asarray(a.obs_names, dtype=object)
        train_obs, eval_obs, _, _ = train_test_split(
            obs_names, labels_all,
            train_size=args.train_frac,
            stratify=labels_all if len(np.unique(labels_all)) > 1 else None,
            random_state=rng_seed,
        )
        return a[train_obs].copy(), a[eval_obs].copy()

    adata_mal_train, adata_mal_eval = _split_by_labels(adata_mal)
    adata_h_train, adata_h_eval = _split_by_labels(adata_healthy)

    def _train_one(a_train, name: str):
        print(f"Training classifier: {name}")
        model, loaders, trainer = train_classifier(
            a_train,
            layer=args.layer,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            seed=args.seed,
            obs_key=obs_key,
            balance_classes=not args.no_balance_classes,
        )
        out_sub = os.path.join(args.out, name)
        os.makedirs(out_sub, exist_ok=True)
        ckpt_path = os.path.join(out_sub, "final.ckpt")
        try:
            trainer.save_checkpoint(ckpt_path)
        except Exception:
            cpu_state = model.to("cpu").state_dict()
            torch.save(cpu_state, os.path.join(out_sub, "model_state_dict.pt"))
        return model, loaders, out_sub

    model_mal, loaders_mal, out_mal = _train_one(adata_mal_train, "malignant")
    model_h, loaders_h, out_h = _train_one(adata_h_train, "healthy")

    _assert_mapping_consistency(model_mal, loaders_mal[0].dataset.class_to_int)
    _assert_mapping_consistency(model_h,   loaders_h[0].dataset.class_to_int)
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def _evaluate(a_eval, model, class_to_int, out_dir: str, prefix: str):
        eval_loader, _ = make_dataloader_streaming(
            a_eval,
            layer=args.layer,
            indices=None,
            batch_size=args.eval_batch_size,
            shuffle=False,
            obs_key=obs_key,
            class_to_int=class_to_int,
            sampling_strategy='uniform'
        )

        pred_labels_batches: List[np.ndarray] = []
        true_labels_batches: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in eval_loader:
                # request probabilities to compute uncertainties later if wanted
                # assume model.predict_labels(x, return_probs=True) -> (labels, probs)
                labels, probs = model.predict_labels(xb, return_probs=True)
                pred_labels_batches.append(labels)
                true_labels_batches.append(model.decode(yb))
        pred_labels = np.concatenate(pred_labels_batches)
        true_labels = np.concatenate(true_labels_batches)

        cm = pd.crosstab(
            pd.Series(true_labels, name="true"),
            pd.Series(pred_labels, name="pred"),
            dropna=False,
        )
        cm = cm.reindex(index=model.classes_, columns=model.classes_, fill_value=0)
        _save_confusion_matrices(cm, out_dir, prefix=prefix)

        prec, rec, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=model.classes_, zero_division=0
        )
        acc = accuracy_score(true_labels, pred_labels)
        metrics_df = pd.DataFrame(
            {"class": model.classes_, "precision": prec, "recall": rec, "f1": f1, "support": support.astype(int)}
        )
        metrics_df.to_csv(os.path.join(out_dir, f"{prefix}_metrics_per_class.csv"), index=False)
        with open(os.path.join(out_dir, f"{prefix}_metrics_summary.txt"), "w") as f:
            f.write(
                f"Accuracy: {acc:.4f}\n"
                f"F1 (macro): {np.mean(f1):.4f}\n"
                f"F1 (weighted): {np.average(f1, weights=support):.4f}\n"
            )
        print(f"[{prefix}] Acc={acc:.4f}; macro-F1={np.mean(f1):.4f}")

    _evaluate(adata_mal_eval, model_mal, loaders_mal[0].dataset.class_to_int, out_mal, prefix="malignant")
    _evaluate(adata_h_eval, model_h, loaders_h[0].dataset.class_to_int, out_h, prefix="healthy")

    def _predict_block(a_block, model, class_to_int) -> Tuple[np.ndarray, np.ndarray]:
        loader, _ = make_dataloader_streaming(
            a_block,
            layer=args.layer,
            indices=None,
            batch_size=args.eval_batch_size,
            shuffle=False,
            obs_key=obs_key,
            class_to_int=class_to_int,
            sampling_strategy='uniform'
        )
        print(class_to_int)
        preds: List[np.ndarray] = []
        maxps: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                labels, probs = model.predict_labels(xb, return_probs=True)
                preds.append(labels)
                maxps.append(np.max(probs, axis=1))
        return np.concatenate(preds), np.concatenate(maxps)

    pred_mal, maxp_mal = _predict_block(adata_mal, model_mal, loaders_mal[0].dataset.class_to_int)
    pred_h, maxp_h = _predict_block(adata_healthy, model_h, loaders_h[0].dataset.class_to_int)

    predicted = pd.Series(index=adata.obs_names, dtype="string")
    uncertainty = pd.Series(index=adata.obs_names, dtype="float")
    predicted.loc[adata_mal.obs_names] = pred_mal.astype(str)
    predicted.loc[adata_healthy.obs_names] = pred_h.astype(str)
    uncertainty.loc[adata_mal.obs_names] = 1.0 - maxp_mal
    uncertainty.loc[adata_healthy.obs_names] = 1.0 - maxp_h

    adata.obs["predicted_cell_types"] = predicted.astype("category")
    adata.obs["uncertainty"] = uncertainty.astype(float)

    out_pred_csv = os.path.join(args.out, "predictions_all_cells.csv")
    pd.DataFrame(
        {
            "obs_names": adata.obs_names.astype(str),
            "true": adata.obs[obs_key].astype(str),
            "predicted": adata.obs["predicted_cell_types"].astype(str),
            "uncertainty": adata.obs["uncertainty"].astype(float),
        }
    ).to_csv(out_pred_csv, index=False)
    print(f"Saved predictions -> {out_pred_csv}")


    _apply_palette_to_key(adata, obs_key, PALETTE_L3)
    _apply_palette_to_key(adata, "predicted_cell_types", PALETTE_L3)

    _plot_umap(
        adata,
        color_key="uncertainty",
        out_path=os.path.join(args.out, "umap_uncertainty.png"),
        title="Uncertainty (1 - max prob)",
    )
    _plot_umap(
        adata,
        color_key=obs_key,
        out_path=os.path.join(args.out, "umap_true_Level_3.png"),
        title="True Level_3",
    )
    _plot_umap(
        adata,
        color_key="predicted_cell_types",
        out_path=os.path.join(args.out, "umap_predicted_cell_types.png"),
        title="Predicted cell types",
    )
    y_true_all = adata.obs[obs_key].astype(str).values
    y_pred_all = adata.obs["predicted_cell_types"].astype(str).values

    overall_metrics = {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "precision": precision_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "recall": recall_score(y_true_all, y_pred_all, average="macro", zero_division=0),
        "f1": f1_score(y_true_all, y_pred_all, average="macro", zero_division=0),
    }

    overall_csv = os.path.join(args.out, "overall_metrics_full_predictions.csv")
    pd.DataFrame([overall_metrics]).to_csv(overall_csv, index=False)
    print(f"Saved overall metrics CSV -> {overall_csv}")

    _save_overall_metrics_bar(overall_metrics, args.out, filename="overall_metrics_bar.png")
    
    print(f"Saved model artifacts and figures to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
