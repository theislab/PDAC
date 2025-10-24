"""CLI script to train MLP malignant classifier from an AnnData file.

Usage: python -m classifiers.train_mlp /path/to/adata.zarr --layer log_norm --out model_dir
"""
import argparse
import os
import torch

from training import train_classifier
from dataset import make_dataloader_streaming
from typing import Optional

def main():
    p = argparse.ArgumentParser(description="Train MLP malignant classifier on an AnnData file.")
    p.add_argument("adata_path", help="Path to AnnData (zarr, h5ad, etc.)")
    p.add_argument("--layer", default=None, help="Layer name to use (default: adata.X if None)")
    p.add_argument("--out", default="models", help="Directory to save model checkpoints and state dict")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    adata_path = args.adata_path
    import anndata as ad
    import numpy as np

    if adata_path.endswith(".zarr"):
        adata = ad.read_zarr(adata_path)
    else:
        try:
            adata = ad.read_zarr(adata_path)
        except Exception:
            adata = ad.read_h5ad(adata_path)
    adata.obs['Malignant'] = ['Malignant' if 'Malignant' in cell else 'TME' for cell in adata.obs.Level_4]

    os.makedirs(args.out, exist_ok=True)

    seed = 42
    train_frac = 0.8

    n = len(adata.obs_names)
    rs = np.random.RandomState(seed)

    perm = rs.permutation(n)
    n_train = int(n * train_frac)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    obs = np.asarray(adata.obs_names)
    train_obs = obs[train_idx].tolist()
    val_obs   = obs[val_idx].tolist()

    adata_train = adata[train_obs, adata.var.Manual_Genes == 'True'].copy()
    adata_test  = adata[val_obs,   adata.var.Manual_Genes == 'True'].copy()

    model, loaders, trainer = train_classifier(
        adata_train,
        layer=args.layer,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    dl, _ = make_dataloader_streaming(
        adata_test, layer=args.layer, indices=None, batch_size=32, shuffle=False, class_to_int={"TME": 0, "Malignant": 1},  # <- add this

    )
    model.eval()

    pred_labels_batches = []
    pred_probs_batches  = []
    with torch.no_grad():
        for xb, yb in dl:
            labels, probs = model.predict_labels(xb, return_probs=True)
            pred_labels_batches.append(labels)
            pred_probs_batches.append(probs)

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
    )

    pred_labels = np.concatenate(pred_labels_batches)   
    pred_probs  = np.concatenate(pred_probs_batches) 
    adata_test.obs['Malignant_predicted'] = pred_labels

    def plot_confusion_matrix_from_obs(
        adata_obs: pd.DataFrame,
        out_dir: str,
        true_col: str = "Malignant",
        pred_col: str = "Malignant_predicted",
        labels = ("TME", "Malignant"),
        normalize: bool = False,
        title: Optional[str] = None,
        figsize=(5, 4),
        filename: Optional[str] = None,
    ):
        """
        Plots and saves a confusion matrix (rows=True, cols=Pred).
        Set normalize=True for row-normalized percentages.
        """
        df = adata_obs[[true_col, pred_col]].dropna(subset=[pred_col]).astype(str)
        cm = pd.crosstab(df[true_col], df[pred_col], dropna=False)\
            .reindex(index=labels, columns=labels, fill_value=0)

        if normalize:
            cm_plot = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
            fmt = ".2f"
            vmin, vmax = 0.0, 1.0
            default_title = "Confusion Matrix (row-normalized)"
        else:
            cm_plot = cm.astype(float)
            fmt = ".0f"
            vmin, vmax = 0.0, None
            default_title = "Confusion Matrix (counts)"

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_plot.values, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title or default_title)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm_plot.values[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        if filename is None:
            suffix = "_norm" if normalize else "_counts"
            filename = f"confusion_matrix{suffix}.png"

        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved confusion matrix to: {save_path}")

        return cm 

    labels_tuple = ("TME", "Malignant")  
    cm_counts = plot_confusion_matrix_from_obs(adata_test.obs, args.out, labels=labels_tuple, normalize=False)
    cm_norm   = plot_confusion_matrix_from_obs(adata_test.obs, args.out, labels=labels_tuple, normalize=True)

    y_true = adata_test.obs["Malignant"].astype(str).values
    y_pred = adata_test.obs["Malignant_predicted"].astype(str).values

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels_tuple), zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    metrics_df = pd.DataFrame({
        "class": list(labels_tuple),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support.astype(int),
    })

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    def plot_metrics_per_class(df: pd.DataFrame, out_dir: str, filename: str = "metrics_per_class.png"):
        x = np.arange(len(df["class"]))
        width = 0.25

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width, df["precision"].values, width, label="Precision")
        ax.bar(x,         df["recall"].values,    width, label="Recall")
        ax.bar(x + width, df["f1"].values,        width, label="F1")

        ax.set_xticks(x)
        ax.set_xticklabels(df["class"].values)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Per-class Metrics")
        ax.legend()
        for i, v in enumerate(df["support"].values):
            ax.text(x[i], 1.03, f"n={v}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved per-class metrics barplot to: {save_path}")

    plot_metrics_per_class(metrics_df, args.out)

    def plot_metrics_summary(out_dir: str, filename: str = "metrics_summary.png"):
        summary_names = ["Accuracy", "F1 (macro)", "F1 (weighted)"]
        summary_vals  = [acc, f1_macro, f1_weighted]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(np.arange(len(summary_vals)), summary_vals)
        ax.set_xticks(np.arange(len(summary_vals)))
        ax.set_xticklabels(summary_names, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Overall Metrics")
        for i, v in enumerate(summary_vals):
            ax.text(i, v + 0.02 if v <= 0.95 else 0.98, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved summary metrics barplot to: {save_path}")

    plot_metrics_summary(args.out)

    ckpt_path = os.path.join(args.out, "final.ckpt")
    try:
        trainer.save_checkpoint(ckpt_path)
    except Exception:
        cpu_model = model.to("cpu")
        torch.save(cpu_model.state_dict(), os.path.join(args.out, "model_state_dict.pt"))

    print(f"Saved model artifacts to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
