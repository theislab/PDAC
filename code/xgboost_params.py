import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
sc.set_figure_params(dpi=100, dpi_save=300, frameon=False, )
import shutup; shutup.please()

def evaluate_parameter(X, Y, param_name, param_values, base_params, num_boost_round=100, save_fig=False, fig_name=""):
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    mean_accuracy_scores = []
    mean_roc_auc_scores = []
    mean_f1_scores = []

    for value in param_values:
        accuracy_scores = []
        roc_auc_scores = []
        f1_scores = []
        for train_index, test_index in skf.split(X, Y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = Y[train_index], Y[test_index]

            dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dtest_fold = xgb.DMatrix(X_test_fold, label=y_test_fold)

            params = base_params.copy()
            params[param_name] = value
            bst_fold = xgb.train(params, dtrain_fold, num_boost_round)

            y_pred_fold = bst_fold.predict(dtest_fold) > 0.5
            accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
            roc_auc_scores.append(roc_auc_score(y_test_fold, bst_fold.predict(dtest_fold)))
            f1_scores.append(f1_score(y_test_fold, y_pred_fold))

        mean_accuracy_scores.append(np.mean(accuracy_scores))
        mean_roc_auc_scores.append(np.mean(roc_auc_scores))
        mean_f1_scores.append(np.mean(f1_scores))

    # Plotting
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(param_values, mean_accuracy_scores, marker='o')
    plt.title(f'Accuracy vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(param_values, mean_roc_auc_scores, marker='o')
    plt.title(f'ROC AUC vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('ROC AUC')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(param_values, mean_f1_scores, marker='o')
    plt.title(f'F1 Score vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('F1 Score')
    plt.grid()

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{fig_name}.png")
    plt.show()


os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/Classifier/')
datasets = ['Regev', 'Moncada', 'Schlesinger', 'Zenodo_OUGS', 'Ding', 'Peng']
adata = sc.read_h5ad(('/home/aih/shrey.parikh/PDAC/PDAC/Classifier/scpoli_donor_final_integration.h5ad'))
adata_labeled = adata[adata.obs.Dataset.isin(datasets)]
adata_unlabeled = adata[~adata.obs.Dataset.isin(datasets)]

X_emb = adata_labeled.obsm['X_emb']
X = adata_labeled.X
Y = adata_labeled.obs['Malignant_Classification']
Y = Y.map({'Malignant': 1, 'Non-Malignant': 0})
base_params = {
    'eta': 0.1,  # Default learning rate
    'max_depth': 20,  # Default max depth
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

max_depth_values = range(1, 51, 5)  
eta_values = np.arange(0.01, 0.22, 0.02)
evaluate_parameter(X_emb, Y, 'max_depth', max_depth_values, base_params, save_fig=True, fig_name="X_emb_max_depth")
evaluate_parameter(X_emb, Y, 'eta', eta_values, base_params, save_fig=True, fig_name="X_emb_eta")
evaluate_parameter(X, Y, 'max_depth', max_depth_values, base_params, save_fig=True, fig_name="X_max_depth")
evaluate_parameter(X, Y, 'eta', eta_values, base_params, save_fig=True, fig_name="X_eta")
