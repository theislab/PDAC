import warnings
warnings.simplefilter(action='ignore')
import scanpy as sc
import torch
import scarches as sca
import numpy as np
import gdown
import pandas as pd
from collections import defaultdict,Counter
import gc
import matplotlib.pyplot as plt
import pickle
import os
sc.set_figure_params(frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final')
adata = sc.read_h5ad('MOFA/adata_combined_even_more_hvg.h5ad')
# binary_matrix = pd.read_csv('MOFA/MOFA_15_Factors_6082HVG/binary_matrix.csv', index_col='Unnamed: 0')
binary_matrix = pd.read_csv('MOFA/MOFA_15_Factors_6082HVG/binary_matrix_15_factors.csv', index_col='Unnamed: 0')
all_genes = binary_matrix.index.tolist()
adata_gp = adata[:, all_genes].copy()
adata_gp.varm['I'] = np.array(binary_matrix)
select_terms = adata_gp.varm['I'].sum(0)>12
adata_gp.uns['terms'] = binary_matrix.columns
adata_gp._inplace_subset_var(adata_gp.varm['I'].sum(1)>0)
adata_gp.uns['terms']
adata_gp.obs.ID = adata_gp.obs.ID.astype(str)
adata_gp.obs.batch_covariate = adata_gp.obs.batch_covariate.astype(str)
adata_gp.obs['ID_batch_covariate'] = adata_gp.obs.ID + '_' + adata_gp.obs.batch_covariate
adata_gp.obs.ID_batch_covariate = adata_gp.obs.ID_batch_covariate.astype('category')
adata_gp.obs.ID = adata_gp.obs.ID.astype('category')

intr_cvae = sca.models.EXPIMAP(
    adata=adata_gp,
    condition_key='ID_batch_covariate',
    hidden_layer_sizes=[300, 300, 300],
    recon_loss='nb'
)

adata_gp.X = adata_gp.X.astype('float32')

ALPHA = 0.7
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
intr_cvae.train(
    n_epochs=200,
    alpha_epoch_anneal=50,
    alpha=ALPHA,
    alpha_kl=0.5,
    weight_decay=0.,
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    seed=2020
)

adata_gp.obsm['X_cvae'] = intr_cvae.get_latent(mean=False, only_active=True)
adata_gp.uns['terms'] = adata_gp.uns['terms'].tolist()
sc.pp.neighbors(adata_gp, use_rep='X_cvae')
sc.tl.umap(adata_gp)
adata_gp.write('Expimap/int_15_factors_selected_norepeatinggenes.h5ad')
intr_cvae.save('Expimap/expimap_15_factors_selected_norepeatinggenes')
