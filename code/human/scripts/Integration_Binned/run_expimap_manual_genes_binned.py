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
os.chdir('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/adata_scpoli_mg_binned.h5ad')

#Assign binned data to X
adata.X = adata.layers['binned_data'].copy()

print("Setting up mask and raw layer...")
mask = np.ones((adata.shape[1], 20)) 
adata.varm['mask'] = mask
adata.X = adata.layers['raw'].copy()
adata.X = adata.X.astype('float32')
adata.uns['terms'] = ['Gene_set_' + str(i+1) for i in range(0,20)]


print("Initializing EXPIMAP model...")
intr_cvae = sca.models.EXPIMAP(
    adata=adata,
    condition_key='ID_batch_covariate',
    hidden_layer_sizes=[300, 300, 300],
    recon_loss='nb',
    mask_key='mask'
)

print("Training model...")

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
print("Training completed. Extracting latent representation...")
adata.obsm['X_expimap'] = intr_cvae.get_latent(mean=False, only_active=True)

print("Performing clustering and UMAP embedding...")
sc.pp.neighbors(adata, use_rep='X_expimap')
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)


#Add metrics requirements
adata.uns['output_type'] = 'embed'
adata.obsm['X_emb'] = adata.obsm['X_expimap']


print('Saving...')
adata.write_zarr('expimap/expimap_mg_binned.zarr')
intr_cvae.save('expimap/expimap_mg_binned', overwrite=True)
