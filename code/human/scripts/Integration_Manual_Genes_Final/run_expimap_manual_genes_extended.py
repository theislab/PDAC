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
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final/Notebooks')
print("Loading gene lists...")

#load the gene list
mofa_genes_df = pd.read_csv('../Expimap/expimap_10_factors_selected_norepeatinggenes/var_names.csv')
broad_markers = pd.read_csv('broad_markers.csv', index_col='Unnamed: 0')
broad_markers_list = list(set(broad_markers.values.flatten().astype(str)[broad_markers.values.flatten().astype(str) != 'nan']))
de_genes_df = pd.read_pickle('de_genes_to_be_added.csv')
mofa_genes = mofa_genes_df.values.flatten().tolist()
de_genes = de_genes_df.values.flatten().tolist()
xenium_df = pd.read_csv('pdac_xenium_panel.csv')
xenium_genes = list(set(xenium_df.Gene.tolist()))

print(f"Length of MOFA genes: {len(mofa_genes)}")
print(f"Length of broad marker genes: {len(broad_markers_list)}")
print(f"Length of DE genes: {len(de_genes)}")
print(f"Length of xenium panel genes: {len(xenium_genes)}")

 
all_genes = list(set(mofa_genes + broad_markers_list + de_genes + xenium_genes))
print(f"Total unique genes combined: {len(all_genes)}")


print("Loading AnnData objects...")

#load the anndata objects
adata_sc = sc.read_h5ad('../single_cell_int/adata_sc_int_cnv.h5ad')
adata_sn = sc.read_h5ad('../single_nuc_int/adata_nuc_int_outlier_genes.h5ad')
valid_genes_sc = [gene for gene in all_genes if gene in adata_sc.var_names]
valid_genes_sn = [gene for gene in all_genes if gene in adata_sn.var_names]
print(f"Total valied genes combined: {len(valid_genes_sc)}")
print(f"Total valied genes combined: {len(valid_genes_sn)}")

print("Subsetting genes in AnnData objects...")
adata_sc = adata_sc[:, valid_genes_sc]
adata_sn = adata_sn[:, valid_genes_sc]

print("Resetting and renaming indices...")
adata_sc.obs.reset_index(inplace=True)
adata_sn.obs.reset_index(inplace=True)
adata_sc.obs.rename(columns={'index':'Barcode'},inplace=True)
adata_sn.obs.rename(columns={'index':'Barcode'},inplace=True)
adata_sc.obs_names = adata_sc.obs.Dataset_Barcode
adata_sn.obs_names = adata_sn.obs.Dataset_Barcode
adata_sc.obs_names= adata_sc.obs_names.astype(str)
adata_sn.obs_names= adata_sn.obs_names.astype(str)
adata_sc.obs_names_make_unique()
adata_sn.obs_names_make_unique()
adata_sn.obs['Level_1'] = adata_sn.obs.scpoli_labels.copy()
adata = adata_sc.concatenate(adata_sn)
print(f"Shape of concatenated AnnData object: {adata.shape}")
# adata = sc.read_h5ad('../Expimap/adata_combined_manual_genes.h5ad')
adata.obs.ID = adata.obs.ID.astype(str)
adata.obs.batch_covariate = adata.obs.batch_covariate.astype(str)
adata.obs['ID_batch_covariate'] = adata.obs.ID + '_' + adata.obs.batch_covariate
adata.obs.ID_batch_covariate = adata.obs.ID_batch_covariate.astype('category')
adata.obs.ID = adata.obs.ID.astype('category')
adata.obs = adata.obs.rename(columns={'Dataset_Barcode': 'Dataset_Barcode_Column'})


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
adata.obsm['X_cvae'] = intr_cvae.get_latent(mean=False, only_active=True)

print("Performing clustering and UMAP embedding...")
sc.pp.neighbors(adata, use_rep='X_cvae')
sc.tl.leiden(adata)
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)
# sc.pl.umap(adata, color=['Level_1', 'Dataset'], frameon=False, save='umap.png')
print('Saving...')
adata.write_zarr('/home/aih/shrey.parikh/PDAC/PDAC_Final/Expimap/int_manually_selected_genes_extended.zarr')
intr_cvae.save('/home/aih/shrey.parikh/PDAC/PDAC_Final/Expimap/Expimap/expimap_manually_selected_genes_extended', overwrite=True)
