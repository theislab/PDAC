# TODO: uncomment for the final version
import warnings
warnings.filterwarnings('ignore')
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt
from IPython.display import display
# from gprofiler import GProfiler
import numpy as np
import drvi
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch
import os
import pandas as pd
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final/Notebooks')
print("Loading gene lists...")

counts_layer = 'raw'
batch_key = 'ID_batch_covariate'
plot_keys = ['Dataset', 'Level_1', 'Condition', 'batch_covariate']

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

subset = sc.pp.subsample(adata, fraction=0.1, copy=True)
raw_counts = subset.X.toarray()
print(f"Are raw counts in X integers? {np.all(raw_counts.astype(int) == raw_counts)}")
print(f"Mean raw counts: {np.mean(raw_counts)}")
print(f"Range of raw counts: {np.min(raw_counts)} to {np.max(raw_counts)}")
print(f"Percentage of zero counts: {np.mean(raw_counts == 0) * 100:.2f}%")
print("-" * 50)
raw_counts = subset.layers[counts_layer].toarray()
print(f"Are raw counts in {counts_layer} integers? {np.all(raw_counts.astype(int) == raw_counts)}")
print(f"Mean raw counts: {np.mean(raw_counts)}")
print(f"Range of raw counts: {np.min(raw_counts)} to {np.max(raw_counts)}")
print(f"Percentage of zero counts: {np.mean(raw_counts == 0) * 100:.2f}%")
print("-" * 50)


DRVI.setup_anndata(
    adata,
    layer=counts_layer,
    categorical_covariate_keys=[batch_key],
    # DRVI accepts count data by default.
    # Set to false if you provide log-normalized data and use normal distribution (mse loss).
    is_count_data=True,
)

# construct the model
model = DRVI(
    adata,
    # Provide categorical covariates keys once again. Refer to advanced usages for more options.
    categorical_covariates=[batch_key],
    n_latent=64,
    # For encoder and decoder dims, provide a list of integers.
    encoder_dims=[64, 64],
    decoder_dims=[64, 64],
)
# train the model
model.train(
    max_epochs=100,
    early_stopping=False,
    early_stopping_patience=20,
)

model.save("../drvi/drvi_v3_64_manual_genes_extended", overwrite=True)