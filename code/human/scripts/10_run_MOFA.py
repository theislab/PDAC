import pandas as pd
import numpy as np
import scanpy as sc
from mofapy2.run.entry_point import mofa
import os
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final')
# load sc data from infercnv
# load sn data from outlier nb because inferCNV failed for some reason, sn anyway has malignant labels

# adata_sc = sc.read_h5ad('single_cell_int/adata_sc_int_cnv.h5ad')
# adata_sn = sc.read_h5ad('single_nuc_int/adata_nuc_int_outlier_genes.h5ad')
# print(adata_sc.shape)
# print(adata_sn.shape)
# def remove_genes(adata):
#     gene_names = adata.var_names  # No need to convert to list, use directly
#     mt_genes = gene_names.str.startswith('MT')  # Boolean mask for MT genes
#     rp_genes = gene_names.str.startswith('RP')  # Boolean mask for RP genes
    
#     # Combine masks
#     mt_rp_genes = mt_genes | rp_genes  # Genes that are either MT or RP
    
#     # Print how many genes are being filtered out
#     print(f"Number of MT and RP genes removed: {mt_rp_genes.sum()}")
    
#     # Filter the adata object and return the modified version
#     adata = adata[:, ~mt_rp_genes].copy()
#     return adata

# Apply the function to both adata_sc and adata_sn and reassign them
# adata_sc = remove_genes(adata_sc)
# adata_sn = remove_genes(adata_sn)
# sc.pp.highly_variable_genes(adata_sc, layer='log_norm', batch_key='Dataset', n_top_genes=12000)
# sc.pp.highly_variable_genes(adata_sn, layer='log_norm', batch_key='Dataset', n_top_genes=12000)
# adata_sc_hvg = adata_sc[:, adata_sc.var.highly_variable].copy()
# adata_sn_hvg = adata_sn[:, adata_sn.var.highly_variable].copy()
# adata_sc_hvg.obs.reset_index(inplace=True)
# adata_sn_hvg.obs.reset_index(inplace=True)
# adata_sc_hvg.obs.rename(columns={'index':'Barcode'},inplace=True)
# adata_sn_hvg.obs.rename(columns={'index':'Barcode'},inplace=True)
# adata_sc_hvg.obs_names = adata_sc.obs.Dataset_Barcode
# adata_sn_hvg.obs_names = adata_sn_hvg.obs.Dataset_Barcode
# adata_sc_hvg.obs_names= adata_sc_hvg.obs_names.astype(str)
# adata_sn_hvg.obs_names= adata_sn_hvg.obs_names.astype(str)
# adata_sc_hvg.obs_names_make_unique()
# adata_sn_hvg.obs_names_make_unique()
# adata_sn_hvg.obs['Level_1'] = adata_sn_hvg.obs.scpoli_labels.copy()
# adata_combined = adata_sc_hvg.concatenate(adata_sn_hvg)
# adata_combined.obs.rename(columns={'Dataset_Barcode': 'Dataset_Barcode_repeated'}, inplace=True)
# adata_combined.obs = adata_combined.obs.loc[:, ~adata_combined.obs.columns.duplicated()].copy()
# adata_combined.obs = adata_combined.obs.astype(str)
# adata_combined.write('MOFA/adata_combined_even_more_hvg.h5ad')
# adata_combined = sc.read_h5ad('MOFA/adata_combined_even_more_hvg.h5ad')
# print(adata_combined.shape)
#Run MOFA separately
# adata_combined = sc.read_h5ad('/home/aih/shrey.parikh/PDAC/PDAC_Final/MOFA/adata_combined.h5ad')


# print('Checking X for SC')
# subset = sc.pp.subsample(adata_sc, fraction=0.1, copy=True)
# raw_counts = subset.X.toarray()
# print(f"Are raw counts integers? {np.all(raw_counts.astype(int) == raw_counts)}")
# print(f"Range of raw counts: {np.min(raw_counts)} to {np.max(raw_counts)}")
# print("-" * 50)

# print('Checking X for SN')
# subset = sc.pp.subsample(adata_sn, fraction=0.1, copy=True)
# raw_counts = subset.X.toarray()
# print(f"Are raw counts integers? {np.all(raw_counts.astype(int) == raw_counts)}")
# print(f"Range of raw counts: {np.min(raw_counts)} to {np.max(raw_counts)}")
# print("-" * 50)

adata_combined = sc.read_h5ad('MOFA/adata_combined_even_more_hvg.h5ad')
print(adata_combined.shape)
cells_to_remove = ['Ambiguous_Immune', 'Ambiguous_Stromal', 'Ambiguous_Epithelial']
adata_filtered = adata_combined[~((adata_combined.obs.outlier == '1') | (adata_combined.obs.Level_1.isin(cells_to_remove)))].copy()
m_sc = mofa(adata_filtered, 
         expectations=["W","Z","AlphaW","AlphaZ"],
         use_raw=False,
         n_factors=10,
         groups_label="batch_covariate", 
         outfile="/home/aih/shrey.parikh/PDAC/PDAC_Final/MOFA/MOFA_results_even_more_hvg_more_factors_filtered_job_gpu.hdf5", quiet=False, use_layer='log_norm', gpu_mode=True, use_float32=True)
# m_sn = mofa(adata_sn_hvg, 
#          expectations=["W","Z","AlphaW","AlphaZ"],
#          use_raw=False,
#          n_factors=10,
#          groups_label="Dataset", 
#          outfile="/home/aih/shrey.parikh/PDAC/PDAC_Final/MOFA/adata_sn_5000.hdf5", quiet=False, use_layer='log_norm')