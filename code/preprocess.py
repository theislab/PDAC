import os
import scanpy as sc
# import seaborn as sb
import numpy as np
# import matplotlib.pyplot as plt
import warnings
# import anndata as ad
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# import gc
# import psutil

os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/concatenated_datasets/')

adata_donor= sc.read_h5ad('PDAC_concat_hvg_batch_key_samples_hvg.h5ad')

adata_dataset = sc.read_h5ad('PDAC_concat_hvg_batch_key_datasets_hvg.h5ad')

adata_dataset_8000 = sc.read_h5ad('PDAC_concat_hvg_batch_key_datasets_hvg_8000.h5ad')

adata_list = [adata_donor, adata_dataset, adata_dataset_8000]
names = ['PDAC_concat_hvg_batch_key_donors_hvg_final.h5ad', 'PDAC_concat_hvg_batch_key_datasets_hvg_final.h5ad', 'PDAC_concat_hvg_batch_key_datasets_hvg_8000_final.h5ad']
for adata, save_name in zip(adata_list, names):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    save_name_umap = save_name.replace('.h5ad', '')  
    sc.pl.umap(adata, color='Dataset', save=f'{save_name_umap}.png')
    adata.write(save_name)

