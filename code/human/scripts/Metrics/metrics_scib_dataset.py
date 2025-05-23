import os
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import zarr
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

os.chdir('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/adata_scpoli_mg_binned_all_embeddings.h5ad')
print(f'OBSM has: {adata.obsm.keys}')
# subset = sc.pp.subsample(adata, fraction=0.001, copy=True)
embeddings = [i for i in adata.obsm if 'emb' in i]
embeddings.append('X_pca')

bm = Benchmarker(
    adata,
    batch_key="Dataset",
    label_key="Level_1_refined",
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=embeddings,
    n_jobs=-1,
)
bm.benchmark()
df_metrics = bm.get_results(min_max_scale=True, clean_names=True)
df_metrics.to_csv('metrics/df_metrics_gpu.csv')
bm.plot_results_table(min_max_scale=True, show=False, save_dir='metrics')
