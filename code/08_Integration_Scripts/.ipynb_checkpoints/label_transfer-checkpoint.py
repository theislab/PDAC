import os
import sys
sys.path.append('/home/aih/shrey.parikh/PDAC/PDAC/code/scripts')
from train_scANVI import train_scanvi_model
from train_scANVI import train_scanvi_model_separated
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/')
import scanpy as sc
import numpy as np


adata = sc.read_h5ad('concatenated_datasets/regev_peng_zenodo_label_harmonized.h5ad')
adata_unlabeled = sc.read_h5ad('concatenated_datasets/PDAC_concat_hvg_batch_key_datasets_hvg.h5ad')
adata_unlabeled = adata_unlabeled[adata_unlabeled.obs.Dataset.isin(['Ding', 'Lee', 'Simeone', 'Steele', 'Caronni', 'Zhang'])]
try:
    print('Running scANVI function')
    train_scanvi_model(adata, adata_unlabeled, 
                       ref_path='scanvae/increased_epochs/scanvae_model_dataset', 
                       surgery_path='scanvae/increased_epochs/surgery_model_dataset', 
                       full_latent_path='scanvae/increased_epochs/scanvae_dataset_full_latent.h5ad')
except Exception as e:
    print(f'scVI function failed because {e}') 

adata_unlabeled = adata_unlabeled[adata_unlabeled.obs.Dataset.isin(['Ding', 'Lee', 'Simeone', 'Steele', 'Caronni', 'Zhang'])]

try:
    print('Running scANVI function')
    train_scanvi_model_separated(adata, adata_unlabeled, 
                       ref_path='scanvae/increased_epochs/separate/scanvae_model_dataset', 
                       surgery_path='scanvae/increased_epochs/separate/surgery_model_dataset', 
                       full_latent_path='scanvae/increased_epochs/separate/scanvae_dataset_full_latent.h5ad')
except Exception as e:
    print(f'scVI function failed because {e}') 