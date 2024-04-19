import os
import sys
sys.path.append('/home/aih/shrey.parikh/PDAC/PDAC/code/scripts')
from train_scVI import train_scvi_model
from train_scPoli import train_scpoli_model
from train_sysVI import train_sysvi_model
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/')
import scanpy as sc
import numpy as np

print('Loading adata')
adata = sc.read_h5ad('concatenated_datasets/PDAC_concat_hvg_batch_key_datasets_hvg.h5ad')
adata.obs['Condition'] = np.where(adata.obs.Dataset == 'Regev', 'snRNA', 'scRNA')
adata.obs = adata.obs[['n_genes', 'n_counts', 'log_counts', 'mt_frac', 'Dataset', 'ID', 'Condition',
                    'Level 1 Annotation', 'Level 2 Annotation', 'Level 3 Annotation', 'Cell_type']]
adata.layers['log-norm'] = adata.X
adata.X = adata.layers['raw']
# del adata.uns
# del adata.obsp
# del adata.obsm

# print('Calculating neighbors')
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)
# print('saving adata')
# adata.write('concatenated_datasets/PDAC_concat_hvg_batch_key_datasets_hvg.h5ad')

try:
    print('Running scVI function')
    train_scvi_model(adata, model_save_path='scVI/scVI_model_dataset', latent_save_path='scVI/scVI_dataset_reference_latent.h5ad')
except Exception as e:
    print(f'scVI function failed because {e}') 
    
try:
    print('Running scPoli function')
    train_scpoli_model(adata, model_save_path='scPoli/scPoli_model_dataset', latent_save_path='scPoli/scPoli_dataset_reference_latent.h5ad')
except Exception as e:
    print(f'scPoli function failed because {e}') 
    
try:
    print('Running sysVI function')
    train_sysvi_model(adata, model_save_path='sysVI/sysVI_model_dataset', embed_save_path='sysVI/sysVI_dataset_reference_latent.h5ad')
    
except Exception as e:
    print(f'sysVI function failed because {e}') 