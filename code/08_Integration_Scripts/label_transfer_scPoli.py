import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from scarches.models.scpoli import scPoli
import traceback
import warnings
import os
import sys
import gc
sys.path.append('/home/aih/shrey.parikh/PDAC/PDAC/code/')
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/')
warnings.filterwarnings('ignore')

from atlas_pipeline.integrate import train_scpoli_model_label_transfer
# Load batch_key = dataset
# print('Loading batch_key = dataset')
# adata = sc.read_h5ad('/home/aih/shrey.parikh/PDAC/PDAC/concatenated_datasets/PDAC_concat_hvg_batch_key_datasets_hvg.h5ad')
# try:
#     train_scpoli_model_label_transfer(adata, training_datasets=['Regev', 'Peng', 'Moncada', 'Schlesinger', 'Zenodo_OUGS'], cell_type_key='Label_Harmonized', 
#     condition_key='Dataset', model_save_path='scPoli/scPoli_model_dataset_label_transfer', 
#     latent_save_path='scPoli/scPoli_dataset_reference_latent_label_transfer.h5ad')
# except Exception as e:
#     print(f'scpoli batch_key=dataset training failed becasuse {e}')
#     traceback.print_exc()
# del adata
# gc.collect()

print('Loading batch_key = donor')
adata_donor = sc.read_h5ad('/home/aih/shrey.parikh/PDAC/PDAC/concatenated_datasets/PDAC_concat_hvg_batch_key_samples_hvg.h5ad')
try:
    train_scpoli_model_label_transfer(adata_donor, training_datasets=['Regev', 'Peng', 'Moncada', 'Schlesinger', 'Zenodo_OUGS'], cell_type_key='Label_Harmonized', 
    condition_key='Dataset', model_save_path='scPoli/scPoli_model_donor_label_transfer', 
    latent_save_path='scPoli/scPoli_donor_reference_latent_label_transfer.h5ad')
except Exception as e:
    print(f'scpoli batch_key=donor training failed becasuse {e}')
    traceback.print_exc()
del adata_donor
gc.collect()

print('Loading batch_key = dataset_8000')   
adata_8000 = sc.read_h5ad('/home/aih/shrey.parikh/PDAC/PDAC/concatenated_datasets/PDAC_concat_hvg_batch_key_datasets_hvg_8000.h5ad')
try:
    train_scpoli_model_label_transfer(adata_8000, training_datasets=['Regev', 'Peng', 'Moncada', 'Schlesinger', 'Zenodo_OUGS'], cell_type_key='Label_Harmonized', 
    condition_key='Dataset', model_save_path='scPoli/scPoli_model_dataset_8000_label_transfer', 
    latent_save_path='scPoli/scPoli_dataset_reference_latent_8000_label_transfer.h5ad')
except Exception as e:
    print(f'scpoli batch_key=dataset_8000 training failed becasuse {e}')
    traceback.print_exc()
del adata_8000
gc.collect()
