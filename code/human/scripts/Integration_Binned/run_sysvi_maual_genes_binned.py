import os
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import torch
import pytorch_lightning as pl
from typing import Literal
# sys.path.insert(0, '/home/aih/shrey.parikh/cross_species/multiple_system_integration/cross_system_integration/')
# sys.path.append('/home/aih/shrey.parikh/cross_species/multiple_system_integration/cross_system_integration/cross_system_integration/')
# from cross_system_integration.model._xxjointmodel import XXJointModel
from scvi.external import SysVI

os.chdir('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/adata_scpoli_mg_binned.h5ad')

#Assign binned data to X
adata.X = adata.layers['binned_data'].copy()

SysVI.setup_anndata(
    adata=adata,
    batch_key="ID_batch_covariate",
    categorical_covariate_keys=["Technology"],
)

model = SysVI(
    adata=adata,
    embed_categorical_covariates=True,
)
max_epochs = 200
model.train(
    max_epochs=max_epochs, check_val_every_n_epoch=1, plan_kwargs={"z_distance_cycle_weight": 5}
)

print("Training completed. Extracting latent representation...")
adata.obsm['X_sysvi'] = model.get_latent_representation(adata=adata)

print("Performing clustering and UMAP embedding...")
sc.pp.neighbors(adata, use_rep='X_sysvi')
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)


#Add metrics requirements
adata.uns['output_type'] = 'embed'
adata.obsm['X_emb'] = adata.obsm['X_sysvi']


print('Saving...')
adata.write_zarr('sysvi/sysvi_mg_binned.zarr')
model.save('sysvi/sysvi_mg_binned', overwrite=True)
print("sysVI training completed.")