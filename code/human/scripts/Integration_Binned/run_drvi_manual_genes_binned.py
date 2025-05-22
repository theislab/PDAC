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
os.chdir('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/adata_scpoli_mg_binned.h5ad')

#Assign binned data to X
adata.X = adata.layers['binned_data'].copy()
counts_layer = "binned_data"
batch_key = 'ID_batch_covariate'
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
model.save("drvi/drvi_mg_binned", overwrite=True)

adata.obsm['X_drvi'] = model.get_latent_representation()
print("Performing clustering and UMAP embedding...")
sc.pp.neighbors(adata, use_rep='X_drvi')
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)


#Add metrics requirements
adata.uns['output_type'] = 'embed'
adata.obsm['X_emb'] = adata.obsm['X_drvi']

#save
adata.write_zarr('drvi/drvi_mg_binned.zarr')
