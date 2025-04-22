import scanpy as sc
import scarches as sca

import pandas as pd
import numpy as np
import warnings
import torch
import anndata as ad

import drvi
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

sc.settings.set_figure_params(dpi=200, frameon=False)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


############################
# Prepare input data
############################

print("Loading AnnData object...")
adata_final = sc.read_h5ad('/mnt/storage/Daniele/atlases/mouse/11_mouse_all_integration_input.h5ad')
adata = adata_final[:, adata_final.var.Manual_genes].copy()

#scvi hotfix
adata.obs['Dataset'] = adata_final.obs['Dataset'].astype(str)
adata.obs['Dataset'] = adata.obs['Dataset'].astype('category')
adata.obs['Level_1_refined'] = adata_final.obs['Level_1_refined'].astype(str)
adata.obs['Level_1_refined'] = adata.obs['Level_1_refined'].astype('category')


print(adata)

#############################
## Variables
#############################

batch_key='Dataset'
labels_key='Level_1_refined'
n_latent=10
n_layers=2
model_save_path='/home/daniele/Code/github_synced/PDAC/models/'
layer = 'counts'

############################
# scVI and scANVI
############################

print('SCVI and scANVI model training...')
print('Setting up scVI...')
print('Setting up AnnData for training...')
sca.models.SCVI.setup_anndata(
    adata, 
    layer=layer,
    batch_key=batch_key, 
    labels_key=labels_key, 
)
print('Training scVI model...')
vae = sca.models.SCVI(
    adata,
    n_layers=n_layers,
    n_latent=n_latent,
    encode_covariates=True,
    deeply_inject_covariates=False,
    use_layer_norm="both",
    use_batch_norm="none",
)
vae.train()
print('Saving trained model...')
vae.save(f'{model_save_path}/scVI', overwrite=True)
print('Embedding latent representation...')
adata.obsm['X_scvi'] = vae.get_latent_representation()
adata_final.obsm['scVI_emb'] = adata.obsm['X_scvi']
scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
scanvae.train()
scanvae.save(f'{model_save_path}/scANVI', overwrite=True)
print("Adding latent representation to AnnData...")
adata.obsm['X_scanvi'] = scanvae.get_latent_representation(adata=adata)
adata_final.obsm['scANVI_emb'] = adata.obsm['X_scanvi']

############################
# scPoli
############################

print('scPOLI model training...')
print("Setting up scPoli model...")
early_stopping_kwargs = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
scpoli_model = sca.models.scPoli(
    adata=adata,
    condition_keys=batch_key,
    cell_type_keys=labels_key,
    embedding_dims=n_latent,
    latent_dim=n_latent,
    recon_loss='nb',
)
print("Training the scPoli model...")
scpoli_model.train(
    n_epochs=50,
    n_latent=n_latent,
    pretraining_epochs=10,
    early_stopping_kwargs=early_stopping_kwargs,
    eta=5,
)
print("Extracting latent representation...")
latent_data = scpoli_model.get_latent(adata, mean=True)
adata.obsm['X_scpoli'] = latent_data
adata_final.obsm['scPOLI_emb'] = adata.obsm['X_scpoli']

############################
# expiMAP
############################

print("EXPIMAP model training...")
print("Setting up mask and counts layer...")
mask = np.ones((adata.shape[1], 20)) 
adata.varm['mask'] = mask
adata.X = adata.layers['counts'].copy()
adata.X = adata.X.astype('float32')
adata.uns['terms'] = ['Gene_set_' + str(i+1) for i in range(0,20)]
print("Initializing EXPIMAP model...")
intr_cvae = sca.models.EXPIMAP(
    adata=adata,
    condition_key=batch_key,
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
adata.obsm['X_expimap'] = intr_cvae.get_latent(mean=False, only_active=True)
adata_final.obsm['expiMAP_emb'] = adata.obsm['X_expimap']


############################
# DRVI
############################


DRVI.setup_anndata(
    adata,
    layer=layer,
    categorical_covariate_keys=[batch_key],
    is_count_data=True,
)

# construct the model
_drvi = DRVI(
    adata,
    categorical_covariates=[batch_key],
    n_latent=64,
    encoder_dims=[64, 64],
    decoder_dims=[64, 64],
)
# train the model
_drvi.train(
    max_epochs=100,
    early_stopping=False,
    early_stopping_patience=20,
)

_drvi.save(f'{model_save_path}/DRVI', overwrite=True)
print("Adding latent representation to AnnData...")
adata.obsm['X_drvi'] = _drvi.get_latent_representation()
adata_final.obsm['DRVI_emb'] = adata.obsm['X_drvi']

############################
# PCA
############################
print("Computing PCA...")
adata.X = adata.layers['log_norm'].copy()
sc.pp.pca(adata, n_comps=50,)
adata_final.obsm['X_pca'] = adata.obsm['X_pca']


############################
# Save AnnData object
############################
print("Saving AnnData object...")
adata_final.write_h5ad('/mnt/storage/Daniele/atlases/mouse/12_mouse_all_integration.h5ad')