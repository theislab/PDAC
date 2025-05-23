import scanpy as sc
import numpy as np
import pandas as pd
import os
import gc
import scarches as sca
from scarches.models.scpoli import scPoli
import traceback
import matplotlib.pyplot as plt
print("Starting supervised integration with scPoli...")

# Define supervised scPoli training function
def train_scpoli_supervised(adata, condition_key='ID_batch_covariate', cell_type_key='Level_1', n_epochs=100, n_latent=10, pretraining_epochs=40, early_stopping_kwargs=None):
    """
    Train an scPoli model in supervised mode with labeled datasets.

    Parameters:
        adata (AnnData): The annotated data matrix.
        condition_key (str): The key for batch/condition labels in adata.obs.
        cell_type_key (str): The key for cell type labels in adata.obs.
        n_epochs (int): Number of epochs for training.
        n_latent (int): Number of latent dimensions.
        pretraining_epochs (int): Number of pretraining epochs.
        early_stopping_kwargs (dict): Parameters for early stopping.

    Returns:
        adata_latent (AnnData): The latent space representation with embeddings.
        scpoli_model (scPoli): Trained scPoli model.
    """
    try:
        print("Setting up scPoli model for supervised training...")
        if early_stopping_kwargs is None:
            early_stopping_kwargs = {
                "early_stopping_metric": "val_prototype_loss",
                "mode": "min",
                "threshold": 0,
                "patience": 20,
                "reduce_lr": True,
                "lr_patience": 13,
                "lr_factor": 0.1,
            }

        scpoli_model = scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            embedding_dims=n_latent,
            latent_dim=n_latent,
            recon_loss='nb',
        )

        print("Training the scPoli model...")
        scpoli_model.train(
            n_epochs=n_epochs,
            n_latent=n_latent,
            pretraining_epochs=pretraining_epochs,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5,
        )

        print("Extracting latent representation...")
        latent_data = scpoli_model.get_latent(adata, mean=True)
        adata.obsm['X_scpoli'] = latent_data
        
        print("Running neighbors, Leiden clustering, and UMAP...")
        sc.pp.neighbors(adata, use_rep='X_scpoli')
        sc.tl.leiden(adata, resolution=0.25)
        sc.tl.umap(adata)
        
        #Add metrics requirements
        adata.uns['output_type'] = 'embed'
        adata.obsm['X_emb'] = adata.obsm['X_scpoli']
        
        print("Supervised training completed successfully.")
        return adata, scpoli_model

    except Exception as e:
        print(f"scPoli training failed: {e}")
        traceback.print_exc()
        return None, None

os.chdir('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/')
adata = sc.read_h5ad('/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/adata_scpoli_mg_binned.h5ad')

#Assign binned data to X
adata.X = adata.layers['binned_data'].copy()


# # Filter and normalize data
# adata.X = adata.layers['raw'].copy()
print(f"Filtered AnnData shape: {adata.shape}")

# Train supervised scPoli
adata_int, scpoli_model = train_scpoli_supervised(
    adata=adata,
    condition_key='ID_batch_covariate',
    cell_type_key='Level_1_refined',
    n_epochs=100,
    n_latent=10,
    pretraining_epochs=40
)

print("Saving latent space and model...")
adata_int.write_zarr('scpoli/scpoli_mg_binned.zarr')
scpoli_model.save('scpoli/scpoli_mg_binned')
print("Data and model saved successfully.")