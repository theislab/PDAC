import scanpy as sc
import scarches as sca
from anndata import AnnData
from scarches.dataset.trvae.data_handling import remove_sparsity
import pandas as pd
import numpy as np
import warnings
from scarches.models.scpoli import scPoli
import torch
import pytorch_lightning as pl

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

def train_scpoli_model(adata, condition_key='Dataset', n_epochs=100, n_latent=20, pretraining_epochs=40, early_stopping_kwargs=None, model_save_path='scPoli/scPoli_model_dataset', latent_save_path='scPoli/scPoli_dataset_reference_latent.h5ad'):
    """
    Train an scPoli model on single-cell RNA-seq data and save the model along with the latent space representation.

    Parameters:
        adata (AnnData): The annotated data matrix.
        condition_key (str): The key for condition labels in adata.obs.
        cell_type_key (str): The key for cell type labels in adata.obs.
        unknown_ct_name (str): The label for unknown cell types.
        n_epochs (int): Number of epochs for training.
        n_latent (int): Number of latent dimensions.
        pretraining_epochs (int): Number of pretraining epochs.
        early_stopping_kwargs (dict): Parameters for early stopping.
        eta (float): Learning rate or other hyperparameter (context dependent).
        model_save_path (str): Path to save the trained model.
        latent_save_path (str): Path to save the latent space AnnData.

    Returns:
        None
    """
    try:
        print('Setting up adata for training')
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

        source_adata = adata.copy()
        source_adata.X = source_adata.X.astype(np.float32)
        cell_type_key='pseudo_label'
        unknown_ct_name='Unknown'
        source_adata.obs[cell_type_key] = unknown_ct_name

        scpoli_model = scPoli(
            adata=source_adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            embedding_dims=5,
            recon_loss='nb',
            unknown_ct_names=[unknown_ct_name],
            labeled_indices=[],
        )
        
        print('Starting scPoli training')
        scpoli_model.train(
            n_epochs=n_epochs,
            n_latent=n_latent,
            pretraining_epochs=pretraining_epochs,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5,
        )
        
        print('Building reference latent')
        data_latent_source = scpoli_model.get_latent(
            source_adata,
            mean=True
        )
        
        # Add latent to obsm 
        adata_latent_source = sc.AnnData(data_latent_source)
        adata_latent_source.obs = source_adata.obs.copy()
        print('Calculating neighbors')
        sc.pp.neighbors(adata_latent_source)
        sc.tl.leiden(adata_latent_source)
        sc.tl.umap(adata_latent_source)

        print('Saving trained model and latent space')
        scpoli_model.save(model_save_path, save_anndata=True, overwrite=True)
        adata_latent_source.write(latent_save_path)

        print('scPoli training and saving completed successfully')

    except Exception as e:
        print(f'scPoli training failed because {e}')