import scanpy as sc
import scarches as sca
from anndata import AnnData
from scarches.dataset.trvae.data_handling import remove_sparsity
import pandas as pd
import numpy as np
import warnings
import torch
import pytorch_lightning as pl

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def train_scvi_model(adata, condition_key='Condition', batch_key='Dataset', 
                     n_layers=2, n_latent=20, max_epochs=100,
                     model_save_path='scVI/scVI_model_dataset', 
                     latent_save_path='scVI/scVI_dataset_reference_latent.h5ad'):
    """
    Train an scVI model on specified dataset.

    This function preprocesses the input data, configures and trains an scVI model,
    and saves both the model and its latent representation of the input data.

    Parameters:
    - adata (AnnData): The annotated data matrix to train on.
    - condition_key (str): The key in adata.obs to use as condition labels. Default is 'Condition'.
    - batch_key (str): The key in adata.obs to use as batch labels. Default is 'Dataset'.
    - n_layers (int): The number of layers in the VAE encoder/decoder. Default is 2.
    - n_latent (int): The dimensionality of the latent space. Default is 20.
    - max_epochs (int): The maximum number of training epochs. Default is 100.
    - model_save_path (str): Path where the trained model will be saved. Default is 'scVI/scVI_model_dataset'.
    - latent_save_path (str): Path where the latent space representation will be saved. Default is 'scVI/scVI_dataset_reference_latent.h5ad'.

    Returns:
    - None: The trained model and latent space are saved to files specified by model_save_path and latent_save_path.

    Raises:
    - Exception: Descriptive error message if training fails.
    """
    try:
        print('Starting scVI')
        source_adata = adata.copy()
        source_adata = remove_sparsity(source_adata)
        print('Setting up adata for training')
        sca.models.SCVI.setup_anndata(source_adata, batch_key=batch_key, categorical_covariate_keys=[condition_key])

        print('Starting Training')
        vae = sca.models.SCVI(
            source_adata,
            n_layers=n_layers,
            n_latent=n_latent,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )

        vae.train(max_epochs=max_epochs)

        print('Training Finished')
        print('Building reference latent')
        reference_latent = AnnData(vae.get_latent_representation())
        reference_latent.obs["ID"] = source_adata.obs['ID'].tolist()
        reference_latent.obs[batch_key] = source_adata.obs[batch_key].tolist()
        reference_latent.obs[condition_key] = source_adata.obs[condition_key].tolist()
        reference_latent.obs['cell_barcodes'] = source_adata.obs.index
        reference_latent.obs['pseudo_barcode'] = source_adata.obs['pseudo_barcode']
        print('Calculating neighbors')
        sc.pp.neighbors(reference_latent)
        sc.tl.leiden(reference_latent)
        sc.tl.umap(reference_latent)
        print('scVI training and saving completed successfully ')
        vae.save(model_save_path, overwrite=True)
        reference_latent.write(latent_save_path)
        
    except Exception as e:
        print(f'scVI training failed because {e}')
