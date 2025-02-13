import scanpy as sc
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import pandas as pd
import numpy as np
import warnings
import torch
import anndata as ad

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

sc.settings.set_figure_params(dpi=200, frameon=False)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def train_scvi_model(
    adata, 
    condition_key='Condition', 
    batch_key='ID_batch_covariate',
    labels_key='Level_1',
    n_layers=2, 
    n_latent=20, 
    max_epochs=100,
    model_save_path='scVI/scVI_model_dataset', 
    latent_save_path='scVI/scVI_dataset_reference_latent.h5ad'
):
    """
    Train an scVI model and embed latent representation in the AnnData object.

    Parameters:
        adata (AnnData): The annotated data matrix to train on.
        condition_key (str): Key in `adata.obs` for condition labels.
        batch_key (str): Key in `adata.obs` for batch labels.
        labels_key (str): Key in `adata.obs` for cell type labels.
        n_layers (int): Number of layers in the VAE encoder/decoder.
        n_latent (int): Dimensionality of the latent space.
        max_epochs (int): Maximum number of training epochs.
        model_save_path (str): Path to save the trained model.
        latent_save_path (str): Path to save the latent space representation.

    Returns:
        adata (AnnData): Updated AnnData object with latent representation in `.obsm`.
    """
    try:
        print('Starting scVI...')
        adata = remove_sparsity(adata)
        print('Setting up AnnData for training...')
        sca.models.SCVI.setup_anndata(
            adata, 
            batch_key=batch_key, 
            labels_key=labels_key, 
            categorical_covariate_keys=[condition_key]
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

        vae.train(max_epochs=max_epochs)

        print('Saving trained model...')
        vae.save(model_save_path, overwrite=True)

        print('Embedding latent representation...')
        adata.obsm['X_scvi'] = vae.get_latent_representation()

        print("Running neighbors, Leiden clustering, and UMAP...")
        sc.pp.neighbors(adata, use_rep='X_scvi')
        sc.tl.leiden(adata, resolution=0.25)
        sc.tl.umap(adata)

        # Prepare for metrics and visualization
        adata.uns['output_type'] = 'embed'
        adata.obsm['X_emb'] = adata.obsm['X_scvi']
        adata.obsm['X_umap_scvi'] = adata.obsm['X_umap']
        print("Saving AnnData with latent representation...")
        adata.write_zarr(latent_save_path)

        print("scVI training and latent embedding completed successfully.")
        return adata

    except Exception as e:
        print(f"scVI training failed because {e}")
        return None


# Main script
print("Loading AnnData object...")
adata = sc.read_h5ad('../concatenated/adata_manual_genes_extended.h5ad')
print(adata)
adata.X = adata.layers['raw'].copy()

print("Starting scVI training...")
adata_updated = train_scvi_model(
    adata=adata,
    condition_key='Condition',
    batch_key='ID_batch_covariate',
    labels_key='Level_1',
    n_latent=10,
    n_layers=2,
    max_epochs=100,
    model_save_path='../EMG_Embedding/scVI/scvi_model_extended',
    latent_save_path='../EMG_Embedding/scVI/scvi_manual_genes_extended.zarr'
)

if adata_updated is not None:
    print("Saving updated AnnData object...")
else:
    print("scVI training failed.")