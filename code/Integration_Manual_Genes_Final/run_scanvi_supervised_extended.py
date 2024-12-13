import scanpy as sc
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import traceback
import os
import anndata as ad

os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final/Notebooks/')

def train_scanvi_add_embedding(
    adata,
    batch_key='ID_batch_covariate',
    cell_type_key='Level_1',
    n_latent=10,
    n_layers=2,
    max_epochs_vae=150,
    max_epochs_scanvae=80,
    ref_path='scanvae/scanvae_model',
    full_adata_path='scanvae/scanvae_adata_with_embedding.h5ad'
):
    """
    Train scANVI model and add latent representation to the input AnnData object.

    Parameters:
        adata (AnnData): Annotated data matrix with labels for supervised training.
        batch_key (str): Key for batch information in adata.obs.
        cell_type_key (str): Key for cell type labels in adata.obs.
        n_latent (int): Number of latent dimensions for scANVI.
        n_layers (int): Number of layers for the scANVI model.
        max_epochs_vae (int): Epochs for pretraining the scVI model.
        max_epochs_scanvae (int): Epochs for training the scANVI model.
        ref_path (str): Path to save the scANVI model.
        full_adata_path (str): Path to save the updated AnnData object.

    Returns:
        adata (AnnData): Updated AnnData object with latent representation in `.obsm`.
        scanvae_model (SCANVI): Trained scANVI model.
    """
    try:
        print("Setting up scVI/scANVI model...")
        # Remove sparsity for compatibility
        adata = remove_sparsity(adata.copy())        
        sca.models.SCVI.setup_anndata(adata, batch_key=batch_key, labels_key=cell_type_key)
        print("Training scVI model...")
        vae = sca.models.SCVI(
            adata,
            n_layers=n_layers,
            n_latent=n_latent,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none"
        )
        vae.train(max_epochs=max_epochs_vae)

        print("Initializing and training scANVI model...")
        scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
        scanvae.train(max_epochs=max_epochs_scanvae)
        scanvae.save(ref_path, overwrite=True)

        print("Adding latent representation to AnnData...")
        adata.obsm['X_scanvi'] = scanvae.get_latent_representation(adata=adata)

        print("Performing clustering and UMAP visualization...")
        sc.pp.neighbors(adata, use_rep='X_scanvi')
        sc.tl.leiden(adata, resolution=0.25)
        sc.tl.umap(adata)
        #prepare for metrics
        adata.uns['output_type'] = 'embed'
        adata.obsm['X_emb'] = adata.obsm['X_scanvi']
        adata.obsm['X_umap_scanvi'] = adata.obsm['X_umap']
        # Save updated AnnData object
        adata.write_zarr(full_adata_path)
        print("scANVI training and embedding generation completed successfully.")
        return adata, scanvae

    except Exception as e:
        print(f"scANVI training failed: {e}")
        traceback.print_exc()
        return None, None


# Example usage
print("Loading AnnData object...")
adata = sc.read_h5ad('../concatenated/adata_manual_genes_extended.h5ad')
print(adata)
adata.X = adata.layers['raw'].copy()

print("Starting scANVI training...")
adata_updated, scanvae_model = train_scanvi_add_embedding(
    adata=adata,
    batch_key='ID_batch_covariate',
    cell_type_key='Level_1',
    n_latent=10,
    n_layers=2,
    max_epochs_vae=100,
    max_epochs_scanvae=80,
    ref_path='../EMG_Embedding/scanvae/scanvi_model_extended',
    full_adata_path='../EMG_Embedding/scanvae/scanvi_manual_genes_extended.zarr'
)

if adata_updated is not None:
    print("Saving updated AnnData object...")
else:
    print("scANVI training failed.")