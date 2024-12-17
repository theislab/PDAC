import scanpy as sc
import numpy as np
import pandas as pd
import os
import gc
import scarches as sca
from scarches.models.scpoli import scPoli
import traceback
import matplotlib.pyplot as plt
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final/Notebooks/')

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

        print("Supervised training completed successfully.")
        return adata, scpoli_model

    except Exception as e:
        print(f"scPoli training failed: {e}")
        traceback.print_exc()
        return None, None


print("Loading gene lists...")
#load the gene list
mofa_genes_df = pd.read_csv('../Expimap/expimap_10_factors_selected_norepeatinggenes/var_names.csv')
broad_markers = pd.read_csv('broad_markers.csv', index_col='Unnamed: 0')
broad_markers_list = list(set(broad_markers.values.flatten().astype(str)[broad_markers.values.flatten().astype(str) != 'nan']))
de_genes_df = pd.read_pickle('de_genes_to_be_added.csv')
mofa_genes = mofa_genes_df.values.flatten().tolist()
de_genes = de_genes_df.values.flatten().tolist()
xenium_df = pd.read_csv('pdac_xenium_panel.csv')
xenium_genes = list(set(xenium_df.Gene.tolist()))

print(f"Length of MOFA genes: {len(mofa_genes)}")
print(f"Length of broad marker genes: {len(broad_markers_list)}")
print(f"Length of DE genes: {len(de_genes)}")
print(f"Length of xenium panel genes: {len(xenium_genes)}")


all_genes = list(set(mofa_genes + broad_markers_list + de_genes + xenium_genes))
print(f"Total unique genes combined: {len(all_genes)}")

#load the anndata objects
adata_sc = sc.read_h5ad('../single_cell_int/adata_sc_int_cnv.h5ad')
adata_sn = sc.read_h5ad('../single_nuc_int/adata_nuc_int_outlier_genes.h5ad')

valid_genes_sc = [gene for gene in all_genes if gene in adata_sc.var_names]
valid_genes_sn = [gene for gene in all_genes if gene in adata_sn.var_names]
print(f"Total valied genes combined: {len(valid_genes_sc)}")
print(f"Total valied genes combined: {len(valid_genes_sn)}")

print("Subsetting genes in AnnData objects...")
adata_sc = adata_sc[:, valid_genes_sc]
adata_sn = adata_sn[:, valid_genes_sc]

print("Resetting and renaming indices...")
adata_sc.obs.reset_index(inplace=True)
adata_sn.obs.reset_index(inplace=True)
adata_sc.obs.rename(columns={'index':'Barcode'},inplace=True)
adata_sn.obs.rename(columns={'index':'Barcode'},inplace=True)
adata_sc.obs_names = adata_sc.obs.Dataset_Barcode
adata_sn.obs_names = adata_sn.obs.Dataset_Barcode
adata_sc.obs_names= adata_sc.obs_names.astype(str)
adata_sn.obs_names= adata_sn.obs_names.astype(str)
adata_sc.obs_names_make_unique()
adata_sn.obs_names_make_unique()
adata_sn.obs['Level_1'] = adata_sn.obs.scpoli_labels.copy()
adata = adata_sc.concatenate(adata_sn)
print(f"Shape of concatenated AnnData object: {adata.shape}")
# adata = sc.read_h5ad('../Expimap/adata_combined_manual_genes.h5ad')
adata.obs_names = adata.obs_names.astype(str)
adata.obs = adata.obs.rename(columns={'Dataset_Barcode': 'Dataset_Barcode_Column'})
adata.obs.ID = adata.obs.ID.astype(str)
adata.obs.batch_covariate = adata.obs.batch_covariate.astype(str)
adata.obs['ID_batch_covariate'] = adata.obs.ID + '_' + adata.obs.batch_covariate
adata.obs.ID_batch_covariate = adata.obs.ID_batch_covariate.astype('category')
adata.obs.ID = adata.obs.ID.astype('category')

# Filter and normalize data
adata.X = adata.layers['raw'].copy()
print(f"Filtered AnnData shape: {adata.shape}")

# Train supervised scPoli
adata_int, scpoli_model = train_scpoli_supervised(
    adata=adata,
    condition_key='ID_batch_covariate',
    cell_type_key='Level_1',
    n_epochs=100,
    n_latent=10,
    pretraining_epochs=40
)

# Save results
if adata_int is not None:
    print("Saving latent space and model...")
    adata_int.write('/home/aih/shrey.parikh/PDAC/PDAC_Final/scPoli/scpoli_manual_genes_extended.h5ad')
    scpoli_model.save('/home/aih/shrey.parikh/PDAC/PDAC_Final/scPoli/scpoli_manual_genes_model_extended')
    print("Data and model saved successfully.")
else:
    adata_int.write('scpoli_manual_genes_extended.h5ad')
    scpoli_model.save('scpoli_manual_genes_extended')
    print("Training failed. Results not saved.")