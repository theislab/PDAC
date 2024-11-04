import scanpy as sc
import numpy as np
import pandas as pd
import os
import gc
import anndata as ad
import scarches as sca
import pytorch_lightning as pl
from scarches.models.scpoli import scPoli
from sklearn.metrics import classification_report
import sys
import traceback
import matplotlib.pyplot as plt
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/processed_datasets/')

def train_scpoli_model_label_transfer(adata, training_datasets=[], cell_type_key=None, condition_key='Dataset', n_epochs=100, n_latent=10, 
                                      pretraining_epochs=40, early_stopping_kwargs=None):
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

        adata.X = adata.X.astype(np.float32)
        if cell_type_key is None:
            source_adata = adata.copy()
            print('No cell type label detected, continuing UN-SUPERVISED training')
            cell_type_key='pseudo_label'
            unknown_ct_name='Unknown'
            source_adata.obs[cell_type_key] = unknown_ct_name

            scpoli_model = scPoli(
                adata=source_adata,
                condition_keys=condition_key,
                cell_type_keys=cell_type_key,
                embedding_dims=10,
                latent_dim=10,
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

            adata_latent_source = sc.AnnData(data_latent_source)
            adata_latent_source.obs = source_adata.obs.copy()
            print('Calculating neighbors')
            sc.pp.neighbors(adata_latent_source)
            sc.tl.leiden(adata_latent_source)
            sc.tl.umap(adata_latent_source)

            print('Saving trained model and latent space')
            # scpoli_model.save(model_save_path, save_anndata=True, overwrite=True)
            # adata_latent_source.write(latent_save_path)
            return adata_latent_source, scpoli_model
        else:
            print('Cell type label detected, continuing SUPERVISED training')
            source_adata = adata[adata.obs[condition_key].isin(training_datasets)].copy()
            target_adata = adata[~adata.obs[condition_key].isin(training_datasets)].copy()
            target_adata.obs[cell_type_key] = target_adata.obs[cell_type_key].astype(str)
            print(source_adata)
            print(target_adata)
            # Initialize and train scPoli model
            scpoli_model = scPoli(
                adata=source_adata,
                condition_keys=condition_key,
                cell_type_keys=cell_type_key,
                embedding_dims=10,
                latent_dim=20,
                recon_loss='nb',
            )
            scpoli_model.train(
                n_epochs=n_epochs,
                n_latent=n_latent,
                pretraining_epochs=pretraining_epochs,
                early_stopping_kwargs=early_stopping_kwargs,
                eta=5,
            )
            
            # target_adata.obs[cell_type_key] = 'Unknown'

            # Load query data and train
            scpoli_query = scPoli.load_query_data(
                adata=target_adata,
                reference_model=scpoli_model,
                labeled_indices=[],
            )
            scpoli_query.train(
                n_epochs=n_epochs,
                pretraining_epochs=pretraining_epochs,
                eta=10
            )

            # Classify and get results
            results_dict = scpoli_query.classify(target_adata, scale_uncertainties=True)

            # Get latent representations and concatenate data
            data_latent_source = scpoli_query.get_latent(source_adata, mean=True)
            adata_latent_source = sc.AnnData(data_latent_source)
            adata_latent_source.obs = source_adata.obs.copy()
            data_latent = scpoli_query.get_latent(target_adata, mean=True)
            adata_latent = sc.AnnData(data_latent)
            adata_latent.obs = target_adata.obs.copy()
            adata_latent.obs['cell_type_pred'] = results_dict[cell_type_key]['preds'].tolist()
            adata_latent.obs['cell_type_uncert'] = results_dict[cell_type_key]['uncert'].tolist()
            adata_latent.obs['classifier_outcome'] = (adata_latent.obs['cell_type_pred'] == adata_latent.obs[cell_type_key])

            # Get prototypes
            labeled_prototypes = scpoli_query.get_prototypes_info()
            labeled_prototypes.obs[condition_key] = 'labeled prototype'
            unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
            unlabeled_prototypes.obs[condition_key] = 'unlabeled prototype'

            # Concatenate and process data for visualization
            adata_latent_full = adata_latent_source.concatenate([adata_latent, labeled_prototypes, unlabeled_prototypes], batch_key='query')
            adata_latent_full.obs['cell_type_pred'][adata_latent_full.obs['query'].isin(['0'])] = np.nan
            sc.pp.neighbors(adata_latent_full, n_neighbors=15)
            sc.tl.umap(adata_latent_full)
            adata_latent_full.obs['scpoli_labels'] = adata_latent_full.obs.Label_Harmonized
            adata_latent_full.obs_names_make_unique()
            adata_latent_full.obs.loc[adata_latent_full.obs['scpoli_labels'] == 'Unknown', 'scpoli_labels'] = adata_latent_full.obs['cell_type_pred']
            adata_latent_full.obs['Condition'] = np.where(adata_latent_full.obs.Dataset == 'Regev', 'snRNA-seq', 'scRNA-seq')
            # Save the adata_latent_full and models
            # scpoli_model.save(model_save_path, save_anndata=True, overwrite=True)
            # model_save_path_query = model_save_path + '_query'
            # scpoli_query.save(model_save_path_query, save_anndata=True, overwrite=True)
            # adata_latent_full.write(latent_save_path)
            print('scPoli training and saving completed successfully')
            return adata_latent_full, scpoli_query
    except Exception as e:
        print(f'scPoli training failed because {e}')
        traceback.print_exc()
        return None,None

adata_filtered = sc.read_h5ad('All_genes/Concat_All_Genes_filtered.h5ad')
#raw counts in adata.X
adata_filtered.X = adata_filtered.layers['raw'].copy()
# make adata_sc
adata_sc = adata_filtered[adata_filtered.obs.Condition == 'scRNA-seq'].copy()
adata_sc.obs['batch_covariate'] = adata_sc.obs['Dataset'].astype(str) + '_' + adata_sc.obs['Condition'].astype(str)


# Normalize the counts
sc.pp.normalize_total(adata_sc, target_sum=1e4)

# Log-transform the normalized counts
sc.pp.log1p(adata_sc)

# Store the log-normalized counts in a new layer
adata_sc.layers['log_norm'] = adata_sc.X.copy()

# Revert the raw counts back into adata_filtered.X
adata_sc.X = adata_sc.layers['raw']
sc.pp.highly_variable_genes(adata_sc, layer='log_norm', batch_key='batch_covariate')
adata_hvg_sc = adata_sc[:, adata_sc.var.highly_variable].copy()
# chek the raw counts
adata_subset = sc.pp.subsample(adata_hvg_sc, copy=True, fraction=0.1)
raw_counts = adata_subset.X.toarray()
print(f"Are raw counts integers? {np.all(raw_counts.astype(int) == raw_counts)}")
print(f"Range of raw counts: {np.min(raw_counts)} to {np.max(raw_counts)}")
print("-" * 50) 
del adata_subset
adata_unsup_10, scpoli_query_unsup_10 = train_scpoli_model_label_transfer(adata_hvg_sc, n_epochs=100,
                                                                          condition_key='batch_covariate', 
                                                                          n_latent=10,)
adata_unsup_10.write('adata_unsup_scRNA.h5ad')
scpoli_query_unsup_10.save('scpoli_scRNA_unsup')