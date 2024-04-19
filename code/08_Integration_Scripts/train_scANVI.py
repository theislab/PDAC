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
import os
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC')
sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)


def train_scanvi_model(adata, adata_unlabeled, batch_key='Dataset', cell_type_key='Label_Harmonized', n_layers=2, n_latent=20, max_epochs_vae=150, max_epochs_scanvae=80, max_epochs_surgery=150, ref_path='scanvae/scanvae_dataset', surgery_path='scanvae/surgery_model_dataset', full_latent_path='scanvae/scanvae_dataset_full_latent.h5ad'):
    """
    Trains scANVI model for label transfer from source to target dataset.
    
    Parameters:
    - adata (AnnData): The annotated anndata object.
    - adata_unlabeled (AnnData): The non-annotated anndata object.
    - batch_key (str): The key in .obs for batch information.
    - cell_type_key (str): The key in .obs for cell type labels.
    - n_layers (int): Number of layers in the scVI model.
    - n_latent (int): Number of latent dimensions in the scVI model.
    - max_epochs_vae (int): Number of epochs for training the scVI model.
    - max_epochs_scanvae (int): Number of epochs for training the scANVI model.
    - ref_path (str): Path to save the trained scANVI model.
    - surgery_path (str): Path to save the scANVI model after label transfer.
    - full_latent_path (str): Path to save the AnnData object with combined source and target latent representations.

    Performs the following steps:
    1. Setup and train an scVI model on source data.
    2. Initialize and train scANVI for semi-supervised learning and label transfer.
    3. Apply the trained scANVI model to transfer labels to target data.
    4. Combine source and target data, analyze latent space, and perform clustering and UMAP visualization.
    5. Save the trained models and resulting AnnData objects.
    """
    
    try:
        # Setup AnnData for SCVI
        print('Starting scANVI')
        source_adata = adata.copy()
        source_adata = remove_sparsity(source_adata)
        sca.models.SCVI.setup_anndata(source_adata, batch_key=batch_key, labels_key=cell_type_key)

        # Train SCVI model
        print('Training scVI model')
        vae = sca.models.SCVI(source_adata, n_layers=n_layers, n_latent=n_latent, encode_covariates=True, deeply_inject_covariates=False, use_layer_norm="both", use_batch_norm="none")
        vae.train(max_epochs=max_epochs_vae)

        # Initialize and train SCANVI
        print('Initializing and training scANVI')
        scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
        scanvae.train(max_epochs=max_epochs_scanvae)
        scanvae.save(ref_path, overwrite=True)
        source_adata.obs['predictions_source'] = scanvae.predict()
        print("ACCURACY ON SOURCE PREDICTION: {}".format(np.mean(source_adata.obs.predictions_source == source_adata.obs[cell_type_key])))

        # Prepare target_adata for label transfer
        target_adata = adata_unlabeled.copy()
        print('Preparing target_adata for label transfer')
        target_adata.obs['Annotation'] = scanvae.unlabeled_category_
        model = scanvae.load_query_data(target_adata, scanvae, freeze_dropout=True)
        model._unlabeled_indices = np.arange(target_adata.n_obs)
        model._labeled_indices = []
        print('Training model with target_adata')
        model.train(max_epochs=max_epochs_surgery, plan_kwargs=dict(weight_decay=0.0), check_val_every_n_epoch=10)
        model.save(surgery_path, overwrite=True)

        # Combine source and target data, generate latent representation, and perform analysis
        print('Building full_latent')
        adata_full = source_adata.concatenate(target_adata)
        full_latent = sc.AnnData(model.get_latent_representation(adata=adata_full))
        full_latent.obs_names = adata_full.obs_names
        full_latent.obs['Label_Harmonized'] = adata_full.obs[cell_type_key].tolist()
        full_latent.obs['Dataset'] = adata_full.obs[batch_key].tolist()
        full_latent.obs['predictions'] = model.predict(adata=adata_full)
        full_latent.obs['pseudo_barcode'] = adata_full.obs.pseudo_barcode.copy()
        print('Calculating neighbors')
        sc.pp.neighbors(full_latent)
        sc.tl.leiden(full_latent)
        sc.tl.umap(full_latent)
        full_latent.write(full_latent_path)
        print('scANVI training and saving completed successfully')

        
    except Exception as e:
        print(f'scANVI training failed because {e}🥲') 
        
def train_scanvi_model_separated(adata, adata_unlabeled, batch_key='Dataset', cell_type_key='Label_Harmonized', n_layers=2, n_latent=20, max_epochs_vae=150, max_epochs_scanvae=80, max_epochs_surgery=150, ref_path='scanvae/scanvae_dataset', surgery_path='scanvae/surgery_model_dataset', full_latent_path='scanvae/scanvae_dataset_full_latent.h5ad'):
    """
    Trains scANVI model for label transfer from source to target dataset.
    
    Parameters:
    - adata (AnnData): The annotated anndata object.
    - adata_unlabeled (AnnData): The non-annotated anndata object.
    - batch_key (str): The key in .obs for batch information.
    - cell_type_key (str): The key in .obs for cell type labels.
    - n_layers (int): Number of layers in the scVI model.
    - n_latent (int): Number of latent dimensions in the scVI model.
    - max_epochs_vae (int): Number of epochs for training the scVI model.
    - max_epochs_scanvae (int): Number of epochs for training the scANVI model.
    - ref_path (str): Path to save the trained scANVI model.
    - surgery_path (str): Path to save the scANVI model after label transfer.
    - full_latent_path (str): Path to save the AnnData object with combined source and target latent representations.

    Performs the following steps:
    1. Setup and train an scVI model on source data.
    2. Initialize and train scANVI for semi-supervised learning and label transfer.
    3. Apply the trained scANVI model to transfer labels to target data.
    4. Combine source and target data, analyze latent space, and perform clustering and UMAP visualization.
    5. Save the trained models and resulting AnnData objects.
    """
    
    try:
        # Setup AnnData for SCVI
        print('Starting scANVI')
        source_adata = adata.copy()
        source_adata = remove_sparsity(source_adata)
        sca.models.SCVI.setup_anndata(source_adata, batch_key=batch_key, labels_key=cell_type_key)

        # Train SCVI model
        print('Training scVI model')
        vae = sca.models.SCVI(source_adata, n_layers=n_layers, n_latent=n_latent, encode_covariates=True, deeply_inject_covariates=False, use_layer_norm="both", use_batch_norm="none")
        vae.train(max_epochs=max_epochs_vae)

        # Initialize and train SCANVI
        print('Initializing and training scANVI')
        scanvae = sca.models.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
        scanvae.train(max_epochs=max_epochs_scanvae)
        scanvae.save(ref_path, overwrite=True)
        # Prepare and predict labels for target_adata individually
        target_adata = adata_unlabeled.copy()
        print('Preparing target_adata for label transfer')
        target_adata.obs['Annotation'] = scanvae.unlabeled_category_
        target_model = scanvae.load_query_data(target_adata, scanvae, freeze_dropout=True)
        target_model._unlabeled_indices = np.arange(target_adata.n_obs)
        target_model._labeled_indices = []
        print('Training model with target_adata')
        target_model.train(max_epochs=max_epochs_surgery, plan_kwargs=dict(weight_decay=0.0), check_val_every_n_epoch=10)
        target_model.save(surgery_path, overwrite=True)
        # Predict labels for target_adata
        target_latent = sc.AnnData(target_model.get_latent_representation(adata=target_adata))
        target_latent.obs['predictions'] = target_model.predict(adata=target_adata)
        target_latent.obs['Label_Harmonized'] = 'Unknown'
        # Predict labels for source_adata individually
        source_latent = sc.AnnData(scanvae.get_latent_representation(adata=source_adata))
        source_latent.obs['predictions_source'] = scanvae.predict(adata=source_adata)
        # source_latent.obs[cell_type_key] = source_adata.obs[cell_type_key]
        # source_latent.obs = source_latent.obs.astype(str)
        # print("ACCURACY ON SOURCE PREDICTION: {}".format(np.mean(source_latent.obs.predictions_source == source_latent.obs[cell_type_key])))
        # Combine source and target data, now with predictions
        print('Building full_latent with individual predictions')
        adata_full = source_adata.concatenate(target_adata)
        full_latent = source_latent.concatenate(target_latent)
        
        # Use metadata from adata_full for full_latent
        full_latent.obs_names = adata_full.obs_names
        full_latent.obs['Label_Harmonized'] = adata_full.obs[cell_type_key].tolist()
        full_latent.obs['Dataset'] = adata_full.obs[batch_key].tolist()
        full_latent.obs['pseudo_barcode'] = adata_full.obs.pseudo_barcode.copy()
        print('Calculating neighbors, performing clustering and UMAP visualization')
        sc.pp.neighbors(full_latent)
        sc.tl.leiden(full_latent)
        sc.tl.umap(full_latent)
        full_latent.write(full_latent_path)
        print('scANVI training and label prediction completed successfully')

    except Exception as e:
        print(f'scANVI training and label prediction failed because {e}')
