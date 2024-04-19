import os
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import torch
import pytorch_lightning as pl

sys.path.insert(0, '/home/aih/shrey.parikh/cross_species/multiple_system_integration/cross_system_integration/')
# sys.path.append('/home/aih/shrey.parikh/cross_species/multiple_system_integration/cross_system_integration/cross_system_integration/')
from cross_system_integration.model._xxjointmodel import XXJointModel

def train_sysvi_model(adata, system_key='Dataset', condition_key='Condition', max_epochs=200, embed_save_path='sysVI/sysVI_dataset_reference_latent.h5ad',
                       model_save_path='sysVI/sysVI_model'):
    """
    Trains an XXJointModel on the provided AnnData object and saves the resulting embedding.

    Parameters:
        adata (AnnData): The annotated data matrix.
        system_key (str): Key in adata.obs to use as system labels.
        condition_key (str): Key in adata.obs to use as condition labels.
        max_epochs (int): Maximum number of training epochs.
        embed_save_path (str): Path to save the embedding AnnData object.

    Returns:
        None
    """
    try:
        print('Setting up adata for XXJointModel')
        adata_copy = adata.copy()
        adata_prepared = XXJointModel.setup_anndata(adata_copy, group_key=None, system_key=system_key, categorical_covariate_keys=[condition_key])

        print('Initialising XXJointModel')
        model = XXJointModel(adata=adata_prepared,
                             prior='vamp', n_prior_components=5, pseudoinputs_data_init=True,
                             trainable_priors=True,
                             encode_pseudoinputs_on_eval_mode=True,
                             z_dist_metric = 'MSE_standard', n_layers=2,
                             n_hidden=256)

        print('Starting training')
        model.train(
            max_epochs=80,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            train_size=0.9,
            plan_kwargs={
                'optimizer': "Adam",
                'lr': 0.001,
                'reduce_lr_on_plateau': False,
                'lr_scheduler_metric': 'loss_train',  # Replace with default value
                'lr_patience': 5,  # Replace with default value
                'lr_factor': 0.1,  # Replace with default value
                'lr_min': 1e-7,  # Replace with default value
                'lr_threshold_mode': 'rel',  # Replace with default value
                'lr_threshold': 0.1,  # Replace with default value
                'log_on_epoch': True,  # Replace with default value
                'log_on_step': False,  # Replace with default value
                'loss_weights': {
                    'kl_weight': 1.0,  # Replace with default value
                    'reconstruction_weight': 1.0,  # Replace with default value
                    'z_distance_cycle_weight': 5.0, 
                },
            }
        )
        print('Generating embedding')
        embed = model.embed(adata=adata_prepared)
        embed_adata = sc.AnnData(embed, obs=adata.obs)
        sc.pp.neighbors(embed_adata, use_rep='X')
        sc.tl.umap(embed_adata)
        embed_adata.write(embed_save_path)
        model.save(model_save_path, overwrite=True)    
        
        print('sysVI training and saving completed successfully')

    except Exception as e:
        print(f'XXJointModel training failed because {e}')
