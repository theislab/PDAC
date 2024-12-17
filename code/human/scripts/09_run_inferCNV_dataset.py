import scanpy as sc
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import warnings
import traceback
import infercnvpy as cnv

warnings.filterwarnings("ignore")

os.chdir('/home/aih/shrey.parikh/PDAC/PDAC_Final/')
adata_sc = sc.read_h5ad('/home/aih/shrey.parikh/PDAC/PDAC_Final/single_cell_int/adata_sc_int_outlier_genes.h5ad')
adata_sc.var_names = adata_sc.var.gene_name
training_datasets = adata_sc.obs.batch_covariate.unique().tolist()

try:
    for dataset in training_datasets:
        print(f'\033[92mRunning on dataset {dataset}\033[0m')
        
        # Subset the data
        adata_temp = adata_sc[adata_sc.obs['batch_covariate'] == dataset].copy()
        print(f"Filtering genes in dataset {dataset} with minimum 5 cells")
        sc.pp.filter_genes(adata_temp, min_cells=5)

        malignant_cells = ["Acinar Cell", "Ductal Cell", "Ductal Cell/Malignant"]
        adata_temp.obs["Reference"] = np.where(adata_temp.obs.Level_1.isin(malignant_cells), 'Potentially Malignant', 'Reference')

        # Run PCA, neighbors, UMAP, and Leiden clustering
        print(f"Running PCA, neighbors, UMAP, and Leiden clustering for dataset {dataset}")
        sc.pp.pca(adata_temp, layer='log_norm')
        sc.pp.neighbors(adata_temp)
        sc.tl.umap(adata_temp)
        sc.tl.leiden(adata_temp)

        # Run inferCNV
        print(f"Inferring CNVs for {dataset}")
        cnv.tl.infercnv(adata_temp, reference_key="Reference", reference_cat="Reference", window_size=100, layer='raw')

        # Run PCA, neighbors, Leiden clustering for CNV data
        print(f"Running PCA, neighbors, Leiden clustering for CNV data in dataset {dataset}")
        cnv.tl.pca(adata_temp)
        cnv.pp.neighbors(adata_temp)
        cnv.tl.leiden(adata_temp)

        # Run UMAP and CNV scoring
        print(f"Running UMAP and CNV scoring for dataset {dataset}")
        cnv.tl.umap(adata_temp)
        cnv.tl.cnv_score(adata_temp)

        # Save results
        print(f"Saving results for dataset {dataset}")
        output_dir = f"inferCNV/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        adata_temp.write(f"{output_dir}/PDAC_{dataset}_inferCNV.h5ad")
        print('Saving Images')

        # Save chromosome heatmap
        cnv.pl.chromosome_heatmap(adata_temp, groupby="Level_1", save=f"{dataset}_chromosome_heatmap_labels_inferCNV.png")
        cnv.pl.chromosome_heatmap(adata_temp, groupby="cnv_leiden", dendrogram=True, save=f"{dataset}_chromosome_heatmap_cnvleiden_inferCNV.png")

        # Save combined UMAP plots
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
            ax4.axis("off")
            np.random.seed(0)
            random_indices = np.random.permutation(range(adata_temp.shape[0]))
            cnv.pl.umap(adata_temp[random_indices,:], color="cnv_leiden", ax=ax1, show=False, size=5)
            cnv.pl.umap(adata_temp[random_indices,:], color="cnv_score", ax=ax2, show=False, size=5)
            cnv.pl.umap(adata_temp[random_indices,:], color="Level_1", ax=ax3, size=5)
            fig.savefig(f"{dataset}_combined_umap.png")
        except Exception as e:
            print(f"An error occurred while saving UMAP plots for {dataset}: {e}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11), gridspec_kw={"wspace": 0.5})
        ax4.axis("off")
        np.random.seed(0)
        random_indices = np.random.permutation(list(range(adata_temp.shape[0])))
        sc.pl.umap(adata_temp[random_indices,:], color="cnv_leiden", ax=ax1, show=False, size=5)
        sc.pl.umap(adata_temp[random_indices,:], color="cnv_score", ax=ax2, show=False, size=5)
        sc.pl.umap(adata_temp[random_indices,:], color="Level_1", ax=ax3, size=5)
        fig.savefig(f"{dataset}_combined_umap_transcriptomic.png")
        print(f'\033[91mCompleted inferring CNVs for {dataset}\033[0m')

        # Clean up
        del adata_temp
        gc.collect()
except Exception as e:
    print(f"An error occurred in dataset {dataset}: {e}")
    traceback.print_exc()