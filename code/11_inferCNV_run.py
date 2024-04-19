import os
import warnings
import warnings ; warnings.warn = lambda *args,**kwargs: None
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import infercnvpy as cnv
import genomic_features as gf
import traceback

# Set the working directory
os.chdir('/home/aih/shrey.parikh/PDAC/PDAC/')

ensdb = gf.ensembl.annotation(species="Hsapiens", version="108")
genes = ensdb.genes()
dict_map = {gene: (start, end, chrom) for gene, start, end, chrom in zip(genes.gene_name, genes.gene_seq_start, genes.gene_seq_end, genes.seq_name)}

    
# Read the reference dataset
ref = sc.read_h5ad("/home/aih/shrey.parikh/PDAC/PDAC/concatenated_datasets/PDAC_concat_all_genes.h5ad")
ref.X = ref.layers['raw']
training_datasets=['Regev', 'Peng', 'Moncada', 'Schlesinger', 'Zenodo_OUGS']

# # Get unique datasets
# unique_datasets = ref.obs['Dataset'].unique()
try:
    for dataset in training_datasets:
        print(f'Running on dataset {dataset}')
    #     # Subset the data
        ref_subset = ref[ref.obs['Dataset'] == dataset].copy()
        sc.pp.filter_genes(ref_subset, min_cells=5)
        ref_subset.var['start'] = ref_subset.var.index.map(lambda x: dict_map[x][0] if x in dict_map else None)
        ref_subset.var['end'] = ref_subset.var.index.map(lambda x: dict_map[x][1] if x in dict_map else None)
        ref_subset.var['chromosome'] = ref_subset.var.index.map(lambda x: dict_map[x][2] if x in dict_map else None)
        ref_subset.var['chromosome'] = 'chr' + ref_subset.var['chromosome']


        malignant_cells = ["Acinar cell", "Ductal Cell", "Malignant Epithelial Cell", "Epithelial Cell"]
        ref_subset.obs["Reference"] = np.where(ref_subset.obs.Label_Harmonized.isin(malignant_cells), 'Potentially Malignant', 'Reference')
    #     # Normalize, log transform, and perform dimensionality reduction
        sc.pp.normalize_total(ref_subset, target_sum=1e4)
        sc.pp.log1p(ref_subset)
        sc.pp.neighbors(ref_subset)
        sc.tl.umap(ref_subset)
        sc.tl.leiden(ref_subset)
        print(f'infering CNVs for {dataset}')
    #     # Run inferCNV
        cnv.tl.infercnv(
            ref_subset,
            reference_key="Reference",
            reference_cat="Reference",
            window_size=150,
        )

    #     # PCA and neighbor graph for CNV data
        cnv.tl.pca(ref_subset)
        cnv.pp.neighbors(ref_subset)
        cnv.tl.leiden(ref_subset)

    #     # Run UMAP and CNV scoring
        cnv.tl.umap(ref_subset)
        cnv.tl.cnv_score(ref_subset)

    #     # Save results
        output_dir = f"inferCNV/{dataset}"
        os.makedirs(output_dir, exist_ok=True)
        ref_subset.write(f"{output_dir}/PDAC_{dataset}_inferCNV.h5ad")
        print('Saving Images')
    #     # Save chromosome heatmap
        cnv.pl.chromosome_heatmap(ref_subset, groupby="Label_Harmonized", save=f"{dataset}_chromosome_heatmap_labels_inferCNV.png")
        cnv.pl.chromosome_heatmap(ref_subset, groupby="cnv_leiden", dendrogram=True, save=f"{dataset}_chromosome_heatmap_cnvleiden_inferCNV.png")

    #     # Save combined UMAP plots
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
            ax4.axis("off")
            np.random.seed(0)
            random_indices = np.random.permutation(range(ref_subset.shape[0]))
            cnv.pl.umap(ref_subset[random_indices,:], color="cnv_leiden", ax=ax1, show=False, size=5)
            cnv.pl.umap(ref_subset[random_indices,:], color="cnv_score", ax=ax2, show=False, size=5)
            cnv.pl.umap(ref_subset[random_indices,:], color="Label_Harmonized", ax=ax3, size=5)
            fig.savefig(f"{dataset}_combined_umap.png")
        except Exception as e:
            print(f"An error occurred while saving UMAP plots for {dataset}: {e}")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11), gridspec_kw={"wspace": 0.5})
        ax4.axis("off")
        np.random.seed(0)
        random_indices = np.random.permutation(list(range(ref.shape[0])))
        sc.pl.umap(ref[random_indices,:], color="cnv_leiden", ax=ax1, show=False, size=5)
        sc.pl.umap(ref[random_indices,:], color="cnv_score", ax=ax2, show=False, size=5)
        sc.pl.umap(ref[random_indices,:], color="Label_Harmonized", ax=ax3, size=5)
        fig.savefig(f"inferCNV/{dataset}_combined_umap_transcriptomic.png")
        print(f'Completed inferring CNVs for {dataset}')
except Exception as e:
    print(f"An error occurred in loop: {dataset}")
    traceback.print_exc()

    # ref.write("inferCNV/PDAC_concat_all_genes_inferCNV.h5ad")