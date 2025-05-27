# Notebook Structure

Below is an overview of the project notebook structure:

### **00_DataProcessor**
- Processing individual datasets from each publication.
- Building `h5ad` files and initial dataset preparation.

### **01_Preprocess**
- Basic filtering of individual datasets from each publication.

### **02_RDS_toH5AD**
- Conversion of datasets from RDS format to `h5ad`.

### **03_Concat_Ind_Dataset**
- Concatenation of individual datasets into one dataset, named by the publication.

### **04_DataStructure_Plots**
- Visualization of dataset characteristics:
  - Bar plot and pie charts for the number of cells obtained vs. quoted in the publication.
  - Normalized count distribution per dataset.

### **04_Explore**
- Exploration of individual datasets.
- Examination of available information in `obs` and `var`.

### **05_Concat_Anndata**
- Check for raw counts in each dataset.
- Removal of some datasets and final concatenation into a single object (`Concat_All_Genes_filtered.h5ad`).

### **06_single_cell_annot**
- Splitting the dataset into scRNA-seq datasets (`adata_sc`).
- Highly Variable Gene (HVG) selection.
- Integration using the `scpoli.py` script.
- Transfer of embeddings to `adata_sc` with all genes.
- Annotation:
  - **Level 0**: Leiden clustering and Differentially Expressed (DE) genes.
  - **Level 1**: Subclustering of Level 0 cell types.

### **07_single_nuc_int_annot_transfer**
- Splitting the dataset into snRNA-seq datasets (`adata_sn`).
- Label harmonization (e.g., Hwang L2 labels).
- Label transfer using `scpoli` from Hwang to Ding snRNA-seq datasets.

### **08_Outliers**
- Quality control (QC) for `adata_sc` and `adata_sn` separately.
- Outlier identification using 4 Median Absolute Deviations (MAD) for:
  - `log1p_total_counts`
  - `log1p_n_genes_by_counts`
  - `pct_counts_mito`
- Visualization of outliers.

### **08_2_Harmonize_Var_Names**
- Harmonization of variable names:
  - Addition of Ensembl IDs.
  - Removal of unusual genes.
- Final datasets saved as:
  - `adata_sc_int_outlier_genes`
  - `adata_nuc_int_outlier`

### **09_InferCNV**
- Running InferCNV per dataset:
  - All cells (except Ductal/Acinar/Malignant) taken as reference.
  - Individual thresholds set for each dataset to identify malignant cells.
- Added `obs` columns:
  - `cnv_score_abs`: Absolute CNV score.
  - `infercnv_score_malignant`: Malignant vs. non-malignant distinction based on score.
  - `infercnv_score_malignant_refined`: Refinement based on cell type.

### **10_MOFA**
- 12,000 HVG selected for `adata_sc` and `adata_sn`.
- 6,082 HVG used to run MOFA.
- MOFA run to deduce 15 factors.

### **11_Analyse_Factors**
- Analysis of MOFA factors:
  - Violin plots showing variance captured by each factor per cell type.

### **13_Refine_MOFA_Factors**
- Refinement of MOFA factors:
  - Selection of 10/15 factors.
  - Removal of overlapping genes across factors.

### **15_Create_MG**
- Creation of Manual Genes:
  - Integration of `MOFA+Broad+DE+Xenium` resulting in 2,505 genes.

### **Integration Scripts**
- Various integration scripts used:
  - `drVI`, `scVI`, `scanVI`, `scPoli`, `sysVI`, `Expimap`.