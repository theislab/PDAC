import warnings

warnings.filterwarnings('ignore')

import anndata as ad
import spatialdata as sd
import numpy as np
from scipy.sparse import issparse

import pandas as pd
import scanpy as sc

from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from typing import Dict, Tuple, Optional



plt.rcParams['axes.grid'] = False
plt.rcParams['axes.axisbelow'] = False  

# Seaborn styling
sns.set_style("white", {"axes.grid": False})

plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams['grid.alpha'] = 0
plt.rcParams['grid.linewidth'] = 0

import scanpy as sc
sc.settings.set_figure_params(
    frameon=False,  
    dpi=100,
    dpi_save=300
)

save_dir = "/mnt/t06/Cell/AG-Saur/KKF2/PhilippP/06_scRNA/collaborations/TK/Projekt_Kras_Inhib/v2/"



cell_type_colors2 = {
    "Fibroblasts": "#1f77b4",
    "T cells": "#ff7f0e",
    "Myeloid cells": "#2ca02c",
    "Tumor cells": "#d62728",
    "Endothelial cells": "#9467bd",
    "B cells": "#8c564b",
    "NK cells": "#e377c2",
    "Pericytes": "#7f7f7f",
    "Other": "#bcbd22",
    "Unknown": "#17becf",
}

def normalize_by_cell_area(adata, target_sum=1e3, key='area', inplace=True):
        """
        Normalize total counts per cell using cell area instead of library size.

        Parameters
        ----------
        adata : AnnData
            The annotated data matrix.
        target_sum : float
            The total count per cell after normalization (scaled from cell area).
        key : str
            Column in `adata.obs` to use for normalization (default is 'cell_area').
        inplace : bool
            Whether to update the adata.X in-place or return a new matrix.

        Returns
        -------
        If inplace=False, returns the normalized matrix.
        """
        if key not in adata.obs:
            raise ValueError(f"'{key}' not found in adata.obs")

        cell_area = adata.obs[key].values.astype(np.float64)
        if np.any(cell_area <= 0):
            raise ValueError("All cell areas must be positive.")

        scale_factors = target_sum / cell_area
        if issparse(adata.X):
            adata.X = adata.X.multiply(scale_factors[:, np.newaxis])
            adata.X = adata.X.tocsr()
        else:
            adata.X *= scale_factors[:, np.newaxis]

        if inplace:
            return None
        else:
            return adata.X
        


def load_data(adata_path: str, sdata_path: str) -> Tuple[ad.AnnData, sd.SpatialData]:
    """
    Load AnnData and SpatialData objects.
    
    """
    print("Loading data...")
    adata = ad.read_h5ad(adata_path)
    sdata = sd.read_zarr(sdata_path)
    print(f"Loaded {adata.n_obs} cells from AnnData")

    return adata, sdata


def transfer_obs_to_sdata(adata: ad.AnnData, 
                         sdata: sd.SpatialData,
                         table_name: str = "table",
                         cell_id_column: str = "cell_id",
                         strip_suffix: bool = True,
                         suffix_pattern: str = "-27537_hard$") -> None:
    """
    Transfer adata.obs annotations to sdata table
    """
    print("\nTransferring annotations from AnnData to SpatialData...")
    
    if table_name not in sdata.tables:
        if len(sdata.tables) > 0:
            table_name = list(sdata.tables.keys())[0]
        else:
            raise ValueError("No tables found in sdata")
    
    sdata_table = sdata.tables[table_name]
    
    # --- Index/ID matching logic ---
    if cell_id_column in sdata_table.obs.columns:
        import re
        sdata_cell_ids = sdata_table.obs[cell_id_column].values
        
        if strip_suffix:
            adata_cell_ids = pd.Series(adata.obs_names).str.replace(suffix_pattern, '', regex=True).values
        else:
            adata_cell_ids = adata.obs_names.values
        
        adata_df = adata.obs.copy()
        name_map = {}
        for col in adata_df.columns:
            new_col = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
            import re
            new_col = re.sub(r'[^\w\.\-]+', '', new_col) 
            if new_col != col:
                name_map[col] = new_col
        
        if name_map:
            adata_df.rename(columns=name_map, inplace=True)
            
        adata_df.index = adata_cell_ids
        common_cells = set(adata_cell_ids).intersection(set(sdata_cell_ids))
        
        if len(common_cells) == 0:
            # Check for the case where the indices might already match
            if len(adata_cell_ids) == len(sdata_cell_ids) and np.array_equal(adata_cell_ids, sdata_cell_ids):
                # If they are the same, proceed with direct index mapping
                pass 
            else:
                 raise ValueError("No matching cells found! Check cell ID format.")
        
        transferred_cols = []
        for col in adata_df.columns:
            if col not in sdata_table.obs.columns:
                cell_id_to_value = dict(zip(adata_cell_ids, adata_df[col].values))
                sdata_table.obs[col] = [cell_id_to_value.get(cid, np.nan) for cid in sdata_cell_ids]
                transferred_cols.append(col)
        
    else:
        # Fallback: try direct index matching
        common_idx = adata.obs.index.intersection(sdata_table.obs.index)
        for col in adata.obs.columns:
            if col not in sdata_table.obs.columns:
                sdata_table.obs[col] = adata.obs.loc[common_idx, col]
        
    
    sdata.tables["table"].obs["region"] = "cell_boundaries"
    sdata.set_table_annotates_spatialelement("table", region="cell_boundaries")

    area_column = None
    if 'area' in adata.obs.columns:
        area_column = 'area'
    elif 'cell_area' in adata.obs.columns:
        area_column = 'cell_area'
    elif 'volume' in adata.obs.columns:
        area_column = 'volume'

    print("\nNormalizing SpatialData counts by cell area...")
    print(f"Using area column: {area_column}")

  #  print("Before normalization: {xxxxxx}")
    normalize_by_cell_area(sdata.tables["table"], target_sum=1e3, key=area_column, inplace=True)
  #  print("After normalization: {xxxxxx}")

    sc.pp.log1p(sdata.tables["table"])
    sdata.tables["table"].layers["log1p_norm"] = sdata.tables["table"].X


    print("Annotations transfer complete.")




def prepare_sdata_for_plotting(sdata: sd.SpatialData, 
                               cell_type_column: str = "annotation_level2") -> None:
    """
    Prepare sdata for plotting by handling NaN values in categorical columns
    """
    print("\nPreparing sdata for plotting...")
    
    sdata_table = sdata.tables["table"]
    
    if cell_type_column in sdata_table.obs.columns:
        # Add empty category for NaN values
        sdata_table.obs[cell_type_column] = sdata_table.obs[cell_type_column].cat.add_categories(" ")
        sdata_table.obs[cell_type_column] = sdata_table.obs[cell_type_column].fillna(" ")
        
    print("Preparation complete.")


def crop_sdata_region(sdata: sd.SpatialData,
                      roi_info: pd.Series,
                      axes: Tuple[str, str] = ("x", "y")) -> sd.SpatialData:
    """
    (KEPT) Crop a region from SpatialData.
    This works perfectly with the new grid_df rows.
    """
    from spatialdata import bounding_box_query
    

    cropped_sdata = bounding_box_query(
        sdata,
        min_coordinate=[roi_info['roi_xmin'] / 0.2125, roi_info['roi_ymin'] / 0.2125],
        max_coordinate=[roi_info['roi_xmax'] / 0.2125, roi_info['roi_ymax'] / 0.2125],
        axes=axes,
        target_coordinate_system="global" 
    )

    # Check if the table exists before accessing n_obs
    if "table" in cropped_sdata.tables:
        n_cells = cropped_sdata.tables["table"].n_obs
    else:
        n_cells = 0
        
    print(f"Cropped region ROI {roi_info['roi_id']}: {n_cells} cells.")
    
    if n_cells == 0:
        print(f"  WARNING: No cells in cropped region! Coordinates:")
        print(f"    X: {roi_info['roi_xmin']:.1f} to {roi_info['roi_xmax']:.1f}")
        print(f"    Y: {roi_info['roi_ymin']:.1f} to {roi_info['roi_ymax']:.1f}")
    
    return cropped_sdata

def plot_roi_cell_types(sdata: sd.SpatialData,
                        roi_info: pd.Series,
                        cell_type_column: str,
                        element_name: str = "cell_boundaries",
                        figsize: Tuple[int, int] = (15, 15),
                        save_path: Optional[str] = None):
    print(f"\nPlotting cell types for ROI {roi_info['roi_id']}...")
    
    try:
        cropped_sdata = crop_sdata_region(sdata, roi_info)
        
        if "table" not in cropped_sdata.tables or cropped_sdata.tables["table"].n_obs == 0:
            print(f"Skipping plot for ROI {roi_info['roi_id']}: No cells in cropped region.")
            return

        if element_name not in cropped_sdata.shapes:
            print(f"Warning: '{element_name}' not in cropped region")
            return
        
        title = f"{cell_type_column}"
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        cell_types_present = cropped_sdata.tables["table"].obs[cell_type_column].unique().tolist()

        # Subset the palette to only those present
        palette_subset = {
            k: v for k, v in cell_type_colors2.items() if k in cell_types_present
        }
        palette_list = [palette_subset[g] for g in cell_types_present]


        cropped_sdata.pl.render_shapes(
            element_name,
            color=cell_type_column,
            outline=False,
            outline_width=0,
            linewidth=0,
            palette=palette_list,  # subset palette
            groups=cell_types_present,
            ax=ax
        ).pl.show(
            title=title,
            pad_extent=15
        )
        
        ax.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close(fig)
            
    except Exception as e:
        print(f"Error plotting cell types: {e}")



def plot_roi_gene_expression(sdata: sd.SpatialData,
                             roi_info: pd.Series,
                             gene_name: str,
                             element_name: str = "cell_boundaries",
                             figsize: Tuple[int, int] = (15, 15),
                             save_path: Optional[str] = None):
    """
    Plot gene expression in a region of interest.
    """
    print(f"\nPlotting {gene_name} expression for ROI {roi_info['roi_id']}...")
    
    try:
        cropped_sdata = crop_sdata_region(sdata, roi_info)

        if "table" not in cropped_sdata.tables or cropped_sdata.tables["table"].n_obs == 0:
            print(f"Skipping plot for ROI {roi_info['roi_id']}: No cells in cropped region.")
            return
        
        if element_name not in cropped_sdata.shapes:
            print(f"Warning: '{element_name}' not in cropped region")
            return
        
        title = f"{gene_name}"
        
        # Create figure and axis explicitly
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Single render_shapes call with all parameters
        cropped_sdata.pl.render_shapes(
            element_name,
            color=gene_name,
            outline=False,
            outline_width=0,
            cmap="plasma",
            linewidth=0,
            ax=ax
        ).pl.show(
            title=title,
            pad_extent=15
        )
        
        ax.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved to {save_path}")
            plt.close(fig)
            
    except Exception as e:
        print(f"Error plotting gene expression: {e}")


def plot_roi_highlight_celltype(
    sdata: sd.SpatialData,
    roi_info: pd.Series,
    target_cell_type: str,
    cell_type_column: str = "annotation_level2",
    color_palette: Optional[Dict[str, str]] = None,
    element_name: str = "cell_boundaries",
    figsize: Tuple[int, int] = (15, 15),
    save_path: Optional[str] = None,
):
    """
    Plot only the target cell type in its assigned color (others grey).
    
    """
    print(f"\nPlotting {target_cell_type} only for ROI {roi_info['roi_id']}...")
    
    try:
        cropped_sdata = crop_sdata_region(sdata, roi_info)
        
        if "table" not in cropped_sdata.tables or cropped_sdata.tables["table"].n_obs == 0:
            print(f"Skipping plot for ROI {roi_info['roi_id']}: No cells in cropped region.")
            return
        
        if element_name not in cropped_sdata.shapes:
            print(f"Warning: '{element_name}' not in cropped region")
            return
        
        table = cropped_sdata.tables["table"]
        temp_col = "temp_highlight"
        
        # Check if target cell type exists
        cats = table.obs[cell_type_column].cat.categories
        if target_cell_type not in cats:
            print(f"Warning: Target cell type '{target_cell_type}' not found in categories. Skipping plot.")
            return
        
        # ---- Get target color from palette ----
        if color_palette is None:
            color_palette = cell_type_colors2
        
        if target_cell_type in color_palette:
            target_color = color_palette[target_cell_type]
        else:
            print(f"Warning: '{target_cell_type}' not in palette, using fallback color.")
            target_color = "#FF7F0E"
        
        # ---- Create new categorical column ----
        table.obs[temp_col] = (
            table.obs[cell_type_column]
            .apply(lambda x: x if x == target_cell_type else "Other")
            .astype("category")
        )
        
        # ---- Define colors and groups ----
        palette_dict = {target_cell_type: target_color, "Other": "#D3D3D3"}
        cats = table.obs[temp_col].cat.categories
        groups = [g for g in [target_cell_type, "Other"] if g in cats]
        palette_list = [palette_dict[g] for g in groups]
        
        title = f"{target_cell_type} highlighted"
        
        # ---- Plot ----
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cropped_sdata.pl.render_shapes(
            element_name,
            color=temp_col,
            groups=groups,
            palette=palette_list,
            outline=False,
            outline_width=0,
            linewidth=0,
            ax=ax,
        ).pl.show(title=title, pad_extent=15)
        
        ax.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Saved to {save_path}")
        
        plt.close(fig)
        del table.obs[temp_col]
        
    except Exception as e:
        print(f"Error plotting cell type '{target_cell_type}': {e}")



def plot_roi_gene_in_target(sdata: sd.SpatialData,
                          roi_info: pd.Series,
                          gene_name: str,
                          cell_type_column: str = "annotation_level2",
                          target_label: str = "Fibroblasts", 
                          element_name: str = "cell_boundaries",
                          figsize: Tuple[int, int] = (15, 15),
                          save_path: Optional[str] = None,
                          vmin: float = 0,
                          vmax: float = 5):
    """
    Plot gene expression only in a target cell type (rest grey).
    
    Parameters
    ----------
    vmin : float
        Minimum value for color scale (default: 0)
    vmax : float
        Maximum value for color scale (default: 3)
        Values above vmax will be clipped to vmax
    """
    print(f"\nPlotting {gene_name} in {target_label} for ROI {roi_info['roi_id']}...")
    try:
        cropped_sdata = crop_sdata_region(sdata, roi_info)
        if "table" not in cropped_sdata.tables or cropped_sdata.tables["table"].n_obs == 0:
            print(f"Skipping plot for ROI {roi_info['roi_id']}: No cells in cropped region.")
            return
        if element_name not in cropped_sdata.shapes:
            print(f"Warning: '{element_name}' not in cropped region")
            return
        
        table = cropped_sdata.tables["table"]
        temp_col = f"temp_{gene_name}_in_{target_label.replace(' ', '_')}"
        
        # Get gene expression
        if gene_name in table.var_names:
            gene_expr = table[:, gene_name].X.toarray().flatten() if hasattr(table[:, gene_name].X, 'toarray') else table[:, gene_name].X.flatten()
        elif gene_name in table.obs.columns:
            gene_expr = table.obs[gene_name].values
        else:
            print(f"Warning: Gene or Obs '{gene_name}' not found in table")
            return
        
        # Mask expression: set to 0 for non-target cells, add 0.1 to target cells for visibility
        is_target = table.obs[cell_type_column] == target_label
        masked_expr = np.where(is_target, gene_expr + 0.1, 0)
        
        # Clip values to vmax
        masked_expr = np.clip(masked_expr, vmin, vmax)
        
        table.obs[temp_col] = masked_expr
        
        title = f"{gene_name} in \n{target_label}"
        
        # Create a custom colormap: 0 expression (non-target) is grey
        base_cmap = plt.cm.get_cmap('plasma') 
        colors = [(0.8, 0.8, 0.8, 1.0), base_cmap(0.0), base_cmap(0.2), base_cmap(0.4), base_cmap(0.6), base_cmap(0.8), base_cmap(1.0)]
        nodes = [0.0, 0.0001, 0.2, 0.4, 0.6, 0.8, 1.0] 
        new_cmap = LinearSegmentedColormap.from_list("GreyStartPlasma", list(zip(nodes, colors)))
        
        # Create normalization with vmin/vmax
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        
        # Create figure and axis explicitly
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Single render_shapes call with norm parameter
        cropped_sdata.pl.render_shapes(
            element_name,
            color=temp_col,
            cmap=new_cmap,
            norm=norm,  
            outline=False,
            outline_width=0,
            linewidth=0,
            ax=ax
        ).pl.show(
            title=title,
            pad_extent=15
        )
        
        ax.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.close(fig)
        
        # Clean up temporary column
        del table.obs[temp_col]
        
    except Exception as e:
        print(f"Error plotting gene in {target_label}: {e}")