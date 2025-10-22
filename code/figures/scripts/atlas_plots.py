import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend 
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib import gridspec
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
import yaml
from pathlib import Path
import warnings
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import nbformat
from collections import defaultdict
import plotly.graph_objects as go
import json

warnings.filterwarnings('ignore')



class AtlasPlotting:
    """
        Class to create atlas figures with config yml
    """

    def __init__(self, config_path, output_dir="figures"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plotting_params()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif isinstance(config_path, dict):
            return config_path
        else:
            raise ValueError("Config must be a file path or dictionary")

    def _setup_plotting_params(self):
        """Set up matplotlib and scanpy plotting parameters"""
        cfg = self.config['plot_configs']['general']
        plt.rcParams['figure.dpi'] = cfg['dpi']
        plt.rcParams['savefig.dpi'] = cfg['dpi_save']
        plt.rcParams['legend.fontsize'] = cfg['legend_fontsize']
        plt.rcParams['axes.titlesize'] = cfg['title_fontsize']
        plt.rcParams['font.family'] = cfg["font_family"]
        sns.set_theme(style="white",font=cfg["font_family"])
        sc.settings.set_figure_params(
            dpi_save=cfg['dpi_save'],
            fontsize=cfg['legend_fontsize']
        )

    ##### UMAPs #####

    def create_masked_umap(self, adata, mask_column, mask_values=None,
                           color_by='Level_4', figure_name=None):
        """
        Create masked UMAP plots showing only specified cell populations.
        """
        print(f"Creating masked UMAP plots for {mask_column}")

        if figure_name is None:
            raise ValueError("figure_name must be provided")

        # Create subdirectory for this figure
        dir_name = figure_name.replace(" ", "_")
        figure_dir = self.output_dir / dir_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(figure_dir)

        if not pd.api.types.is_categorical_dtype(adata.obs[color_by]):
            adata.obs[color_by] = adata.obs[color_by].astype("category")

        if mask_values is None:
            mask_values = adata.obs[mask_column].unique().tolist()

        color_col_filtered = f"{color_by}_filtered"
        adata.obs[color_col_filtered] = adata.obs[color_by].astype(str)
        adata.obs.loc[~adata.obs[color_by].isin(mask_values), color_col_filtered] = np.nan
        categories = list(mask_values)
        adata.obs[color_col_filtered] = pd.Categorical(
            adata.obs[color_col_filtered],
            categories=categories
        )

        #use palette for mask_values,for rest ##lightblue if no value found
        palette = self.config['palettes'].get(color_by, {})
        new_palette = [palette.get(ct, "skyblue") for ct in mask_values] 
        adata.uns[f"{color_col_filtered}_colors"] = new_palette

        # Generate masked UMAP plots
        sc.pl.umap(
            adata,
            color=color_col_filtered,
            legend_loc=None,
            show=False,
            na_color="white",
            outline_width=(0.1,0.05),
            add_outline=True,
            title=figure_name,
            frameon=False,
            save=f"_{dir_name}_masked_colored_{color_by}.png"
        )

        sc.pl.umap(
            adata,
            color=color_col_filtered,
            show=False,
            na_color="white",
            outline_width=(0.1,0.05),
            add_outline=True,
            na_in_legend=False,
            title=figure_name,
            frameon=False,
            save=f"_{dir_name}_masked_colored_{color_by}_legend.png"
        )

        # Create version without m/a ##lightblue if no value found
        adata_subset = adata[adata.obs[color_by].isin(mask_values)].copy()
        adata_subset.obs[color_by] = adata_subset.obs[color_by].cat.remove_unused_categories()

        subset_palette = [palette.get(ct, "skyblue") for ct in adata_subset.obs[color_by].cat.categories]
        adata_subset.uns[f"{color_by}_colors"] = subset_palette

        sc.pl.umap(
            adata_subset,
            color=color_by,
            show=False,
            na_color="white",
            add_outline=True,
            outline_width=(0.1,0.05),
            na_in_legend=False,
            title=figure_name,
            frameon=False,
            save=f"_{dir_name}_subset_only_{color_by}_legend.png"
        )


    def create_masked_umap_highlight(self, adata, mask_column, mask_values=None,
                                color_by='Level_4', figure_name=None,
                                highlight_size=1.5, background_size=0.25,ordered=True):

        print(f"Creating masked UMAP plots for {mask_column}")

        dir_name = figure_name.replace(" ", "_")
        figure_dir = self.output_dir / dir_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(figure_dir)
        
        if not pd.api.types.is_categorical_dtype(adata.obs[color_by]):
            adata.obs[color_by] = adata.obs[color_by].astype("category")
        
        if mask_values is None:
            mask_values = adata.obs[mask_column].unique().tolist()
        
        color_col_filtered = f"{color_by}_filtered"
        adata.obs[color_col_filtered] = adata.obs[color_by].astype(str)
        adata.obs.loc[~adata.obs[color_by].isin(mask_values), color_col_filtered] = np.nan
        
        size_col = f"{color_col_filtered}_sizes"
        adata.obs[size_col] = background_size  # default small size
        adata.obs.loc[~pd.isna(adata.obs[color_col_filtered]), size_col] = highlight_size
        mask_value_counts = adata.obs[color_by].value_counts()
        mask_values_ordered = sorted(mask_values, key=lambda x: mask_value_counts.get(x, 0),reverse=True)
        categories = mask_values_ordered
  
        adata.obs[color_col_filtered] = pd.Categorical(
            adata.obs[color_col_filtered],
            categories=categories
        )
        
        palette = self.config['palettes'].get(color_by, {})
        new_palette = [palette.get(ct, "skyblue") for ct in mask_values_ordered] #mask_values
        adata.uns[f"{color_col_filtered}_colors"] = new_palette
        
        if ordered:
            sort_idx = adata.obs[color_col_filtered].cat.codes.argsort(kind='stable')
            adata = adata[sort_idx, :]
        
        
        sc.pl.umap(
            adata,
            color=color_col_filtered,
            size=adata.obs[size_col],
            legend_loc=None,
            na_color="white",
            outline_width=(0.1,0.05),
            add_outline=True,
            sort_order=False,
            show=False,
            title=figure_name,
            frameon=False,
            save=f"_{dir_name}_masked_colored_{color_by}_{str(highlight_size)}_highlight.png"
        )
        
        sc.pl.umap(
            adata,
            color=color_col_filtered,
            size=adata.obs[size_col],
            show=False,
            na_color="white",
            outline_width=(0.1,0.05),
            add_outline=True,
            sort_order=False,
            na_in_legend=False,
            title=figure_name,
            frameon=False,
            save=f"_{dir_name}_masked_colored_{color_by}_{str(highlight_size)}_legend_highlight.png"
        )
        
##### Compositional plots #####


    def cluster_samples(self,ct_props):
        dist_matrix = pdist(ct_props.values, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        return ct_props.index[leaves_list(linkage_matrix)]

    def cluster_grouped_samples(self,ct_props, adata, sample_column, order_by_column, order_ascending):
        group_labels = adata.obs.groupby(sample_column)[order_by_column].first().sort_values(ascending=order_ascending)
        grouped_samples = []
        for group_value in group_labels.unique():
            group_sample_ids = group_labels[group_labels == group_value].index
            group_props = ct_props.loc[group_sample_ids]
            if len(group_props) > 1:
                d = pdist(group_props.values, metric='euclidean')
                link = linkage(d, method='ward')
                cluster_idx = leaves_list(link)
                ordered_samples = group_props.index[cluster_idx]
            else:
                ordered_samples = group_props.index
            grouped_samples.extend(ordered_samples)
        return grouped_samples

    def plot_barplot(self,ct_props, color_dict, title, ylabel, figsize, sample_column, order_values=None, order_color_dict=None, order_by_column=None,xlabel=None):
        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(ct_props))
        
        # Add spacing between bars
        bar_width = 1  # Reduce from default 1.0 to create gaps for daniele
        x_positions = np.arange(len(ct_props))
        
        for cell_type in ct_props.columns:
            values = ct_props[cell_type].values
            clean_label = cell_type.replace("Malignant Cell - ", "")
            ax.bar(x_positions, values, bottom=bottom, width=bar_width,
                color=color_dict.get(cell_type, 'gray'), label=clean_label,
                edgecolor='white', linewidth=1.0)
            bottom += values
        
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        else:
            ax.set_xlabel(sample_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(title, fontsize=self.config["plot_configs"]["general"]["title_fontsize"], fontweight=self.config["plot_configs"]["general"]["title_fontweight"], pad=20)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_xlim(-0.5, len(ct_props) - 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_axisbelow(True)

        # Color bar below heatmap (adjust positioning for new bar positions)
        if order_values and order_color_dict:
            bar_height = 0.025
            y_offset = -bar_height - 0.01 
            for i, val in enumerate(order_values):
                ax.add_patch(plt.Rectangle(
                    (i - bar_width/2, y_offset),  # Adjust x position for bar width
                    bar_width,                    # Use same width as bars
                    bar_height,      
                    linewidth=0,
                    edgecolor=order_color_dict[val],
                    facecolor=order_color_dict[val],
                    transform=ax.transData,
                    clip_on=False
                ))

        # Add legends to right of plot
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[key]) for key in ct_props.columns]
        labels = [key.replace("Malignant Cell - ", "") for key in ct_props.columns]
        legend1 = ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left',
                            title="Cell Types",title_fontproperties=font_manager.FontProperties(weight='bold'),
                            fontsize=self.config["plot_configs"]["general"]["legend_fontsize"],
                            frameon=False)
        ax.add_artist(legend1)

        if order_values and order_color_dict and order_by_column:
            
            order_handles = [plt.Rectangle((0, 0), 1, 1, color=order_color_dict[val]) for val in order_color_dict]
            order_labels = [str(val) for val in order_color_dict]
            legend2 = ax.legend(order_handles, order_labels, bbox_to_anchor=(1.01, 0),
                                loc='lower left', title=order_by_column.replace('_', ' ').title(),
                                frameon=False)
            legend2.get_title().set_fontweight('bold')

        plt.subplots_adjust(right=0.95)
        return fig
    def create_all_stacked_barplots(self, adata, level_column, sample_column="Sample_ID", 
                                    subset_level=None, subset_value=None,
                                    order_by_column=None, order_ascending=True,
                                    figsize=(16, 8), save_name_prefix="composition",xlabel=None):
        if sample_column is None:
            for col in adata.obs.columns:
                if adata.obs[col].dtype in ['object', 'category']:
                    sample_column = col
                    break
            else:
                raise ValueError("No suitable sample column found.")

        if subset_level and subset_value:
            if isinstance(subset_value, str):
                subset_value = [subset_value]
            adata = adata[adata.obs[subset_level].isin(subset_value), :]

        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError("Required columns not found.")

        ct_data = pd.crosstab(adata.obs[sample_column], adata.obs[level_column])
        ct_props = ct_data.div(ct_data.sum(axis=1), axis=0)

        color_dict = self.config["palettes"][level_column]
        ylabel = f"Cell Type Proportion ({subset_value[0]}s)" if subset_value else "Cell Type Proportion"
        title_base = f"Cell Type Composition by {sample_column.replace('_', ' ').title()}"

        figs = {}
        order_color_dict = None
        order_values_basic = None

        if order_by_column:
            order_values = [adata.obs[adata.obs[sample_column] == s][order_by_column].iloc[0] for s in ct_props.index]
            unique_vals = list(dict.fromkeys(order_values))  # Preserves order
            try:
                order_color_dict = self.config["palettes"][order_by_column]
            except KeyError:
                raise ValueError(f"No palette found in self.config['palettes'] for order_by_column '{order_by_column}'")

            order_bar_colors = [order_color_dict[val] for val in order_values]
        else:
            order_bar_colors = None

        # Basic
        figs["basic"] = self.plot_barplot(ct_props, color_dict, f"{title_base}", ylabel, figsize, sample_column, order_values_basic, order_color_dict, order_by_column,xlabel)

        # Clustered
        clustered_order = self.cluster_samples(ct_props)
        order_values_clustered = [adata.obs[adata.obs[sample_column] == s][order_by_column].iloc[0] for s in ct_props.loc[clustered_order].index] if order_color_dict else None
        figs["clustered"] = self.plot_barplot(ct_props.loc[clustered_order], color_dict, f"{title_base} (Clustered)", ylabel, figsize, sample_column, order_values_clustered, order_color_dict, order_by_column,xlabel)

        # Grouped Clustered
        if order_by_column and order_by_column in adata.obs.columns:
            grouped_order = self.cluster_grouped_samples(ct_props, adata, sample_column, order_by_column, order_ascending)
            order_values_grouped = [adata.obs[adata.obs[sample_column] == s][order_by_column].iloc[0] for s in ct_props.loc[grouped_order].index]
            figs["clustered_grouped"] = self.plot_barplot(ct_props.loc[grouped_order], color_dict, f"{title_base} (Clustered per Group)", ylabel, figsize, sample_column, order_values_grouped, order_color_dict, order_by_column,xlabel)

        # Save plots
        save_dir = self.output_dir / "compositional_plot"
        save_dir.mkdir(parents=True, exist_ok=True)
        for key, fig in figs.items():
            ax = fig.axes[0]
            legends = [artist for artist in ax.get_children() if isinstance(artist, Legend)]
            fig.savefig(save_dir / f"{save_name_prefix}_{key}.png",
                        dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight',
                        bbox_extra_artists=legends,
                        facecolor='white')

        return figs



    def composition_heatmap(self, adata, level_column, sample_column="Sample_ID", 
                        subset_level=None, subset_value=None, 
                        order_by_column=None, figsize=(16, 6), 
                        title=None, save_name=None, **kwargs):


        if sample_column is None:
            if 'sample' in adata.obs.columns:
                sample_column = 'sample'
            else:
                for col in adata.obs.columns:
                    if adata.obs[col].dtype in ['object', 'category']:
                        sample_column = col
                        break
                else:
                    raise ValueError("No suitable sample column found. Please specify sample_column.")

        # Subset
        if subset_level is not None and subset_value is not None:
            if subset_level not in adata.obs.columns:
                raise ValueError(f"Subset level '{subset_level}' not found in adata.obs")
            if isinstance(subset_value, str):
                subset_value = [subset_value]
            mask = adata.obs[subset_level].isin(subset_value)
            adata = adata[mask, :]
            print(f"Subset to {mask.sum()} cells with {subset_level} in {subset_value}")

        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError("Required columns not found in adata.obs")
        if order_by_column and order_by_column not in adata.obs.columns:
            raise ValueError(f"Order column '{order_by_column}' not found in adata.obs")

        # Compute proportions
        ct_data = pd.crosstab(adata.obs[sample_column], adata.obs[level_column])
        ct_props = ct_data.div(ct_data.sum(axis=1), axis=0)

        # Cluster samples using Euclidean distance
        dist_matrix = pdist(ct_props.values, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        ordered_indices = leaves_list(linkage_matrix)
        ct_props = ct_props.iloc[ordered_indices]

        if order_by_column:
            order_values = [adata.obs[adata.obs[sample_column] == s][order_by_column].iloc[0] for s in ct_props.index]
            unique_vals = list(dict.fromkeys(order_values))  # Preserves order
            try:
                order_color_dict = self.config["palettes"][order_by_column]
            except KeyError:
                raise ValueError(f"No palette found in self.config['palettes'] for order_by_column '{order_by_column}'")

            order_bar_colors = [order_color_dict[val] for val in order_values]
        else:
            order_bar_colors = None


        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(ct_props.T, annot=False, cmap=self.config["plot_configs"]["continuous_plots"]["cmap"], ax=ax, cbar=True, **kwargs, 
            linewidths=0, linecolor='white', 
            xticklabels=False, yticklabels=True, 
            square=False)

        ax.set_aspect('auto') 


        ax.set_title(title or 'Cell Type Composition (Clustered)', fontsize=self.config["plot_configs"]["general"]["title_fontsize"], fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax.set_xlabel('')
        ax.set_ylabel(f"Cell Types ({subset_value[0]}s)", fontsize=12)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels(ct_props.columns, rotation=0)

        # Color bar below heatmap
        if order_by_column:
            bar_height = 0.5
            for idx, color in enumerate(order_bar_colors):
                ax.add_patch(plt.Rectangle((idx, ct_props.shape[1] + 0.05), 1, bar_height,
                                            linewidth=0,edgecolor=color, 
                                             color=color, transform=ax.transData, clip_on=False))

            # Add legend below plot
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=order_color_dict[val]) for val in unique_vals]
            legend_labels = [str(val) for val in unique_vals]
            legend = fig.legend(legend_handles, legend_labels,
                    title=order_by_column.replace("_", " ").title(),
                    loc='lower center', bbox_to_anchor=(0.5, -0.12),
                    ncol=len(unique_vals), fontsize=self.config["plot_configs"]["general"]["legend_fontsize"], title_fontsize=self.config["plot_configs"]["general"]["legend_fontsize"], frameon=False)
            legend.get_title().set_fontweight('bold')

        # Adjust layout for legend space
        plt.tight_layout(rect=[0, 0.01, 1, 1])

        # Save
        if save_name is not None:
            save_dir = self.output_dir / "compositional_plot"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        return fig
    
    ##### Dataset overview plots #####

    def sample_and_cell_counts_barplot(self, adata, level_column, sample_column="Sample_ID",
                                figsize=(10, 5), title=None, save_name=None, custom_palette=False,xlabel=None):

        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError(f"Columns '{level_column}' and/or '{sample_column}' not found in adata.obs.")

        # Counts
        sample_counts = adata.obs.groupby(level_column)[sample_column].nunique()
        cell_counts = adata.obs[level_column].value_counts()

        # Sort by number of samples
        sorted_categories = sample_counts.sort_values(ascending=False).index.tolist()
        sample_counts = sample_counts.reindex(sorted_categories)
        cell_counts = cell_counts.reindex(sorted_categories)

        # Palette
        if custom_palette:
            try:
                color_dict = self.config["palettes"][level_column]
                colors = [color_dict[cat] for cat in sorted_categories]
            except KeyError:
                raise ValueError(f"Custom palette not found for level '{level_column}' in config['palettes']")
        else:
            palette = sns.color_palette("pastel", len(sorted_categories))
            color_dict = dict(zip(sorted_categories, palette))
            colors = [color_dict[cat] for cat in sorted_categories]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
        axs = []

    
        ax = fig.add_subplot(gs[:, 0])
        ax.bar(sorted_categories, sample_counts.values, color=colors)
        ax.set_title("Samples per Category",
                    fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                    fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax.set_ylabel("Number of Samples", fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12,labelpad=10)
        else:
            ax.set_xlabel(level_column.replace("_", " ").title(), fontsize=12,labelpad=10)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax.spines.values()]
        ax.grid(False)

       
        ax = fig.add_subplot(gs[:, 1])
        ax.bar(sorted_categories, cell_counts.values, color=colors)
        ax.set_title("Cells per Category",
                    fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                    fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax.set_ylabel("Number of Cells", fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12,labelpad=10)
        else:
            ax.set_xlabel(level_column.replace("_", " ").title(), fontsize=12,labelpad=10)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax.spines.values()]
        ax.grid(False)


        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sample_cell_counts"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        fig_grey = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
        axs = []

    
        ax = fig_grey.add_subplot(gs[:, 0])
        ax.bar(sorted_categories, sample_counts.values, color="grey")
        ax.set_title("Samples per Category",
                    fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                    fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax.set_ylabel("Number of Samples", fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12,labelpad=10)
        else:
            ax.set_xlabel(level_column.replace("_", " ").title(), fontsize=12,labelpad=10)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax.spines.values()]
        ax.grid(False)

       
        ax = fig_grey.add_subplot(gs[:, 1])
        ax.bar(sorted_categories, cell_counts.values, color="grey")
        ax.set_title("Cells per Category",
                    fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                    fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax.set_ylabel("Number of Cells", fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12,labelpad=10)
        else:
            ax.set_xlabel(level_column.replace("_", " ").title(), fontsize=12,labelpad=10)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax.spines.values()]
        ax.grid(False)


        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sample_cell_counts"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_all_grey.png"
            fig_grey.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        return fig , fig_grey
    
    def sample_and_cell_counts_barplot_break_axis(self, adata, level_column, sample_column="Sample_ID",
                                              figsize=(5, 5), title=None, save_name=None, 
                                              custom_palette=False, xlabel=None,
                                              break_point=None, break_ratio=0.3, break_gap=0.02,
                                              plot_type="samples"):

        
        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError(f"Columns '{level_column}' and/or '{sample_column}' not found in adata.obs.")
        
        if plot_type not in ["samples", "cells"]:
            raise ValueError("plot_type must be either 'samples' or 'cells'")
        
        # Calculate counts based on plot type
        if plot_type == "samples":
            counts = adata.obs.groupby(level_column)[sample_column].nunique()
            ylabel = "Number of Samples"
            plot_title = "Samples per Category"
        else:  # cells
            counts = adata.obs[level_column].value_counts()
            ylabel = "Number of Cells"
            plot_title = "Cells per Category"
        
        # Sort by count values
        sorted_categories = counts.sort_values(ascending=False).index.tolist()
        counts = counts.reindex(sorted_categories)
        
        # Auto-calculate break point if not provided
        if break_point is None:
            break_point = np.percentile(counts.values, 75)
        
        # Palette
        if custom_palette:
            try:
                color_dict = self.config["palettes"][level_column]
                colors = [color_dict[cat] for cat in sorted_categories]
            except KeyError:
                raise ValueError(f"Custom palette not found for level '{level_column}' in config['palettes']")
        else:
            palette = sns.color_palette("pastel", len(sorted_categories))
            color_dict = dict(zip(sorted_categories, palette))
            colors = [color_dict[cat] for cat in sorted_categories]
        
        # Create figure with broken axis layout
        fig = plt.figure(figsize=figsize)
        
        # Grid: 2 rows for upper and lower sections
        gs = gridspec.GridSpec(2, 1, height_ratios=[break_ratio, 1-break_ratio], 
                            hspace=break_gap)
        
        # Create upper and lower subplots
        ax_upper = fig.add_subplot(gs[0])
        ax_lower = fig.add_subplot(gs[1], sharex=ax_upper)
        
        # Plot the same data on both axes
        ax_upper.bar(sorted_categories, counts.values, color=colors)
        ax_lower.bar(sorted_categories, counts.values, color=colors)
        
        # Set y-limits
        max_val = np.max(counts.values)
        ax_upper.set_ylim(break_point, max_val * 1.15)
        ax_lower.set_ylim(0, break_point * 0.9)
        
        # Hide connecting spines
        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
        
        # Remove x-axis ticks from upper plot
        ax_upper.tick_params(bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_lower.xaxis.tick_bottom()
        
        # Add break lines
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
        ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)
        
        # Styling
        ax_upper.set_title(plot_title if title is None else title,
                        fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                        fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax_lower.set_ylabel(ylabel, fontsize=12)
        
        # X-label
        if xlabel:
            ax_lower.set_xlabel(xlabel, fontsize=12, labelpad=10)
        else:
            ax_lower.set_xlabel(level_column.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        # Apply styling to both axes
        for ax in [ax_upper, ax_lower]:
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='both', width=1.8)
            [sp.set_linewidth(1.8) for sp in ax.spines.values()]
            ax.grid(False)
        
        plt.tight_layout()
        
        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sample_cell_counts"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_{plot_type}_break_axis.png"
            fig.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                    bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        # Create figure with broken axis layout
        fig_grey = plt.figure(figsize=figsize)
        
        # Grid: 2 rows for upper and lower sections
        gs = gridspec.GridSpec(2, 1, height_ratios=[break_ratio, 1-break_ratio], 
                            hspace=break_gap)
        
        # Create upper and lower subplots
        ax_upper = fig_grey.add_subplot(gs[0])
        ax_lower = fig_grey.add_subplot(gs[1], sharex=ax_upper)
        
        # Plot the same data on both axes
        ax_upper.bar(sorted_categories, counts.values, color="grey")
        ax_lower.bar(sorted_categories, counts.values, color="grey")
        
        # Set y-limits
        max_val = np.max(counts.values)
        ax_upper.set_ylim(break_point, max_val * 1.15)
        ax_lower.set_ylim(0, break_point * 0.9)
        
        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
        
        # Remove x-axis ticks from upper plot
        ax_upper.tick_params(bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_lower.xaxis.tick_bottom()
        
        # Add break lines
        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **kwargs)
        ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **kwargs)
        
        # Styling
        ax_upper.set_title(plot_title if title is None else title,
                        fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                        fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        ax_lower.set_ylabel(ylabel, fontsize=12)
        
        # X-label
        if xlabel:
            ax_lower.set_xlabel(xlabel, fontsize=12, labelpad=10)
        else:
            ax_lower.set_xlabel(level_column.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        # Apply styling to both axes
        for ax in [ax_upper, ax_lower]:
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='both', width=1.8)
            [sp.set_linewidth(1.8) for sp in ax.spines.values()]
            ax.grid(False)
        
        plt.tight_layout()
        
        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sample_cell_counts"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_{plot_type}_break_axis_grey.png"
            fig_grey.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                    bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        
    
    def matrix_markers(self, adata, groupby_column, n_markers=5, method="wilcoxon",
                    standard_scale="var", figsize=(12, 8), title=None, save_name=None,
                     **kwargs):
        
        if groupby_column not in adata.obs.columns:
            raise ValueError(f"Groupby column '{groupby_column}' not found in adata.obs")
        
        print(f"Computing marker genes for '{groupby_column}' using {method} method...")
        
        # Get config values
        general_config = self.config["plot_configs"]["general"]
        rank_genes_config = self.config["plot_configs"]["rank_genes_plots"]
        
        # Use layer from config if not specified in kwargs
        layer = kwargs.pop('layer', rank_genes_config.get("layer", "log_norm"))
        
        # Compute dendrogram for hierarchical clustering of groups
        sc.tl.dendrogram(adata, groupby=groupby_column)
        
        # Compute marker genes with config parameters
        sc.tl.rank_genes_groups(
            adata, 
            groupby=groupby_column, 
            method=method,
            layer=layer,
            min_logfoldchange=rank_genes_config.get("min_logfoldchange", 2)
        )
        
        # Generate dotplot
        print("Generating dotplot...")
        fig_dot = plt.figure(figsize=figsize, dpi=general_config["dpi"])
        
        # Set font family for the plot
        plt.rcParams['font.family'] = general_config["font_family"]
        
        sc.pl.rank_genes_groups_dotplot(
            adata,
            groupby=groupby_column,
            standard_scale=standard_scale,
            n_genes=n_markers,
            use_raw=False,
            layer=layer,
            ax=fig_dot.gca(),
            var_group_rotation=rank_genes_config.get("var_group_rotation", 90),
            values_to_plot=rank_genes_config.get("values_to_plot", "logfoldchanges"),
            vmin=rank_genes_config.get("vmin", -5),
            vmax=rank_genes_config.get("vmax", 5),
            cmap=rank_genes_config.get("cmap", "bwr"),
            **kwargs
        )
        
        # Apply styling from config
        ax_dot = fig_dot.gca()
        if title:
            ax_dot.set_title(f"{title} - Dotplot",
                            fontsize=general_config["title_fontsize"],
                            fontweight=general_config["title_fontweight"])
        else:
            ax_dot.set_title(f"Top {n_markers} Marker Genes - Dotplot",
                            fontsize=general_config["title_fontsize"],
                            fontweight=general_config["title_fontweight"])
        
        # Update legend styling
        legend = ax_dot.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(general_config["legend_fontsize"])
                text.set_fontweight(general_config["legend_fontweight"])
        
        plt.tight_layout()
        
        # Save dotplot
        if save_name is not None:
            save_dir = self.output_dir / "marker_analysis"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_dotplot.png"
            fig_dot.savefig(save_path, 
                        dpi=general_config['dpi_save'],
                        bbox_inches='tight', 
                        facecolor='white')
            print(f"Dotplot saved to: {save_path}")
        
        # Generate matrixplot
        print("Generating matrixplot...")
        fig_matrix = plt.figure(figsize=figsize, dpi=general_config["dpi"])
        
        sc.pl.rank_genes_groups_matrixplot(
            adata,
            groupby=groupby_column,
            standard_scale=standard_scale,
            n_genes=n_markers,
            use_raw=False,
            layer=layer,
            ax=fig_matrix.gca(),
            var_group_rotation=rank_genes_config.get("var_group_rotation", 90),
            values_to_plot=rank_genes_config.get("values_to_plot", "logfoldchanges"),
            vmin=rank_genes_config.get("vmin", -5),
            vmax=rank_genes_config.get("vmax", 5),
            cmap=rank_genes_config.get("cmap", "bwr"),
            **kwargs
        )
        
        plt.tight_layout()
        
        # Save matrixplot
        if save_name is not None:
            save_path = save_dir / f"{save_name}_matrixplot.png"
            fig_matrix.savefig(save_path, 
                            dpi=general_config['dpi_save'],
                            bbox_inches='tight', 
                            facecolor='white')
            print(f"Matrixplot saved to: {save_path}")
        
        return fig_dot, fig_matrix

    #Matrix plot but using markers from json file
    def matrix_annotation_markers(self, adata, json_file, json_name, figsize=(12, 8),save_name=None):
        markers_flat = [
            marker
            for key in sorted(json_file[json_name]["markers"].keys())
            for marker in json_file[json_name]["markers"][key]
        ]
        celltypes_list = sorted(json_file[json_name]["markers"].keys())
        fig = plt.figure(figsize=figsize, dpi=self.config["plot_configs"]["general"]["dpi"])

        sc.pl.matrixplot(adata[adata.obs["Level_4"].isin(celltypes_list)],
                 standard_scale="var",layer=self.config["plot_configs"]["continuous_plots"]["layer"],var_names=markers_flat,
                 groupby="Level_4",cmap="coolwarm",ax=fig.gca())
        plt.tight_layout()

        if save_name is not None:
            save_dir = self.output_dir / "marker_analysis"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_matrixplot_json_markers.png"
            fig.savefig(save_path, 
                            dpi=self.config["plot_configs"]["general"]['dpi_save'],
                            bbox_inches='tight', 
                            facecolor='white')
            print(f"Matrixplot saved to: {save_path}")




    ## Sankey plot ##

    def create_hierarchical_cell_list(self, adata, levels=[]):
        if not levels:
            raise ValueError("You must specify at least one level.")

        obs_df = adata.obs
        #hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        if len(levels) == 2:
            hierarchy = defaultdict(lambda: defaultdict(int))
        elif len(levels) == 3:
            hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for _, row in obs_df.iterrows():
            if len(levels) == 1:
                level1 = row[levels[0]]
                hierarchy[level1] += 1

            elif len(levels) == 2:
                level1 = row[levels[0]]
                level2 = row[levels[1]]
                hierarchy[level1][level2] += 1

            elif len(levels) == 3:
                level1 = row[levels[0]]
                level2 = row[levels[1]]
                level3 = row[levels[2]]
                hierarchy[level1][level2][level3] += 1

            else:
                raise ValueError("Only 1 to 3 levels supported.")

        result_list = []

        # Format results based on levels
        if len(levels) == 1:
            for level1, count in hierarchy.items():
                result_list.append(f"{level1} [{count}]")

        elif len(levels) == 2:
            for level1, level2_dict in hierarchy.items():
                level1_count = sum(level2_dict.values())
                result_list.append(f"{level1} [{level1_count}]")
                for level2, count in level2_dict.items():
                    result_list.append(f"  {level2}  [{count}]")

        elif len(levels) == 3:
            for level1, level2_dict in hierarchy.items():
                level1_count = sum(sum(level3_dict.values()) for level3_dict in level2_dict.values())
                result_list.append(f"{level1} [{level1_count}]")
                for level2, level3_dict in level2_dict.items():
                    level2_count = sum(level3_dict.values())
                    result_list.append(f"  {level2}  [{level2_count}]")
                    for level3, count in level3_dict.items():
                        result_list.append(f"    {level3}   [{count}]")

        return result_list

    def create_hierarchical_cell_string(self, adata, levels):
        hierarchical_list = self.create_hierarchical_cell_list(adata, levels=levels)
        return '\n'.join(hierarchical_list)
    
    def plot_sankey(self, adata,levels=[],save_name=None,width=1100,height=800):

        hierarchy_string = self.create_hierarchical_cell_string(adata,levels)

        labels = []
        sources = []
        targets = []
        values = []

        label_map = {}
        def get_label_index(label):
            if label not in label_map:
                label_map[label] = len(labels)
                labels.append(label)
            return label_map[label]

        sankey_text = hierarchy_string
        # Parse lines
        parent_stack = []
        for line in sankey_text.strip().split('\n'):
            indent = len(line) - len(line.lstrip())
            name, count = line.strip().rsplit(' [', 1)
            value = int(count[:-1])
            level = indent // 2

            parent_stack = parent_stack[:level]
            parent_stack.append(name)

            if level > 0:
                source = get_label_index(parent_stack[level - 1])
                target = get_label_index(parent_stack[level])
                sources.append(source)
                targets.append(target)
                values.append(value)
            else:
                get_label_index(name)

        # Plot Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20,color="grey"), #label=labels
            link=dict(source=sources, target=targets, value=values,color="lightgrey")
        )])


        step = 1.0 / len(labels)
        y_positions = [i * step for i in range(len(labels))]


        fig.update_layout(
            width=(width/3),
            height=height,
            font_size=12,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.data[0].arrangement = "snap"

        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sankey_plot"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}.png"
            fig.write_image(save_path,
                width=(width/3), 
                height=height, 
                scale=2,
                )
            print(f"Figure saved to: {save_path}")

        # Plot Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20,color="grey", label=labels),
            link=dict(source=sources, target=targets, value=values,color="lightgrey")
        )])


        step = 1.0 / len(labels)
        y_positions = [i * step for i in range(len(labels))]


        fig.update_layout(
            width=width,
            height=height,
            font_size=12,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.data[0].arrangement = "snap"

        # Save
        if save_name is not None:
            save_dir = self.output_dir / "sankey_plot"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_labels.png"
            fig.write_image(save_path,
                width=width, 
                height=height, 
                scale=2,
                )
            print(f"Figure saved to: {save_path}")

    

    def cells_per_patient_boxplot(self, adata, level_column, sample_column="Sample_ID",
                              figsize=(10, 6), title=None, save_name=None, custom_palette=False, xlabel=None):
        
        
        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError(f"Columns '{level_column}' and/or '{sample_column}' not found in adata.obs.")
        
        # Calculate total cells per patient

        cells_per_patient = adata.obs.groupby(sample_column).agg({
            level_column: 'first',  
            sample_column: 'size'   
        }).rename(columns={sample_column: 'cell_count'})
        
        # Get category order by average number of cells per patient (largest first)
        avg_cells_per_category = cells_per_patient.groupby(level_column)['cell_count'].mean()
        sorted_categories = avg_cells_per_category.sort_values(ascending=False).index.tolist()
        
        # Print some debug info
        print(f"Total patients: {len(cells_per_patient)}")
        print(f"Categories ordered by average cells per patient:")
        for cat in sorted_categories:
            cat_data = cells_per_patient[cells_per_patient[level_column] == cat]['cell_count']
            print(f"{cat}: {len(cat_data)} patients, avg cells: {cat_data.mean():.1f}, range: {cat_data.min()}-{cat_data.max()}")
        
        # Palette
        if custom_palette:
            try:
                color_dict = self.config["palettes"][level_column]
                colors = [color_dict[cat] for cat in sorted_categories]
            except KeyError:
                raise ValueError(f"Custom palette not found for level '{level_column}' in config['palettes']")
        else:
            palette = sns.color_palette("pastel", len(sorted_categories))
            color_dict = dict(zip(sorted_categories, palette))
            colors = [color_dict[cat] for cat in sorted_categories]
        
        # Create colored version
        fig, ax = plt.subplots(figsize=figsize)
        
        data_for_boxplot = []
        labels = []
        for category in sorted_categories:
            category_data = cells_per_patient[cells_per_patient[level_column] == category]['cell_count']
            data_for_boxplot.append(category_data.values)
            labels.append(category)
        
        # Create boxplot
        bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True, 
                        boxprops=dict(linewidth=1.8),
                        whiskerprops=dict(linewidth=1.8),
                        capprops=dict(linewidth=1.8),
                        medianprops=dict(linewidth=1.8, color='black'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        if title:
            ax.set_title(title,
                        fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                        fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        else:
            ax.set_title("Cells per Patient by Category",
                        fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                        fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        
        ax.set_ylabel("Number of Cells per Patient", fontsize=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        else:
            ax.set_xlabel(level_column.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax.spines.values()]
        ax.grid(False)
        
        plt.tight_layout()
        
        # Save colored version
        if save_name is not None:
            save_dir = self.output_dir / "cells_per_patient_boxplot"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        # Create grey version
        fig_grey, ax_grey = plt.subplots(figsize=figsize)
        

        bp_grey = ax_grey.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                                boxprops=dict(linewidth=1.8),
                                whiskerprops=dict(linewidth=1.8),
                                capprops=dict(linewidth=1.8),
                                medianprops=dict(linewidth=1.8, color='black'))
        
        # Color the boxes grey
        for patch in bp_grey['boxes']:
            patch.set_facecolor('grey')
            patch.set_alpha(0.7)
        
        # Styling for grey version
        if title:
            ax_grey.set_title(title,
                            fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                            fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        else:
            ax_grey.set_title("Cells per Patient by Category",
                            fontsize=self.config["plot_configs"]["general"]["title_fontsize"],
                            fontweight=self.config["plot_configs"]["general"]["title_fontweight"])
        
        ax_grey.set_ylabel("Number of Cells per Patient", fontsize=12)
        if xlabel:
            ax_grey.set_xlabel(xlabel, fontsize=12, labelpad=10)
        else:
            ax_grey.set_xlabel(level_column.replace("_", " ").title(), fontsize=12, labelpad=10)
        
        ax_grey.tick_params(axis='x', rotation=45)
        ax_grey.tick_params(axis='both', width=1.8)
        [sp.set_linewidth(1.8) for sp in ax_grey.spines.values()]
        ax_grey.grid(False)
        
        plt.tight_layout()
        
        # Save grey version
        if save_name is not None:
            save_dir = self.output_dir / "cells_per_patient_boxplot"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_all_grey.png"
            fig_grey.savefig(save_path, dpi=self.config['plot_configs']['general']['dpi_save'],
                            bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        return fig, fig_grey
  
    
    def stacked_violin_annotation_markers(self, adata, json_file, json_name, figsize=(12, 8),save_name=None):
        markers_flat = [
            marker
            for key in sorted(json_file[json_name]["markers"].keys())
            for marker in json_file[json_name]["markers"][key]
        ]
        celltypes_list = sorted(json_file[json_name]["markers"].keys())
        fig = plt.figure(figsize=figsize, dpi=self.config["plot_configs"]["general"]["dpi"])

        
        sc.pl.stacked_violin(adata[adata.obs["Level_4"].isin(celltypes_list)],
                 standard_scale="var",layer=self.config["plot_configs"]["continuous_plots"]["layer"],var_names=markers_flat,
                 groupby="Level_4",cmap="coolwarm",ax=fig.gca())
        plt.tight_layout()

        if save_name is not None:
            save_dir = self.output_dir / "marker_analysis"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{save_name}_stacked_violin_json_markers.png"
            fig.savefig(save_path, 
                            dpi=self.config["plot_configs"]["general"]['dpi_save'],
                            bbox_inches='tight', 
                            facecolor='white')
            print(f"Stacked violin plot saved to: {save_path}")



    def plot_barplot_multi_meta(self, ct_props, color_dict, title, ylabel, figsize, sample_column, 
                                meta_annotations_list, meta_color_dicts_list, meta_columns_list, xlabel=None):

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(ct_props))
        
        # Add spacing between bars
        bar_width = 1  
        x_positions = np.arange(len(ct_props))
        
        # Stacked Bar Plot 
        for cell_type in ct_props.columns:
            values = ct_props[cell_type].values
            clean_label = cell_type.replace("Malignant Cell - ", "")
            ax.bar(x_positions, values, bottom=bottom, width=bar_width,
                color=color_dict.get(cell_type, 'gray'), label=clean_label,
                edgecolor='white', linewidth=1.0)
            bottom += values
        
        ax.set_xlabel("", fontsize=12)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_title(title, fontsize=self.config["plot_configs"]["general"]["title_fontsize"], 
                    fontweight=self.config["plot_configs"]["general"]["title_fontweight"], pad=20)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_xlim(-0.5, len(ct_props) - 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_axisbelow(True)

        #  Multiple Color Bars below plot 
        if meta_annotations_list and meta_color_dicts_list and meta_columns_list:
            bar_height = 0.025
            # Calculate y_offset dynamically based on number of bars
            y_offset_base = -bar_height - 0.01 
            
            for idx, (annotations, color_dict_meta, column_name) in enumerate(zip(meta_annotations_list, meta_color_dicts_list, meta_columns_list)):
                # Offset each bar
                y_offset = y_offset_base - idx * (bar_height + 0.005) 
                
                for i, val in enumerate(annotations):
                    color = color_dict_meta.get(val, 'lightgray') 
                    ax.add_patch(patches.Rectangle(
                        (i - bar_width/2, y_offset),  # Adjust x position for bar width
                        bar_width,                    # Use same width as bars
                        bar_height,      
                        linewidth=0,
                        edgecolor=color,
                        facecolor=color,
                        transform=ax.transData,
                        clip_on=False
                    ))

        #handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[key]) for key in ct_props.columns]
        handles = [
                plt.Rectangle((0, 0), 1, 1, color=color_dict.get(key, 'red')) 
                for key in ct_props.columns
]
        labels = [key.replace("Malignant Cell - ", "") for key in ct_props.columns]
        legend1 = ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left',
                            title="Cell Types",
                            title_fontproperties=font_manager.FontProperties(weight='bold'),
                            fontsize=self.config["plot_configs"]["general"]["legend_fontsize"],
                            frameon=False)
        ax.add_artist(legend1)

        if meta_annotations_list and meta_color_dicts_list and meta_columns_list:
            
            current_y_anchor = -0.15 
            
            
            for idx, (color_dict, column_name) in enumerate(zip(meta_color_dicts_list, meta_columns_list)):
                
                present_values = sorted(list(set(meta_annotations_list[idx])))
                
                order_handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[val]) for val in present_values if val in color_dict]
                order_labels = [str(val) for val in present_values if val in color_dict]
                
                if not order_handles: continue # Skip if no handles are found
                
                legend_title = column_name.replace('_', ' ').title()
                
                legend_pos = (0.0, current_y_anchor) 
                
                legend = ax.legend(order_handles, order_labels, 
                                bbox_to_anchor=legend_pos, 
                                loc='upper left', #
                                title=legend_title, frameon=False,
                                fontsize=self.config["plot_configs"]["general"]["legend_fontsize"],
                                ncol=1) 
                
                legend.get_title().set_fontweight('bold')
                ax.add_artist(legend)
                
                fig.canvas.draw() 
                bbox = legend.get_window_extent(fig.canvas.get_renderer())
                

                
                # Get height in figure fractional units
                figure_height = fig.get_figheight() * fig.dpi
                legend_height_fractional = bbox.height / figure_height
                
                # Update the anchor for the next legend to be below the current one
                current_y_anchor -= (legend_height_fractional + 0.05) 


        plt.subplots_adjust(right=0.95)
        return fig
    
    def create_all_stacked_barplots_multi_meta(self, adata, level_column, sample_column="Sample_ID", 
                                          metadata_columns=None, 
                                          subset_level=None, subset_value=None,
                                          order_by_column=None, order_ascending=True,
                                          figsize=(16, 8), save_name_prefix="composition_multi_meta",
                                          xlabel=None):
       
        if metadata_columns is None:
            metadata_columns = []
        
        # 1. Input Validation and Filtering
        if sample_column is None:
            for col in adata.obs.columns:
                if adata.obs[col].dtype in ['object', 'category']:
                    sample_column = col
                    break
            else:
                raise ValueError("No suitable sample column found.")
                
        if subset_level and subset_value:
            if isinstance(subset_value, str):
                subset_value = [subset_value]
            adata = adata[adata.obs[subset_level].isin(subset_value), :]

        if level_column not in adata.obs.columns or sample_column not in adata.obs.columns:
            raise ValueError("Required columns not found.")
        
        # Check that order_by_column is included in metadata_columns if it's set
        if order_by_column and order_by_column not in metadata_columns:
            metadata_columns.insert(0, order_by_column)

        ct_data = pd.crosstab(adata.obs[sample_column], adata.obs[level_column])
        ct_props = ct_data.div(ct_data.sum(axis=1), axis=0)

        color_dict = self.config["palettes"][level_column]
        ylabel = f"Cell Type Proportion ({subset_value[0]}s)" if subset_value else "Cell Type Proportion"
        title_base = f"Cell Type Composition by {sample_column.replace('_', ' ').title()}"

        figs = {}
        
        
        meta_color_dicts = []
        
        # Keep only the columns present in both metadata_columns and adata.obs.columns
        valid_meta_columns = [col for col in metadata_columns if col in adata.obs.columns]
        
        if not valid_meta_columns:
            meta_annotations = [None] 
            meta_color_dicts = [None]
            meta_columns_list = [None]
            
        else:
            for col in valid_meta_columns:
                try:
                    meta_color_dicts.append(self.config["palettes"][col])
                except KeyError:
                    raise ValueError(f"No palette found in self.config['palettes'] for metadata column '{col}'")
            
        # Helper to retrieve metadata values for a given sample order
        def get_meta_annotations(current_sample_order, valid_meta_columns, adata, sample_column):
            annotations_list = []
            for col in valid_meta_columns:
                # Get the metadata value for each sample in the current order
                order_values = [adata.obs[adata.obs[sample_column] == s][col].iloc[0] 
                                for s in current_sample_order]
                annotations_list.append(order_values)
            return annotations_list

        
        #  Basic Ordering
        meta_annotations_basic = get_meta_annotations(ct_props.index, valid_meta_columns, adata, sample_column)
        figs["basic"] = self.plot_barplot_multi_meta(
            ct_props, color_dict, f"{title_base}", ylabel, figsize, sample_column, 
            meta_annotations_basic, meta_color_dicts, valid_meta_columns, xlabel
        )

        # Clustered Ordering 
        clustered_order = self.cluster_samples(ct_props)
        meta_annotations_clustered = get_meta_annotations(clustered_order, valid_meta_columns, adata, sample_column)
        figs["clustered"] = self.plot_barplot_multi_meta(
            ct_props.loc[clustered_order], color_dict, f"{title_base} (Clustered)", ylabel, figsize, sample_column, 
            meta_annotations_clustered, meta_color_dicts, valid_meta_columns, xlabel
        )

        #  Grouped Clustered Ordering 
        if order_by_column and order_by_column in adata.obs.columns:
            grouped_order = self.cluster_grouped_samples(ct_props, adata, sample_column, order_by_column, order_ascending)
            meta_annotations_grouped = get_meta_annotations(grouped_order, valid_meta_columns, adata, sample_column)
            figs["clustered_grouped"] = self.plot_barplot_multi_meta(
                ct_props.loc[grouped_order], color_dict, f"{title_base} (Clustered per Group)", ylabel, figsize, sample_column, 
                meta_annotations_grouped, meta_color_dicts, valid_meta_columns, xlabel
            )
        
        # 6. Save Plots
        save_dir = self.output_dir / "compositional_plot_multi_meta"
        save_dir.mkdir(parents=True, exist_ok=True)
        for key, fig in figs.items():
            ax = fig.axes[0]
            # Collect all legends created by the plot function
            legends = [artist for artist in ax.get_children() if isinstance(artist, Legend)]
            fig.savefig(save_dir / f"{save_name_prefix}_{key}.png",
                        dpi=self.config['plot_configs']['general']['dpi_save'],
                        bbox_inches='tight',
                        bbox_extra_artists=legends,
                        facecolor='white')

        return figs
        
    def cell_abundance_barplot(self, adata, cell_type_column, figsize=(8, 5), title_suffix=None, save_name=None, xlabel=None):
       
        if cell_type_column not in adata.obs.columns:
            raise ValueError(f"Column '{cell_type_column}' not found in adata.obs.")

        # Calculate absolute cell counts
        cell_counts = adata.obs[cell_type_column].value_counts()
        
        # Calculate relative abundance
        total_cells = cell_counts.sum()
        relative_abundance = (cell_counts / total_cells) * 100

        # Sort categories by absolute cell count (descending)
        sorted_categories = cell_counts.sort_values(ascending=False).index.tolist()
        cell_counts = cell_counts.reindex(sorted_categories)
        relative_abundance = relative_abundance.reindex(sorted_categories)

        # X-axis label formatting
        x_label = xlabel if xlabel else cell_type_column.replace("_", " ").title()
        title_suffix = title_suffix if title_suffix else ""

    
        title_fontsize = self.config["plot_configs"]["general"]["title_fontsize"]
        title_fontweight = self.config["plot_configs"]["general"]["title_fontweight"]
        dpi_save = self.config['plot_configs']['general']['dpi_save']
        
        line_width = 1.8
        
        # Define save directory
        save_dir = Path(self.output_dir) / "cell_abundance_barplot"
        if save_name is not None:
            save_dir.mkdir(parents=True, exist_ok=True)


        fig_counts, ax_counts = plt.subplots(1, 1, figsize=figsize)
        
        ax_counts.bar(cell_counts.index, cell_counts.values, color="grey")
        ax_counts.set_title(f"Absolute Cell Abundance{title_suffix}",
                            fontsize=title_fontsize,
                            fontweight=title_fontweight)
        ax_counts.set_ylabel("Number of Cells", fontsize=12)
        ax_counts.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax_counts.tick_params(axis='x', rotation=90)
        ax_counts.tick_params(axis='both', width=line_width)
        [sp.set_linewidth(line_width) for sp in ax_counts.spines.values()]
        ax_counts.grid(False)

        plt.tight_layout()

        # Saving Absolute Counts Plot
        if save_name is not None:
            file_name = f"{save_name}_absolute_counts_grey.png"
            save_path = save_dir / file_name
            try:
                fig_counts.savefig(save_path, dpi=dpi_save, bbox_inches='tight', facecolor='white')
                print(f"Absolute Counts figure saved to: {save_path}")
            except Exception as e:
                print(f"Error saving Absolute Counts figure: {e}")


        fig_relative, ax_relative = plt.subplots(1, 1, figsize=figsize)
        
        ax_relative.bar(relative_abundance.index, relative_abundance.values, color="grey")
        ax_relative.set_title(f"Relative Cell Abundance{title_suffix}",
                            fontsize=title_fontsize,
                            fontweight=title_fontweight)
        ax_relative.set_ylabel("Percentage of Total Cells (%)", fontsize=12)
        ax_relative.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax_relative.tick_params(axis='x', rotation=90)
        ax_relative.tick_params(axis='both', width=line_width)
        [sp.set_linewidth(line_width) for sp in ax_relative.spines.values()]
        ax_relative.grid(False)

        plt.tight_layout()

        # Saving Relative Abundance Plot
        if save_name is not None:
            file_name = f"{save_name}_relative_abundance_grey.png"
            save_path = save_dir / file_name
            try:
                fig_relative.savefig(save_path, dpi=dpi_save, bbox_inches='tight', facecolor='white')
                print(f"Relative Abundance figure saved to: {save_path}")
            except Exception as e:
                print(f"Error saving Relative Abundance figure: {e}")

        return fig_counts, fig_relative
    

    def compare_cell_abundance_barplot(self, adata1, adata2, cell_type_column, 
                                    label1="Dataset 1", label2="Dataset 2", 
                                    figsize=(12, 6), title_suffix=None, save_name=None, 
                                    xlabel=None, custom_palette=False):
           

            # --- 1. Data Preparation and Validation ---
            if cell_type_column not in adata1.obs.columns or cell_type_column not in adata2.obs.columns:
                raise ValueError(f"Column '{cell_type_column}' not found in one or both adata.obs.")

            # Calculate counts and relative abundance for both
            counts1 = adata1.obs[cell_type_column].value_counts()
            counts2 = adata2.obs[cell_type_column].value_counts()
            
            total1 = counts1.sum()
            total2 = counts2.sum()

            relative1 = (counts1 / total1) * 100
            relative2 = (counts2 / total2) * 100

            # Combine all unique cell types and sort by the sum of counts across both datasets
            all_cell_types = pd.Index(counts1.index.union(counts2.index))
            
            # Fill missing types with 0 in the counts/relative series
            counts1 = counts1.reindex(all_cell_types, fill_value=0)
            counts2 = counts2.reindex(all_cell_types, fill_value=0)
            relative1 = relative1.reindex(all_cell_types, fill_value=0)
            relative2 = relative2.reindex(all_cell_types, fill_value=0)

            # Determine sorting order based on total counts
            total_counts = counts1 + counts2
            sorted_categories = total_counts.sort_values(ascending=False).index.tolist()
            
            # Reindex for plotting order
            counts1 = counts1.reindex(sorted_categories)
            counts2 = counts2.reindex(sorted_categories)
            relative1 = relative1.reindex(sorted_categories)
            relative2 = relative2.reindex(sorted_categories)

            
            # Define colors
            if custom_palette:
                try:
                    # Assuming a custom palette defined for this specific comparison (or general dataset comparison)
                    color1 = self.config["palettes"].get("comparison_colors", {}).get("color1", 'lightgray')
                    color2 = self.config["palettes"].get("comparison_colors", {}).get("color2", 'dimgray')
                except (AttributeError, KeyError):
                    print("Warning: Custom palette not found or misconfigured. Using default grey scale.")
                    color1 = 'lightgray'
                    color2 = 'dimgray'
            else:
                # Default grey scale
                color1 = 'lightgray'
                color2 = 'dimgray'

 
            title_fontsize = self.config["plot_configs"]["general"]["title_fontsize"]
            title_fontweight = self.config["plot_configs"]["general"]["title_fontweight"]
            dpi_save = self.config['plot_configs']['general']['dpi_save']
            line_width = 1.8
  
            # Axes setup
            x_label = xlabel if xlabel else cell_type_column.replace("_", " ").title()
            title_suffix = title_suffix if title_suffix else ""
            save_dir = Path(self.output_dir) / "cell_abundance_barplot"
            if save_name is not None:
                save_dir.mkdir(parents=True, exist_ok=True)

            # Bar plot parameters for grouped bars
            x = np.arange(len(sorted_categories))  
            width = 0.35 

            fig_counts, ax_counts = plt.subplots(1, 1, figsize=figsize)
            
            # Plot bars side-by-side
            rects1 = ax_counts.bar(x - width/2, counts1.values, width, label=label1, color=color1)
            rects2 = ax_counts.bar(x + width/2, counts2.values, width, label=label2, color=color2)
            
            ax_counts.set_title(f"Absolute Cell Abundance Comparison{title_suffix}",
                                fontsize=title_fontsize, fontweight=title_fontweight)
            ax_counts.set_ylabel("Number of Cells", fontsize=12)
            ax_counts.set_xlabel(x_label, fontsize=12, labelpad=10)
            ax_counts.set_xticks(x, sorted_categories, rotation=90)
            
            # Styling
            ax_counts.tick_params(axis='both', width=line_width)
            [sp.set_linewidth(line_width) for sp in ax_counts.spines.values()]
            ax_counts.legend(loc='upper right')
            ax_counts.grid(False)

            plt.tight_layout()

            # Saving Absolute Counts Plot
            if save_name is not None:
                file_name = f"{save_name}_absolute_counts_comparison_grey.png"
                save_path = save_dir / file_name
                try:
                    fig_counts.savefig(save_path, dpi=dpi_save, bbox_inches='tight', facecolor='white')
                    print(f"Absolute Counts comparison figure saved to: {save_path}")
                except Exception as e:
                    print(f"Error saving Absolute Counts comparison figure: {e}")



            fig_relative, ax_relative = plt.subplots(1, 1, figsize=figsize)
            
            rects3 = ax_relative.bar(x - width/2, relative1.values, width, label=label1, color=color1)
            rects4 = ax_relative.bar(x + width/2, relative2.values, width, label=label2, color=color2)
            
            ax_relative.set_title(f"Relative Cell Abundance Comparison{title_suffix}",
                                fontsize=title_fontsize, fontweight=title_fontweight)
            ax_relative.set_ylabel("Percentage of Total Cells (%)", fontsize=12)
            ax_relative.set_xlabel(x_label, fontsize=12, labelpad=10)
            ax_relative.set_xticks(x, sorted_categories, rotation=90)

            # Styling
            ax_relative.tick_params(axis='both', width=line_width)
            [sp.set_linewidth(line_width) for sp in ax_relative.spines.values()]
            ax_relative.legend(loc='upper right')
            ax_relative.grid(False)

            plt.tight_layout()

            # Saving Relative Abundance Plot
            if save_name is not None:
                file_name = f"{save_name}_relative_abundance_comparison_grey.png"
                save_path = save_dir / file_name
                try:
                    fig_relative.savefig(save_path, dpi=dpi_save, bbox_inches='tight', facecolor='white')
                    print(f"Relative Abundance comparison figure saved to: {save_path}")
                except Exception as e:
                    print(f"Error saving Relative Abundance comparison figure: {e}")

            return fig_counts, fig_relative