import pandas as pd
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

def calculate_distance_matrix(df_vst, distance_metric="euclidean"):
    """
        Calculate a distance matrix between observations based on numeric data.
        
        Parameters
        ----------
        Data
        ----------
        df_data : pandas.DataFrame
            Numeric data matrix. Rows represent variables/features (e.g., genes, metabolites, survey items),
            columns represent observations/samples. All values must be numeric.
        distance_metric : str, default='euclidean'
            Distance metric to use. Options include 'euclidean', 'correlation', 'cosine', etc.
            Refer to sklearn.metrics.pairwise_distances for full list.
        
        Returns
        -------
        pandas.DataFrame
            Symmetric distance matrix between observations. Rows and columns are observation names
            (column names from `df_data`).
        
        Notes
        -----
        - Observations are treated as samples and variables as features (transpose is used internally).
        - Can be used for clustering, heatmaps, similarity/dissimilarity analysis, or any downstream analysis
          that requires a distance matrix.
        - Works for any type of numeric dataset, not limited to transcriptomics.
    """
    # =========================
    # Input validation
    # =========================
    if not isinstance(df_vst, pd.DataFrame):
        raise TypeError("df_vst must be a pandas DataFrame")

    if df_vst.shape[1] < 2:
        raise ValueError("df_vst must have at least two samples (columns) to compute distances")

    # =========================
    # Compute distance matrix
    # =========================
    dist_matrix = pairwise_distances(df_vst.T, metric=distance_metric)

    dist_df = pd.DataFrame(
        dist_matrix,
        index=df_vst.columns,
        columns=df_vst.columns
    )

    return dist_df




def plot_distance_heatmap(
    # =========================
    # Data
    # =========================
    dist_df,

    # =========================
    # Heatmap 
    # =========================
    cmap="viridis",
    cell_linewidth=2,
    cell_linecolor="black",

    # =========================
    # Title
    # =========================
    show_title=False,
    title=None,
    fontsize_title=10,
    fontweight_title="normal",

    # =========================
    # Axis labels
    # =========================
    show_xlabel=False,
    show_ylabel=False,
    xlabel=None,
    ylabel=None,
    fontsize_labels=9,
    fontweight_labels="normal",

    # =========================
    # Ticks (heatmap)
    # =========================
    show_xticks=True,
    show_yticks=True,
    xtick_labels_map=None,
    ytick_labels_map=None,
    fontsize_ticks=8,
    fontweight_ticks="normal",
    tick_width=2,
    tick_length=4,
    xtick_label_rotation=None,
    ytick_label_rotation=None,
    
    # =========================
    # Spines (heatmap)
    # =========================
    spine_width=2,

    # =========================
    # Dendrograms
    # =========================
    show_row_dendrogram=True,
    show_col_dendrogram=True,
    dendrogram_linewidth=2,
    dendrogram_color="black",

    # =========================
    # Colorbar
    # =========================
    show_colorbar=True,
    cbar_label=None,
    cbar_left=0.02,
    cbar_bottom=0.8,
    cbar_width=0.02,
    cbar_height=0.18,
    cbar_tick_size=8,
    cbar_tick_width=2,
    cbar_tick_length=4,
    fontweight_cbar_ticks="normal",
    cbar_label_size=9,
    fontweight_cbar_label="normal",
    cbar_show_spines=False,
    cbar_spine_width=2,

    # =========================
    # Figure and export
    # =========================
    figsize=(8, 8),
    dpi=300,
    save_path=None
):
    """
        Plot a clustered heatmap of a distance matrix with optional dendrograms.
        
        Parameters
        ----------
        
        Data
        ----------
        dist_df : pandas.DataFrame
            Square distance matrix between samples(rows = columns = samples/objects). Usually the output of calculate_distance_matrix.
            Can be derived from any numeric dataset, not limited to biological data.
        
        
        Heatmap 
        ----------
        cmap : str, default='viridis'
            Colormap for heatmap.
        cell_linewidth : float, default=2
            Width of the lines separating cells.
        cell_linecolor : str, default='black'
            Color of the lines separating cells.
        
        
        Title
        ----------
        show_title : bool, default=False
        title : str, optional
        fontsize_title : int, default=10
        fontweight_title : str, default='normal'
        
        
        Axis labels
        ----------
        show_xlabel, show_ylabel : bool, default=False
        xlabel, ylabel : str, optional
        fontsize_labels : int, default=9
        fontweight_labels : str, default='normal'
        
        
        Ticks
        ----------
        show_xticks, show_yticks : bool, default=True
        xtick_labels_map, ytick_labels_map : dict, optional
        fontsize_ticks : int, default=8
        fontweight_ticks : str, default='normal'
        tick_width, tick_length : int, default=2, 4
        xtick_label_rotation, ytick_label_rotation : int, optional
        
        
        Spines
        ----------
        spine_width : float, default=2
        
        
        Dendrograms
        ----------
        show_row_dendrogram, show_col_dendrogram : bool, default=True
        dendrogram_linewidth : float, default=2
        dendrogram_color : str, default='black'
        
        
        Colorbar
        ----------
        show_colorbar : bool, default=True
        cbar_label : str, optional
        cbar_left, cbar_bottom, cbar_width, cbar_height : float, default=0.02,0.8,0.02,0.18
        cbar_tick_size : int, default=8
        cbar_tick_width, cbar_tick_length : int, default=2,4
        fontweight_cbar_ticks : str, default='normal'
        cbar_label_size : int, default=9
        fontweight_cbar_label : str, default='normal'
        cbar_show_spines : bool, default=False
        cbar_spine_width : float, default=2
        
        
        Figure and export
        ----------
        figsize : tuple, default=(8,8)
        dpi : int, default=300
        save_path : str, optional
        
        Returns
        -------
        None
            Displays a clustered heatmap. Saves the figure if save_path is provided.
        
        Notes
        -----
        - Expects a square distance matrix as input.
        - Can optionally customize dendrograms, ticks, labels, spines, and colorbar.
    """
    # =========================
    # Input validation
    # =========================
    if not isinstance(dist_df, pd.DataFrame):
        raise TypeError("dist_df must be a pandas DataFrame")
    
    if dist_df.shape[0] != dist_df.shape[1]:
        raise ValueError("dist_df must be a square matrix (same number of rows and columns)")
    
    if dist_df.shape[0] < 2:
        raise ValueError("dist_df must have at least 2 samples to plot a heatmap")
  
    # -------------------------
    # Clustermap
    # -------------------------
    g = sns.clustermap(
        dist_df,
        cmap=cmap,
        linewidths=cell_linewidth,
        linecolor=cell_linecolor,
        figsize=figsize,
        row_cluster=show_row_dendrogram,
        col_cluster=show_col_dendrogram,
        cbar_pos=(cbar_left, cbar_bottom, cbar_width, cbar_height)
        if show_colorbar else None
    )

    ax = g.ax_heatmap

    # =========================
    # Title
    # =========================
    if show_title:
        if title is None:
            title = "Distance heatmap"
        g.fig.suptitle(
            title,
            fontsize=fontsize_title,
            fontweight=fontweight_title,
            y=1.05
        )

    # =========================
    # Labels
    # =========================
    if show_xlabel:
        if xlabel is None:
            xlabel = "X axis"
        ax.set_xlabel(
            xlabel,
            fontsize=fontsize_labels,
            fontweight=fontweight_labels
        )
    else:
        ax.set_xlabel("")

    if show_ylabel:
        if ylabel is None:
            ylabel = "Y axis"
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize_labels,
            fontweight=fontweight_labels
        )
    else:
        ax.set_ylabel("")

    # =========================
    # Ticks visibility
    # =========================
    if not show_xticks:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if not show_yticks:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # =========================
    # Ticks rename
    # =========================
    if xtick_labels_map and show_xticks:
        new_labels = [
            xtick_labels_map.get(label.get_text(), label.get_text())
            for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(new_labels)

    if ytick_labels_map and show_yticks:
        new_labels = [
            ytick_labels_map.get(label.get_text(), label.get_text())
            for label in ax.get_yticklabels()
        ]
        ax.set_yticklabels(new_labels)

    # =========================
    # Ticks style
    # =========================
    ax.tick_params(
        axis="both",
        labelsize=fontsize_ticks,
        width=tick_width,
        length=tick_length
    )
    
    # ----- X ticks -----
    for label in ax.get_xticklabels():
        label.set_fontweight(fontweight_ticks)
    
        if xtick_label_rotation is not None:
            label.set_rotation(xtick_label_rotation)
    
            # recommended automatic alignment
            if xtick_label_rotation == 0:
                label.set_ha("center")
            elif xtick_label_rotation > 0:
                label.set_ha("right")
            else:
                label.set_ha("left")
    
    # ----- Y ticks -----
    for label in ax.get_yticklabels():
        label.set_fontweight(fontweight_ticks)
    
        if ytick_label_rotation is not None:
            label.set_rotation(ytick_label_rotation)
    
            if ytick_label_rotation == 0:
                label.set_va("center")
            elif ytick_label_rotation > 0:
                label.set_va("bottom")
            else:
                label.set_va("top")

    # =========================
    # Heatmap spines
    # =========================
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # =========================
    # Dendograms
    # =========================
    if show_row_dendrogram:
        for collection in g.ax_row_dendrogram.collections:
            collection.set_linewidth(dendrogram_linewidth)
            collection.set_color(dendrogram_color)
        g.ax_row_dendrogram.tick_params(width=tick_width)

    else:
        g.ax_row_dendrogram.set_visible(False)

    if show_col_dendrogram:
        for collection in g.ax_col_dendrogram.collections:
            collection.set_linewidth(dendrogram_linewidth)
            collection.set_color(dendrogram_color)
        g.ax_col_dendrogram.tick_params(width=tick_width)

    else:
        g.ax_col_dendrogram.set_visible(False)

    # =========================
    # Colorbar
    # =========================
    if show_colorbar:

        cbar = g.cax

        cbar.tick_params(
            labelsize=cbar_tick_size,
            width=cbar_tick_width,
            length=cbar_tick_length
        )

        for label in cbar.get_yticklabels():
            label.set_fontweight(fontweight_cbar_ticks)

        if cbar_label:
            cbar.set_ylabel(
                cbar_label,
                fontsize=cbar_label_size,
                fontweight=fontweight_cbar_label
            )

        if cbar_show_spines:
            for spine in cbar.spines.values():
                spine.set_linewidth(cbar_spine_width)
        else:
            for spine in cbar.spines.values():
                spine.set_visible(False)

    else:
        g.cax.set_visible(False)

    # =========================
    # Figure saving
    # =========================
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def hierarchical_heatmap_matrix(df_vst, scale="zscore"):
    
    """
        Prepare a matrix for hierarchical heatmap visualization.
    
        Parameters
        ----------
        
        Data
        ----------
        df_vst : pandas.DataFrame
            Numeric matrix with variables in rows and observations in columns.
            Can represent gene expression, metabolite levels, proteomics, or any other numeric dataset.
        scale : str or None, default="zscore"
            Row-wise normalization method:
            - "zscore": normalize each row to zero mean and unit variance.
            - None: keep original values.
            Rows with zero variance are set to zero if zscore normalization is applied.
        
        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame prepared for plotting.
            Normalized values if `scale="zscore"`, otherwise unchanged.
        
        Notes
        -----
        - Designed to prepare data for hierarchical clustering and heatmap visualization.
        - Works with any numeric dataset, not limited to transcriptomic data.
        - Z-score normalization is applied row-wise to highlight relative differences across observations.
    """

    # =========================
    # Input validation
    # =========================
    if not isinstance(df_vst, pd.DataFrame):
        raise TypeError("df_vst must be a pandas DataFrame")
    
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_vst.dtypes):
        raise ValueError("All columns in df_vst must be numeric")
    
    if scale not in ["zscore", None]:
        raise ValueError('scale must be either "zscore" or None')

    # =========================
    # Copy input
    # =========================
    df_plot = df_vst.copy()

    # =========================
    # Apply Z-score normalization
    # =========================
    if scale == "zscore":
        df_plot = df_plot.apply(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else np.zeros_like(x),
            axis=1
        )

    return df_plot



def plot_hierarchical_heatmap_highlighted_genes (
    # =========================
    # Data
    # =========================
    df_plot,

    # =========================
    # Heatmap
    # =========================
    cmap="viridis",
    center=None,
    clustering_method="euclidean",
    linkage_method="average",
    cell_linewidth=0,

    # ===========================
    # Clustering and dendogram
    # ===========================
    row_cluster=True,
    col_cluster=True,
    show_row_dendrogram=False,
    show_col_dendrogram=True,
    dendrogram_linewidth=2,
    
    # =========================
    # Labels (ticks)
    # =========================
    show_xtick_labels=True,
    xtick_label_map=None,
    show_ytick_labels=False,
    fontsize_samples=8,
    fontsize_genes=6,
    max_yticks=100,
    fontweight_samples="normal",
    fontweight_genes="normal",
    tick_width=2,
    tick_length=4,
    xtick_label_rotation=90,
    
    # =========================
    # Axis labels
    # =========================
    xlabel=None,
    ylabel=None,
    show_xlabel=True,
    show_ylabel=True,
    fontsize_axis_labels=9,
    fontweight_axis_labels="normal",

    # =========================
    # Title
    # =========================
    title=None,
    show_title=True,
    fontsize_title=10,
    fontweight_title="normal",   

    # =========================
    # Colorbar
    # =========================
    show_colorbar=True,
    cbar_left=0.02,
    cbar_bottom=0.8,
    cbar_width=0.02,
    cbar_height=0.18,
    cbar_tick_size=8,
    cbar_tick_width=2,
    cbar_tick_length=4,
    cbar_label=None,
    cbar_label_size=9,
    fontweight_cbar_ticks="normal",
    fontweight_cbar_label="normal",
    cbar_show_spines=False,
    cbar_spine_width=2,
    
    # =========================
    # Highlighted genes
    # =========================
    highlighted_genes=None,
    marker_show=True,
    marker_color="red",
    marker_style="o",
    marker_size=100,
    marker_xpos=None,
    label_color="red",
    label_fontsize=8,
    label_fontweight="bold",
    label_fontstyle="normal",   # normal, italic, oblique
    label_xpos=0,
    label_ypos=None,

    # =========================
    # Highlighted rectangle
    # =========================
    highlight_rect=True,
    rect_color="red",
    rect_linewidth=2,
    rect_height=1.0,

    # =========================
    # Figure saving and export
    # =========================
    figsize=(7.09, 8),
    dpi=600,
    save_path=None,
 
):
    """
        Plot a hierarchical heatmap of genes and samples, optionally highlighting specific genes 
        with markers, labels, and rectangles.
        
        This function uses seaborn's clustermap to generate a clustered heatmap, with 
        customizable dendrograms, colorbar, axis labels, ticks, and highlighted genes. 
        It is designed for visualizing gene expression data, VST-normalized counts, or any 
        numeric matrix where rows represent genes and columns represent samples.
        
        This function is typically used for gene expression data (rows = genes, columns = samples),
        but it can be applied to any numerical matrix where rows represent items/features
        and columns represent conditions/samples.
        
        Parameters
        ----------
        Data
        ----------
        df_plot : pd.DataFrame
            Numeric DataFrame with genes as rows and samples as columns.
        
        Heatmap
        ----------
        cmap : str, default="viridis"
            Colormap for the heatmap.
        center : float or None, default=None
            The value at which to center the colormap.
        clustering_method : str, default="euclidean"
            Distance metric for clustering ('euclidean', 'correlation', etc.).
        linkage_method : str, default="average"
            Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward').
        cell_linewidth : float, default=0
            Width of the lines separating heatmap cells.
        
        Clustering and Dendrograms
        ----------
        row_cluster : bool, default=True
            Whether to cluster rows.
        col_cluster : bool, default=True
            Whether to cluster columns.
        show_row_dendrogram : bool, default=False
            Show the row dendrogram.
        show_col_dendrogram : bool, default=True
            Show the column dendrogram.
        dendrogram_linewidth : float, default=2
            Line width of dendrogram lines.
        
        Labels (ticks)
        ----------
        show_xtick_labels : bool, default=True
            Show sample labels on x-axis.
        xtick_label_map : dict or None, default=None
            Mapping of original column names to new labels.
        show_ytick_labels : bool, default=False
            Show gene labels on y-axis.
        fontsize_samples : int, default=8
            Font size of x-axis tick labels.
        fontsize_genes : int, default=6
            Font size of y-axis tick labels.
        max_yticks : int, default=100
            Maximum number of y-axis tick labels to display.
        fontweight_samples, fontweight_genes : str, default='normal'
            Font weight for x and y tick labels ('normal', 'bold', 'light').
        tick_width, tick_length : float, default=2,4
            Width and length of axis ticks.
        xtick_label_rotation : float, default=90
            Rotation angle for x-axis labels.
        
        Axis labels
        ----------
        xlabel, ylabel : str or None
            Labels for x and y axes.
        show_xlabel, show_ylabel : bool, default=True
            Whether to show x and y axis labels.
        fontsize_axis_labels : int, default=9
            Font size of axis labels.
        fontweight_axis_labels : str, default='normal'
            Font weight of axis labels.
        
        Title
        ----------
        title : str or None
            Plot title.
        show_title : bool, default=True
            Whether to display the title.
        fontsize_title : int, default=10
            Font size of the title.
        fontweight_title : str, default='normal'
            Font weight of the title.
        
        Colorbar
        ----------
        show_colorbar : bool, default=True
            Show colorbar.
        cbar_left, cbar_bottom, cbar_width, cbar_height : float
            Position and size of colorbar.
        cbar_tick_size, cbar_tick_width, cbar_tick_length : float
            Tick size, width, and length on colorbar.
        cbar_label : str or None
            Label of the colorbar.
        cbar_label_size : int, default=9
            Font size of colorbar label.
        fontweight_cbar_ticks, fontweight_cbar_label : str, default='normal'
            Font weight of colorbar ticks and label.
        cbar_show_spines : bool, default=False
            Show spines around the colorbar.
        cbar_spine_width : float, default=2
            Spine line width if colorbar spines are shown.
        
        Highlighted genes
        ----------
        highlighted_genes : list of str or None
            Genes to highlight on the heatmap.
        marker_show : bool, default=True
            Show markers for highlighted genes.
        marker_color : str, default='red'
            Color of gene markers.
        marker_style : str, default='o'
            Marker style.
        marker_size : float, default=100
            Size of gene markers.
        marker_xpos : float or None
            X-coordinate of marker, default is center of heatmap columns.
        label_color : str, default='red'
            Color of gene label text.
        label_fontsize : int, default=8
            Font size of gene label text.
        label_fontweight : str, default='bold'
            Font weight of gene label text.
        label_fontstyle : str, default='normal'
            Font style ('normal', 'italic', 'oblique').
        label_xpos, label_ypos : float
            Coordinates for placing gene labels; defaults to left and gene position.
        
        Highlighted rectangle
        ----------
        highlight_rect : bool, default=True
            Draw a rectangle around highlighted genes.
        rect_color : str, default='red'
            Color of rectangle.
        rect_linewidth : float, default=2
            Line width of rectangle border.
        rect_height : float, default=1.0
            Height of rectangle to cover gene row.
        
        Figure and Saving
        ----------
        figsize : tuple, default=(7.09, 8)
            Figure size (width, height) in inches.
        dpi : int, default=600
            Resolution of the figure.
        save_path : str or None
            File path to save the figure. If None, figure is not saved.
        
        Returns
        -------
        None
            The function displays the heatmap and optionally saves it to a file.
        
        Notes
        -----
        - This function requires a numeric DataFrame with genes as rows.
        - Highlighted genes must exist in df_plot.index.
        - If the number of rows exceeds max_yticks, y-axis labels may be suppressed.
        - Suitable for visualizing gene expression or other numeric matrices with clustering.
    """
    
    # =========================
    # Input validation
    # =========================
    
    # df_plot
    if not isinstance(df_plot, pd.DataFrame):
        raise TypeError("df_plot must be a pandas DataFrame")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_plot.dtypes):
        raise ValueError("All columns in df_plot must be numeric")
    
    # highlighted_genes
    if highlighted_genes is not None:
        if not isinstance(highlighted_genes, (list, tuple)):
            raise TypeError("highlighted_genes must be a list or tuple of gene names")
        if not all(isinstance(g, str) for g in highlighted_genes):
            raise TypeError("All items in highlighted_genes must be strings")
        missing_genes = [g for g in highlighted_genes if g not in df_plot.index]
        if missing_genes:
            raise ValueError(f"The following genes are not in df_plot index: {missing_genes}")
    
    # Numeric parameters
    for param_name in ["marker_size", "rect_linewidth", "rect_height", "dpi", "cbar_tick_size",
                       "cbar_tick_width", "cbar_tick_length", "fontsize_samples", "fontsize_genes",
                       "fontsize_axis_labels", "fontsize_title", "label_fontsize"]:
        param_value = locals()[param_name]
        if param_value is not None and param_value < 0:
            raise ValueError(f"{param_name} must be non-negative")
    
    # String parameters
    valid_fontweights = ["normal", "bold", "light"]
    for fw_param in ["fontweight_samples", "fontweight_genes", "fontweight_axis_labels",
                     "fontweight_title", "fontweight_cbar_label", "fontweight_cbar_ticks",
                     "label_fontweight"]:
        param_value = locals()[fw_param]
        if param_value not in valid_fontweights:
            raise ValueError(f"{fw_param} must be one of {valid_fontweights}")
    
    # save_path
    if save_path is not None:
        if not isinstance(save_path, str):
            raise TypeError("save_path must be a string")
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            raise FileNotFoundError(f"The directory for save_path does not exist: {save_dir}")
      
        n_genes = df_plot.shape[0]

    # =====================================================
    # Clustermap
    # =====================================================
    g = sns.clustermap(
        df_plot,
        cmap=cmap,
        center=center,
        metric=clustering_method,
        method=linkage_method,
        figsize=figsize,
        linewidths=cell_linewidth,
        xticklabels=True,
        yticklabels=True if (show_ytick_labels and n_genes <= max_yticks) else False,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cbar=show_colorbar
    )

    ax = g.ax_heatmap

    # =====================================================
    # Colorbar
    # =====================================================
    if show_colorbar and g.cax is not None:

        g.cax.set_position([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = g.cax

        cbar.tick_params(
            labelsize=cbar_tick_size,
            width=cbar_tick_width,
            length=cbar_tick_length
        )

        for label in cbar.get_yticklabels():
            label.set_fontweight(fontweight_cbar_ticks)

        if cbar_label:
            cbar.set_ylabel(
                cbar_label,
                fontsize=cbar_label_size,
                fontweight=fontweight_cbar_label
            )

        if cbar_show_spines:
            for spine in cbar.spines.values():
                spine.set_linewidth(cbar_spine_width)
        else:
            for spine in cbar.spines.values():
                spine.set_visible(False)

    elif g.cax is not None:
        g.cax.set_visible(False)

    # =====================================================
    # Dendograms
    # =====================================================
    if not show_row_dendrogram and g.ax_row_dendrogram is not None:
        g.ax_row_dendrogram.set_visible(False)

    if not show_col_dendrogram and g.ax_col_dendrogram is not None:
        g.ax_col_dendrogram.set_visible(False)
    elif col_cluster and g.ax_col_dendrogram is not None:
        for collection in g.ax_col_dendrogram.collections:
            collection.set_linewidth(dendrogram_linewidth)

    # =====================================================
    # X labels (robust to reordering)
    # =====================================================
    if show_xtick_labels:
    
        if xtick_label_map is not None:
    
            if col_cluster and g.dendrogram_col is not None:
                reordered_cols = [
                    df_plot.columns[i]
                    for i in g.dendrogram_col.reordered_ind
                ]
            else:
                reordered_cols = df_plot.columns
    
            nuevos_labels = [
                xtick_label_map.get(col, col)
                for col in reordered_cols
            ]
    
            ax.set_xticklabels(nuevos_labels)
    
        for label in ax.get_xticklabels():
            label.set_fontsize(fontsize_samples)
            label.set_fontweight(fontweight_samples)
            label.set_rotation(xtick_label_rotation)
    
            # Alineación automática recomendable
            if xtick_label_rotation == 0:
                label.set_ha("center")
            elif xtick_label_rotation > 0:
                label.set_ha("right")
            else:
                label.set_ha("left")
    
    else:
        ax.set_xticklabels([])

    # =====================================================
    # Y labels
    # =====================================================
    if show_ytick_labels and n_genes <= max_yticks:
        for label in ax.get_yticklabels():
            label.set_fontsize(fontsize_genes)
            label.set_fontweight(fontweight_genes)
            label.set_rotation(xtick_label_rotation)

    # =====================================================
    # Axis labels
    # =====================================================
    if show_xlabel:
        ax.set_xlabel(
            xlabel if xlabel is not None else "Samples",
            fontsize=fontsize_axis_labels,
            fontweight=fontweight_axis_labels
        )
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(
            ylabel if ylabel is not None else "Genes",
            fontsize=fontsize_axis_labels,
            fontweight=fontweight_axis_labels
        )
    else:
        ax.set_ylabel("")

    # =====================================================
    # Ticks
    # =====================================================
    ax.tick_params(axis="x", width=tick_width, length=tick_length)
    ax.tick_params(axis="y", width=tick_width, length=tick_length)

    # =====================================================
    # Title
    # =====================================================
    if show_title:
        if not title:
            title = "Hierarchical Heatmap"
        g.fig.suptitle(
            title,
            fontsize=fontsize_title,
            fontweight=fontweight_title,
            y=1.02
        )

    # =====================================================
    # Highlighted genes
    # =====================================================
    if highlighted_genes and g.dendrogram_row is not None:

        orden_genes = df_plot.index[g.dendrogram_row.reordered_ind]

        for gene in highlighted_genes:
            if gene in orden_genes:

                pos = list(orden_genes).index(gene)
                xpos = df_plot.shape[1]/2 if marker_xpos is None else marker_xpos

                if marker_show:
                    ax.scatter(
                        x=[xpos], y=[pos],
                        color=marker_color,
                        s=marker_size,
                        marker=marker_style,
                        zorder=10
                    )

                ypos = pos if label_ypos is None else label_ypos

                ax.text(
                    x=label_xpos,
                    y=ypos,
                    s=gene,
                    color=label_color,
                    va="center",
                    fontsize=label_fontsize,
                    fontweight=label_fontweight,
                    fontstyle=label_fontstyle
                )

                if highlight_rect:
                    rect = Rectangle(
                        xy=(0, pos - rect_height/2),
                        width=df_plot.shape[1],
                        height=rect_height,
                        fill=False,
                        edgecolor=rect_color,
                        linewidth=rect_linewidth,
                        zorder=5
                    )
                    ax.add_patch(rect)

    # =====================================================
    # Figure saving
    # =====================================================
    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()