from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

def calculate_samples_dendogram(
    df_vst,
    distance_method="euclidean",
    linkage_method="average",
    scale_by_gene=False
):
    """
        Compute hierarchical clustering (linkage) among samples.
        
        Parameters
        ----------
        df_vst : pd.DataFrame
            Numeric matrix with features (genes, metabolites, etc.) as rows and samples as columns.
        distance_method : str, default="euclidean"
            Distance metric to use ('euclidean', 'correlation', 'cosine', etc.).
            Must be valid for scipy.spatial.distance.pdist.
        linkage_method : str, default="average"
            Linkage method for hierarchical clustering ('single', 'complete', 'average', 
            'weighted', 'centroid', 'median', 'ward').
        scale_by_gene : bool, default=False
            If True, scale each row (feature) to zero mean and unit variance (z-score).
        
        Returns
        -------
        Z : np.ndarray
            Linkage matrix for hierarchical clustering (suitable for dendrograms).
        labels : list
            Sample names in original order.
        
        Notes
        -----
        - Designed for any numeric matrix; not limited to transcriptomic data.
        - Row-wise scaling prevents features with large variance from dominating distances.
        - Works with pd.DataFrame containing numeric values only.
    """
    # =========================
    # Input validations
    # =========================
    if not isinstance(df_vst, pd.DataFrame):
        raise TypeError("df_vst must be a pandas DataFrame")
    
    if df_vst.shape[1] < 2:
        raise ValueError("df_vst must have at least two columns (samples) to compute distances")
    
    if df_vst.shape[0] < 1:
        raise ValueError("df_vst must have at least one row (feature/gene)")
    
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df_vst.dtypes):
        raise ValueError("All columns in df_vst must be numeric")
    
    # =========================
    # Validate distance and linkage methods
    # =========================
    from scipy.spatial.distance import _METRICS
    valid_linkages = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    
    if distance_method not in _METRICS:
        raise ValueError(f"distance_method must be one of {_METRICS}")
    
    if linkage_method not in valid_linkages:
        raise ValueError(f"linkage_method must be one of {valid_linkages}")
    
    # =========================
    # Safe  (z-score)
    # =========================
    if scale_by_gene:
        df_proc = df_vst.apply(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1), axis=1)
    else:
        df_proc = df_vst
   
    # -------------------------
    # Distance
    # -------------------------
    dist_condensed = pdist(
        df_proc.T,
        metric=distance_method
    )

    # -------------------------
    # Clustering
    # -------------------------
    Z = linkage(
        dist_condensed,
        method=linkage_method
    )

    return Z, df_vst.columns.tolist()



def plot_samples_dendogram(
    # =========================
    # Data
    # =========================
    Z,
    labels,

    # =========================
    # Leaf renaming
    # =========================
    rename_dict=None,

    # =========================
    # Dendogram
    # =========================
    orientation="top",
    leaf_rotation=90,
    dendrogram_linewidth=2,
    
    # =========================
    # Cluster thresholding
    # =========================
    color_threshold=None,   
    n_clusters=None,        
    
    # =========================
    # Colors
    # =========================
    mode_color="auto",          
    cluster_palette=None,       
    above_threshold_color="black",

    # =========================
    # Title
    # =========================
    show_title=False,
    title=None,
    fontsize_title=10,
    fontweight_title="bold",

    # =========================
    # Axis labels
    # =========================
    show_ylabel=True,
    show_xlabel=False,
    ylabel="Distancia",
    xlabel=None,
    fontsize_labels=9,
    fontweight_ylabel="normal",
    fontweight_xlabel="normal",

    # =========================
    # Ticks
    # =========================
    fontsize_ticks=8,
    fontweight_ticks="normal",
    tick_width=2,
    tick_length=6,
    xtick_label_rotation=None,

    # =========================
    # Spines
    # =========================
    show_spines=True,
    spine_width=2,

    # =========================
    # Figure saving and export
    # =========================
    figsize=(7.09, 4),
    dpi=600,
    save_path=None
):
    
    """
        Plot a hierarchical clustering dendrogram for samples with flexible options
        for leaf renaming, cluster coloring, axis labels, ticks, spines, and figure saving.
        
        This function visualizes the hierarchical clustering results obtained from a linkage
        matrix (e.g., output of `scipy.cluster.hierarchy.linkage`) and allows customization
        of labels, dendrogram orientation, colors, clusters thresholds, title, and export options.
        
        Parameters
        ----------
        
        Data 
        ----------
        Z : np.ndarray or list
            Linkage matrix obtained from hierarchical clustering. Must have shape (n-1, 4) if ndarray.
        labels : list of str
            Sample names corresponding to the rows/columns used to generate Z.
        
        Leaf renaming 
        ----------
        rename_dict : dict or None, default=None
            Dictionary mapping original labels to new labels for display. If None, original labels are used.
        
        Dendrogram 
        ----------
        orientation : str, default="top"
            Orientation of the dendrogram. Options: 'top', 'bottom', 'left', 'right'.
        leaf_rotation : float, default=90
            Rotation angle (degrees) of leaf labels.
        dendrogram_linewidth : float, default=2
            Line width of dendrogram branches.
        
        Cluster thresholding
        ----------
        color_threshold : float or None, default=None
            Manual threshold to color clusters; overrides n_clusters if provided.
        n_clusters : int or None, default=None
            Number of desired clusters; determines color threshold automatically if color_threshold is None.
        
        Colors
        ----------
        mode_color : str, default="auto"
            How to color dendrogram branches: 'auto' (default behavior), 'black' (all black), or 'custom'.
        cluster_palette : list or tuple, default=None
            List of colors for clusters if mode_color='custom'.
        above_threshold_color : str, default="black"
            Color for branches above threshold (only relevant if mode_color is 'custom').
        
        Title
        ----------
        show_title : bool, default=False
            Whether to display the plot title.
        title : str or None, default=None
            Plot title text. Defaults to "Hierarchical clustering" if show_title is True.
        fontsize_title : int, default=10
            Font size of the title.
        fontweight_title : str, default="bold"
            Font weight of the title text.
        
        Axis labels
        ----------
        show_ylabel : bool, default=True
            Whether to show Y-axis label.
        show_xlabel : bool, default=False
            Whether to show X-axis label.
        ylabel : str, default="Distance"
            Label for Y-axis.
        xlabel : str or None, default=None
            Label for X-axis.
        fontsize_labels : int, default=9
            Font size of axis labels.
        fontweight_ylabel : str, default="normal"
            Font weight for Y-axis label.
        fontweight_xlabel : str, default="normal"
            Font weight for X-axis label.
        
        Ticks
        ----------
        fontsize_ticks : int, default=8
            Font size of tick labels.
        fontweight_ticks : str, default="normal"
            Font weight of tick labels.
        tick_width : float, default=2
            Width of tick lines.
        tick_length : float, default=6
            Length of tick lines.
        xtick_label_rotation : float or None, default=None
            Custom rotation angle for X-axis labels; overrides leaf_rotation if provided.
        
        Spines
        ----------
        show_spines : bool, default=True
            Whether to display axis spines.
        spine_width : float, default=2
            Line width of axis spines.
        
        Figure saving and export
        -------------------------
        figsize : tuple of float, default=(7.09, 4)
            Figure size as (width, height) in inches.
        dpi : int, default=600
            Resolution of the figure in dots per inch.
        save_path : str or None, default=None
            Path to save the figure. If None, the figure is only displayed.
        
        Returns
        -------
        None
            This function displays the dendrogram plot and optionally saves it to a file.
        
        Notes
        -----
        - The linkage matrix Z must correspond to the samples in `labels`.
        - If both `color_threshold` and `n_clusters` are provided, `color_threshold` takes precedence.
        - `cluster_palette` must be provided when mode_color='custom'; otherwise, an error is raised.
        - Supports flexible dendrogram orientations and label rotations for publication-quality figures.
    """

    # -------------------------
    # Validate linkage matrix Z
    # -------------------------
    if not isinstance(Z, (np.ndarray, list)):
        raise TypeError("Z must be a numpy array or a list from hierarchical clustering linkage.")
    
    if isinstance(Z, np.ndarray) and Z.shape[1] != 4:
        raise ValueError("Z must have 4 columns as a valid linkage matrix.")
    
    # -------------------------
    # Validate labels
    # -------------------------
    if not isinstance(labels, (list, tuple)):
        raise TypeError("labels must be a list or tuple of strings.")
    
    if not all(isinstance(l, str) for l in labels):
        raise TypeError("All items in labels must be strings.")
    
    # -------------------------
    # Validate color options
    # -------------------------
    valid_modes = ["auto", "black", "custom"]
    if mode_color not in valid_modes:
        raise ValueError(f"mode_color must be one of {valid_modes}")
    
    if mode_color == "custom" and (cluster_palette is None or not isinstance(cluster_palette, (list, tuple))):
        raise ValueError("cluster_palette must be provided as a list or tuple when mode_color='custom'.")
    
    # -------------------------
    # Validate n_clusters and color_threshold
    # -------------------------
    if n_clusters is not None:
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer >= 2.")
    
    if color_threshold is not None and not isinstance(color_threshold, (int, float)):
        raise TypeError("color_threshold must be numeric or None.")
    
    # -------------------------
    # Validate figure and save_path
    # -------------------------
    if figsize is not None and (not isinstance(figsize, (tuple, list)) or len(figsize) != 2):
        raise ValueError("figsize must be a tuple of length 2 (width, height).")
    
    if save_path is not None:
        if not isinstance(save_path, str):
            raise TypeError("save_path must be a string.")

    # -------------------------
    # Apply renaming
    # -------------------------
    if rename_dict is not None:
        labels_to_plot = [
            rename_dict.get(lab, lab)
            for lab in labels
        ]
    else:
        labels_to_plot = labels

    # -------------------------
    # Figurae
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)
    
    # =========================
    # Threshold definition
    # =========================
    
    if color_threshold is not None:
    
        threshold = color_threshold
    
    elif n_clusters is not None:
    
        if n_clusters < 2:
            raise ValueError("n_clusters debe ser >= 2")
    
        threshold = Z[-(n_clusters - 1), 2]
    
    else:
    
        threshold = None


    # =========================
    # Managing colors
    # =========================
    
    if mode_color == "black":
    
        dendrogram(
            Z,
            labels=labels_to_plot,
            orientation=orientation,
            leaf_rotation=leaf_rotation,
            ax=ax,
            color_threshold=0,                     # todo negro
            above_threshold_color="black"
        )
    
    elif mode_color == "custom" and cluster_palette is not None:
    
        set_link_color_palette(cluster_palette)
    
        dendrogram(
            Z,
            labels=labels_to_plot,
            orientation=orientation,
            leaf_rotation=leaf_rotation,
            ax=ax,
            color_threshold=threshold,              
            above_threshold_color=above_threshold_color
        )
    
        set_link_color_palette(None)
    
    else:  
    
        dendrogram(
            Z,
            labels=labels_to_plot,
            orientation=orientation,
            leaf_rotation=leaf_rotation,
            ax=ax,
            color_threshold=threshold               
        )
    

    for collection in ax.collections:
        collection.set_linewidth(dendrogram_linewidth)
        
    # =========================
    # Title
    # =========================
    if show_title:
        if title is None:
            title = "Hierarchical clustering"
        ax.set_title(
            title,
            fontsize=fontsize_title,
            fontweight=fontweight_title
        )

    # =========================
    # Y label
    # =========================
    if show_ylabel:
        if ylabel is None:
            ylabel = "Distance"
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize_labels,
            fontweight=fontweight_ylabel
        )
    else:
        ax.set_ylabel("")

    # =========================
    # X label
    # =========================
    if show_xlabel:
        if xlabel is None:
            xlabel = "Samples"
        ax.set_xlabel(
            xlabel,
            fontsize=fontsize_labels,
            fontweight=fontweight_xlabel
        )
    else:
        ax.set_xlabel("")

    # =========================
    # Ticks
    # =========================
    ax.tick_params(
        axis="both",
        labelsize=fontsize_ticks,
        width=tick_width,
        length=tick_length
    )

    # Rotación personalizada
    if xtick_label_rotation is not None:
        for label in ax.get_xticklabels():
            label.set_rotation(xtick_label_rotation)

    # Peso de fuente ticks
    for label in ax.get_xticklabels():
        label.set_fontweight(fontweight_ticks)

    for label in ax.get_yticklabels():
        label.set_fontweight(fontweight_ticks)

    # =========================
    # Spines
    # =========================
    if show_spines:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

    # =========================
    # Figure saving
    # =========================
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.tight_layout()
    plt.show()