# __init__.py for data_analysis_pro_plotting
# ===================================================
# This file exposes the main functions of the package
# so they can be imported directly from the package.
# Only import functions here; do NOT copy full definitions.
# ===================================================

# -------------------------------
# General plotting
# -------------------------------
# Functions used across multiple analyses
from .plotting import plot_barplot
from .plotting import plot_venn2

# -------------------------------
# PCA analysis
# -------------------------------
# Functions related to PCA computation and visualization
from .pca_analysis import calculate_pca
from .pca_analysis import plot_pca
from .pca_analysis import obtain_top_variables
from .pca_analysis import plot_top_variables_pc

# -------------------------------
# Heatmap analysis
# -------------------------------
# Functions to compute and plot heatmaps, hierarchical clustering, distances
from .heatmap_analysis import calculate_distance_matrix
from .heatmap_analysis import plot_distance_heatmap
from .heatmap_analysis import hierarchical_heatmap_matrix
from .heatmap_analysis import plot_hierarchical_heatmap_highlighted_genes

# -------------------------------
# Dendrogram metrics
# -------------------------------
# Functions for dendrogram computation and plotting
from .dendogram_metrics import calculate_samples_dendogram
from .dendogram_metrics import plot_samples_dendogram

# -------------------------------
# Volcano & GSEA analysis
# -------------------------------
# Functions related to volcano plots and GSEA improvements
from .volcano_gsea_improvements import plot_volcano_highlighted_genes
from .volcano_gsea_improvements import plot_gsea_with_genes

# -------------------------------
# Optional: package version
# -------------------------------
__version__ = "0.1.0"