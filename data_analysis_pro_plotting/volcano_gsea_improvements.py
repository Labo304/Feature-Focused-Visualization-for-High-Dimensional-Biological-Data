import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from gseapy.plot import gseaplot  


def plot_volcano_highlighted_genes(
    # =========================
    # Data
    # =========================
    df,
    x="log2FoldChange",
    y="minus_log10_padj",
    significance_col="significance",
    gene_col=None,             # columna con nombres de genes

    # =========================
    # Groups ans colors
    # =========================
    palette={"NS": "gray", "Up": "red", "Down": "blue"},
    point_size=10,
    alpha=0.4,

    # =========================
    # Cut lines
    # =========================
    fc_threshold=0.5,
    pval_threshold=0.05,
    cut_linewidth=1,
    cut_linestyle="--",

    # =========================
    # Axis labels
    # =========================
    xlabel="log2 Fold Change",
    ylabel="-log10(padj)",
    show_axis_labels=True,
    fontsize_labels=9,
    fontweight_xlabel="normal",
    fontweight_ylabel="normal",

    # =========================
    # Title
    # =========================
    title=None,
    show_title=True,
    fontsize_title=10,
    fontweight_title="bold",

    # =========================
    # Legend
    # =========================
    show_legend=True,
    legend_title="Significance",
    fontsize_legend=8,
    fontsize_legend_title=9,
    fontweight_legend_title="bold",
    fontweight_legend_labels="normal",
    fontstyle_legend_title="normal",     
    fontstyle_legend_labels="normal",
    legend_loc="upper right",
    legend_frame=False,
    legend_edgecolor="black",
    legend_linewidth=1,
    legend_facecolor="white",
    legend_alpha=1,

    # =========================
    # Ticks
    # =========================
    show_ticks=True,
    tick_width=2,
    tick_length=4,
    fontsize_ticks=8,
    fontweight_xticks="normal",
    fontweight_yticks="normal",

    # =========================
    # Spines
    # =========================
    show_spines=True,
    spine_width=2,

    # =========================
    # Y limits
    # =========================
    ylim=None,   # (ymin, ymax) o None para automático

    # =========================
    # Highlight genes
    # =========================
    highlight_genes=None,
    highlight_color="green",
    highlight_size=30,
    highlight_edge=True,
    highlight_edgecolor="black",
    highlight_edgewidth=1,
    show_labels=True,
    highlight_fontweight="bold",
    highlight_fontstyle="normal",
    highlight_labelcolor=None,
    highlight_label_xpos=None,
    highlight_label_ypos=None,
    highlight_label_ha="right",
    highlight_label_va="bottom",

    # =========================
    # Figure saving and export
    # =========================
    figsize=(3.35, 3.35),
    dpi=600,
    save_path=None
):
    """
        Volcano plot with option to highlight specific points (genes or other entities) and full control over text weights, colors, labels, and axes aesthetics.
        
        This function creates a volcano-style scatter plot where points are colored according to significance groups, optional cut lines are drawn for fold change and p-value thresholds, and selected points can be highlighted with customized colors, sizes, and labels. 
        Although parameter names mention "genes," this function can be used with any dataset that has numeric x and y values and a categorical significance column.
        
        Data
        ----
        df : pandas.DataFrame
            Dataframe containing the data to plot.
        
        x : str, default="log2FoldChange"
            Column in df to use for the x-axis (fold change or other numeric measure).
        
        y : str, default="minus_log10_padj"
            Column in df to use for the y-axis (-log10 p-value or other numeric measure).
        
        significance_col : str, default="significance"
            Column in df defining the significance groups (categorical).
        
        gene_col : str or None, default=None
            Column in df containing labels for points (e.g., gene names). If None, df index is used.
        
        Groups and Colors
        -----------------
        palette : dict, default={"NS": "gray", "Up": "red", "Down": "blue"}
            Dictionary mapping significance categories to colors.
        
        point_size : int or float, default=10
            Size of scatter points for all categories.
        
        alpha : float, default=0.4
            Transparency for scatter points (0 to 1).
        
        Cut Lines
        ---------
        fc_threshold : float, default=0.5
            Fold-change threshold for vertical lines.
        
        pval_threshold : float, default=0.05
            P-value threshold for horizontal line.
        
        cut_linewidth : float, default=1
            Line width for threshold lines.
        
        cut_linestyle : str, default="--"
            Line style for threshold lines.
        
        Axis Labels
        -----------
        xlabel : str, default="log2 Fold Change"
        ylabel : str, default="-log10(padj)"
        show_axis_labels : bool, default=True
        fontsize_labels : int, default=9
        fontweight_xlabel : str, default="normal"
        fontweight_ylabel : str, default="normal"
        
        Title
        -----
        title : str or None, default=None
        show_title : bool, default=True
        fontsize_title : int, default=10
        fontweight_title : str, default="bold"
        
        Legend
        ------
        show_legend : bool, default=True
        legend_title : str, default="Significance"
        fontsize_legend : int, default=8
        fontsize_legend_title : int, default=9
        fontweight_legend_title : str, default="bold"
        fontweight_legend_labels : str, default="normal"
        fontstyle_legend_title : str, default="normal"
        fontstyle_legend_labels : str, default="normal"
        legend_loc : str, default="upper right"
        legend_frame : bool, default=False
        legend_edgecolor : str, default="black"
        legend_linewidth : float, default=1
        legend_facecolor : str, default="white"
        legend_alpha : float, default=1
        
        Ticks
        -----
        show_ticks : bool, default=True
        tick_width : float, default=2
        tick_length : float, default=4
        fontsize_ticks : int, default=8
        fontweight_xticks : str, default="normal"
        fontweight_yticks : str, default="normal"
        
        Spines
        ------
        show_spines : bool, default=True
        spine_width : float, default=2
        
        Y Limits
        --------
        ylim : tuple or list of 2 floats, default=None
            Y-axis limits as (ymin, ymax). If None, limits are automatic.
        
        Highlight Genes / Points
        -----------------------
        highlight_genes : iterable or None, default=None
            List of points (genes or other entities) to highlight.
        
        highlight_color : str or list of str
            Color or list of colors matching highlight_genes order.
        highlight_size : float, default=30
        highlight_edge : bool, default=True
        highlight_edgecolor : str, default="black"
        highlight_edgewidth : float, default=1
        show_labels : bool, default=True
        highlight_fontweight : str, default="bold"
        highlight_fontstyle : str, default="normal"
        highlight_labelcolor : str or None, default=None
        highlight_label_xpos : dict, list, or None, default=None
        highlight_label_ypos : dict, list, or None, default=None
        highlight_label_ha : str, default="right"
        highlight_label_va : str, default="bottom"
        
        Figure Saving and Export
        ------------------------
        figsize : tuple of 2 floats, default=(3.35, 3.35)
        dpi : int or float, default=600
        save_path : str or None, default=None
            File path to save the figure. If None, the figure is not saved.
        
        Returns
        -------
        None
            The function displays the volcano plot and optionally saves it to disk.
        
        Notes
        -----
        - The function is general and can be used with any dataset that has numeric x and y columns and a categorical column for grouping.
        - Highlighting can be customized with position adjustments using dicts or lists for precise label placement.
        - If gene_col is None, the function uses the index of the dataframe for highlighting and labeling.
    """
    # =========================
    # Input validations
    # =========================
    
    # Validate df
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    # Validate x, y, and significance_col exist in df
    for col in [x, y, significance_col]:
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe columns")
    
    # Validate gene_col
    if gene_col is not None and gene_col not in df.columns:
        raise ValueError(f"{gene_col} not found in dataframe columns")
    
    # Validate palette
    if not isinstance(palette, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in palette.items()):
        raise TypeError("palette must be a dictionary with string keys and string color values")
    
    # Validate thresholds
    if not isinstance(fc_threshold, (int, float)):
        raise TypeError("fc_threshold must be numeric")
    if not isinstance(pval_threshold, (int, float)):
        raise TypeError("pval_threshold must be numeric")
    
    # Validate highlight_genes
    if highlight_genes is not None:
        if not hasattr(highlight_genes, "__iter__"):
            raise TypeError("highlight_genes must be iterable (list, set, tuple, etc.)")
    
    # Validate figure size and dpi
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple/list of length 2 (width, height)")
    if not isinstance(dpi, (int, float)):
        raise TypeError("dpi must be numeric")
    
    # Validate save_path
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string")
    
    # -------------------------
    # Figure
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------
    # Scatter by category
    # -------------------------
    for group, color in palette.items():
        subset = df[df[significance_col] == group]
        ax.scatter(subset[x], subset[y], s=point_size, alpha=alpha, color=color, label=group)

    # -------------------------
    # Cut lines
    # -------------------------
    ax.axvline(fc_threshold, linestyle=cut_linestyle, linewidth=cut_linewidth, color="black")
    ax.axvline(-fc_threshold, linestyle=cut_linestyle, linewidth=cut_linewidth, color="black")
    ax.axhline(-np.log10(pval_threshold), linestyle=cut_linestyle, linewidth=cut_linewidth, color="black")

    # -------------------------
    # Highlight genes
    # -------------------------
    if highlight_genes is not None:
    
        if gene_col is not None:
            subset = df[df[gene_col].isin(highlight_genes)]
            gene_names = subset[gene_col].tolist()
        else:
            subset = df[df.index.isin(highlight_genes)]
            gene_names = subset.index.tolist()
    
        edgecolor = highlight_edgecolor if highlight_edge else None
    
        # asegurar lista de colores
        if isinstance(highlight_color, (list, tuple)):
            colors = list(highlight_color)
        else:
            colors = [highlight_color] * len(gene_names)
    
        ax.scatter(
            subset[x], subset[y],
            s=highlight_size,
            c=colors[:len(subset)],
            alpha=1,
            edgecolor=edgecolor,
            linewidth=highlight_edgewidth if highlight_edge else 0,
            zorder=5
        )
    
        if show_labels:
            for i, (gene, xval, yval) in enumerate(zip(gene_names, subset[x], subset[y])):
                xpos = xval
                ypos = yval
    
                if isinstance(highlight_label_xpos, dict):
                    xpos = highlight_label_xpos.get(gene, xval)
                elif isinstance(highlight_label_xpos, list):
                    xpos = highlight_label_xpos[i] if i < len(highlight_label_xpos) else xval
    
                if isinstance(highlight_label_ypos, dict):
                    ypos = highlight_label_ypos.get(gene, yval)
                elif isinstance(highlight_label_ypos, list):
                    ypos = highlight_label_ypos[i] if i < len(highlight_label_ypos) else yval
    
                # color label
                if highlight_labelcolor:
                    if isinstance(highlight_labelcolor, (list, tuple)):
                        txt_color = highlight_labelcolor[i]
                    else:
                        txt_color = highlight_labelcolor
                else:
                    txt_color = colors[i]
    
                ax.text(
                    xpos, ypos, gene,
                    fontsize=fontsize_labels,
                    ha=highlight_label_ha,
                    va=highlight_label_va,
                    fontweight=highlight_fontweight,
                    fontstyle=highlight_fontstyle,
                    color=txt_color
                )
    # -------------------------
    # Axis labels
    # -------------------------
    if show_axis_labels:
        ax.set_xlabel(xlabel, fontsize=fontsize_labels, fontweight=fontweight_xlabel)
        ax.set_ylabel(ylabel, fontsize=fontsize_labels, fontweight=fontweight_ylabel)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # -------------------------
    # Title
    # -------------------------
    if title and show_title:
        ax.set_title(title, fontsize=fontsize_title, fontweight=fontweight_title)

    # -------------------------
    # Legend
    # -------------------------
    if show_legend:
        legend = ax.legend(
            title=legend_title,
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend_title,
            loc=legend_loc,
            frameon=legend_frame
        )
        if legend:
            legend.get_title().set_fontweight(fontweight_legend_title)
            legend.get_title().set_fontstyle(fontstyle_legend_title)
            
            for text in legend.get_texts():
                text.set_fontweight(fontweight_legend_labels)
                text.set_fontstyle(fontstyle_legend_labels)
                
            if legend_frame:
                frame = legend.get_frame()
                frame.set_edgecolor(legend_edgecolor)
                frame.set_linewidth(legend_linewidth)
                frame.set_facecolor(legend_facecolor)
                frame.set_alpha(legend_alpha)

    else:
        if ax.get_legend():
            ax.get_legend().remove()

    # -------------------------
    # Ticks
    # -------------------------
    if show_ticks:
        ax.tick_params(axis="x", which="major", labelsize=fontsize_ticks, width=tick_width, length=tick_length)
        ax.tick_params(axis="y", which="major", labelsize=fontsize_ticks, width=tick_width, length=tick_length)
        for label in ax.get_xticklabels():
            label.set_fontweight(fontweight_xticks)
        for label in ax.get_yticklabels():
            label.set_fontweight(fontweight_yticks)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # -------------------------
    # Spines
    # -------------------------
    for spine in ax.spines.values():
        spine.set_visible(show_spines)
        spine.set_linewidth(spine_width)

    # -------------------------
    # Y limits
    # -------------------------
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    # -------------------------
    # Figure saving
    # -------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_gsea_with_genes(

    # =========================
    # Data
    # =========================
    ranking,
    res,
    term,

    # =========================
    # Highlight genes
    # =========================
    genes=None,
    gene_sets=None,
    show_lines=True,
    show_gene_labels=True,
    show_legend=True,

    # =========================
    # Gene lines
    # =========================
    default_line_color="red",
    default_line_style="--",
    default_line_width=2,
    default_line_alpha=0.8,

    # =========================
    # Gene labels
    # =========================
    gene_label_rotation=90,
    gene_label_color="red",
    sync_label_color_with_line =False,
    gene_label_size=8,
    gene_label_offset_pts=10,
    gene_label_y=0.0,
    gene_label_va="bottom",
    default_label_alpha=1.0,
    default_label_weight="normal",
    default_label_style="normal",

    # =========================
    # Stats
    # =========================
    show_stats=True,
    show_NES=True,
    show_pval=True,
    show_FDR=True,
    stats_text_size=9,
    stats_box=False,
    stats_box_edgecolor="black",
    stats_box_facecolor="white",
    stats_box_linewidth=0.8,
    stats_box_alpha=0.8,
    stats_x=0.02,
    stats_y=0.98,
    stats_ha="left",
    stats_va="top",

    # =========================
    # Title
    # =========================
    show_title=True,
    title_text=None,
    title_size=12,
    title_bold=False,

    # =========================
    # Axis labels
    # =========================
    show_es_ylabel=True,
    es_ylabel=None,
    es_ylabel_size=10,
    es_ylabel_weight="normal",

    show_rank_ylabel=True,
    rank_ylabel=None,
    rank_ylabel_size=10,
    rank_ylabel_weight="normal",

    show_rank_xlabel=True,
    rank_xlabel=None,
    rank_xlabel_size=10,
    rank_xlabel_weight="normal",

    # =========================
    # Ticks
    # =========================
    show_es_ticks=True,
    show_rank_ticks=True,
    es_tick_label_size=9,
    rank_tick_label_size=9,
    tick_width=1.2,
    tick_length=4,

    # =========================
    # Spines
    # =========================
    visible_spines=("left", "bottom", "right", "top"),
    spine_linewidth=1.2,

    # =========================
    # Figure size
    # =========================
    figsize=(6, 4),

    # =========================
    # Figure saving
    # =========================
    save_path=None,
    dpi=600
):
    
    """
        Plot a GSEA (Gene Set Enrichment Analysis) result with optional highlighting of specific items.
    
        This function creates a GSEA plot using gseapy's gseaplot function and adds the ability
        to highlight specific elements (genes, proteins, metabolites, or any indexed identifiers)
        with custom lines and labels. Although the function mentions "genes", it works for any
        identifiers that match the index of the `ranking` Series.
    
        Data
        ----
        ranking : pd.Series
            Indexed ranking metric for all items (e.g., genes, proteins, metabolites). Index values
            will be used to match the items to highlight.
        res : dict
            Dictionary with GSEA results compatible with gseapy.plot.gseaplot.
        term : str
            Name of the gene set or term to display in the plot.
    
        Highlight items
        ---------------
        genes : dict, list, str, or None, default=None
            Items to highlight in the plot. Can be:
            - dict: keys are items and values are dicts with line/label customization.
            - list: list of items to highlight with default styling.
            - str: single item to highlight.
        gene_sets : dict or None, default=None
            Optional dictionary defining groups of items for legend purposes. Keys are group names,
            values are lists of items.
        show_lines : bool, default=True
            Whether to draw vertical lines for highlighted items.
        show_gene_labels : bool, default=True
            Whether to annotate highlighted items with text labels.
        show_legend : bool, default=True
            Whether to show a legend for highlighted items.
    
        Gene lines
        ----------
        default_line_color : str, default="red"
            Color of highlight lines if not specified individually.
        default_line_style : str, default="--"
            Line style of highlight lines.
        default_line_width : float, default=2
            Line width of highlight lines.
        default_line_alpha : float, default=0.8
            Transparency of highlight lines.
    
        Gene labels
        -----------
        gene_label_rotation : float, default=90
            Rotation angle of highlight labels.
        gene_label_color : str, default="red"
            Default text color for labels.
        sync_label_color_with_line : bool, default=True
            Whether label text should match line color by default.
        gene_label_size : float, default=8
            Font size of labels.
        gene_label_offset_pts : float, default=10
            Horizontal offset of label text in points.
        gene_label_y : float, default=0.0
            Y position of label relative to axis fraction.
        gene_label_va : str, default="bottom"
            Vertical alignment of labels.
        default_label_alpha : float, default=1.0
            Transparency of label text.
        default_label_weight : str, default="normal"
            Font weight of labels.
        default_label_style : str, default="normal"
            Font style of labels.
    
        Stats
        -----
        show_stats : bool, default=True
            Show enrichment statistics (NES, p-value, FDR) on the plot.
        show_NES, show_pval, show_FDR : bool, default=True
            Controls which statistics to display.
        stats_text_size : float, default=9
            Font size of stats text.
        stats_box : bool, default=False
            Whether to show a background box for stats.
        stats_box_edgecolor : str, default="black"
            Edge color of stats box.
        stats_box_facecolor : str, default="white"
            Fill color of stats box.
        stats_box_linewidth : float, default=0.8
            Line width of stats box.
        stats_box_alpha : float, default=0.8
            Transparency of stats box.
        stats_x, stats_y : float, default=(0.02,0.98)
            Position of stats text in axes fraction.
        stats_ha, stats_va : str, default=("left","top")
            Horizontal and vertical alignment of stats text.
    
        Title
        -----
        show_title : bool, default=True
            Whether to display a title.
        title_text : str or None, default=None
            Custom title text. If None, uses `term`.
        title_size : float, default=12
            Font size of the title.
        title_bold : bool, default=False
            Whether to render title in bold.
    
        Axis labels
        -----------
        show_es_ylabel, show_rank_ylabel, show_rank_xlabel : bool, default=True
            Whether to display the respective axis labels.
        es_ylabel, rank_ylabel, rank_xlabel : str or None, default=None
            Custom labels for axes.
        es_ylabel_size, rank_ylabel_size, rank_xlabel_size : float, default=10
            Font size of axis labels.
        es_ylabel_weight, rank_ylabel_weight, rank_xlabel_weight : str, default="normal"
            Font weight of axis labels.
    
        Ticks
        -----
        show_es_ticks, show_rank_ticks : bool, default=True
            Whether to display tick marks for respective axes.
        es_tick_label_size, rank_tick_label_size : float, default=9
            Font size of tick labels.
        tick_width, tick_length : float, default=(1.2, 4)
            Width and length of tick marks.
    
        Spines
        ------
        visible_spines : tuple or list, default=("left","bottom","right","top")
            Which spines to display.
        spine_linewidth : float, default=1.2
            Width of axis spines.
    
        Figure
        ------
        figsize : tuple of float, default=(6,4)
            Figure size (width, height) in inches.
    
        Figure saving
        -------------
        save_path : str or None, default=None
            Path to save the figure. If None, figure is not saved.
        dpi : float, default=600
            Resolution of saved figure.
    
        Returns
        -------
        None
            Displays the plot and optionally saves it to `save_path`.
    
        Notes
        -----
        - `genes` can actually contain any type of identifier present in `ranking.index`, not limited to transcriptomic genes.
        - Custom styling for highlighted items can be passed via a dictionary for each item.
        - Requires gseapy.plot.gseaplot to generate the underlying GSEA plot.
        - Input validation is performed for type checking of key parameters.
    """
    # =========================
    # Input validations
    # =========================
    
    # Validate ranking
    if not isinstance(ranking, pd.Series):
        raise TypeError("ranking must be a pandas Series with elements as index")
    
    # Validate res
    if not isinstance(res, dict):
        raise TypeError("res must be a dictionary with GSEA results compatible with gseaplot")
    
    # Validate term
    if not isinstance(term, str):
        raise TypeError("term must be a string")
    
    # Validate genes
    if genes is not None and not isinstance(genes, (dict, list, str)):
        raise TypeError("genes must be a dict, list, string, or None")
    
    # Validate gene_sets
    if gene_sets is not None and not isinstance(gene_sets, dict):
        raise TypeError("gene_sets must be a dictionary or None")
    
    # Validate booleans
    for flag_name in [
        "show_lines", "show_gene_labels", "show_legend",
        "show_stats", "show_NES", "show_pval", "show_FDR",
        "show_title",
        "show_es_ylabel", "show_rank_ylabel", "show_rank_xlabel",
        "show_es_ticks", "show_rank_ticks"
    ]:
        if not isinstance(locals()[flag_name], bool):
            raise TypeError(f"{flag_name} must be boolean")
    
    # Validate numeric parameters
    for num_param in [
        "default_line_width", "default_line_alpha",
        "gene_label_size", "gene_label_offset_pts", "gene_label_y",
        "stats_text_size", "stats_box_linewidth", "stats_box_alpha",
        "title_size", "es_ylabel_size", "rank_ylabel_size", "rank_xlabel_size",
        "es_tick_label_size", "rank_tick_label_size", "tick_width", "tick_length",
        "dpi"
    ]:
        if not isinstance(locals()[num_param], (int, float)):
            raise TypeError(f"{num_param} must be numeric")
    
    # Validate visible_spines
    if not isinstance(visible_spines, (list, tuple)):
        raise TypeError("visible_spines must be a list or tuple of spine positions (e.g., ['left','bottom'])")
    
    # Validate figsize
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple/list of length 2 (width, height)")
    
    # Validate save_path
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string or None")
    
    spine_style = {
        "axes.linewidth": spine_linewidth,
        "axes.spines.left": "left" in visible_spines,
        "axes.spines.bottom": "bottom" in visible_spines,
        "axes.spines.right": "right" in visible_spines,
        "axes.spines.top": "top" in visible_spines,
    }

    # Normalize genes
    if genes is None:
        genes_dict = {}
    elif isinstance(genes, dict):
        genes_dict = genes
    elif isinstance(genes, list):
        genes_dict = {g: {} for g in genes}
    elif isinstance(genes, str):
        genes_dict = {genes: {}}
    else:
        raise ValueError("genes debe ser dict, lista, string o None")

    # -------------------------------------------------
    # GSEA
    # -------------------------------------------------
    with mpl.rc_context(spine_style):

        axes = gseaplot(
            rank_metric=ranking,
            term=term,
            **res,
            show_stats=True,
            figsize=figsize
        )

    ax_rank = axes[0]
    ax_es = axes[-1]
    fig = ax_es.figure

    # -------------------------------------------------
    # Stats
    # -------------------------------------------------
    stat_texts = [
        txt for txt in ax_es.texts
        if any(k in txt.get_text() for k in ["NES", "Pval", "FDR"])
    ]

    if show_stats and stat_texts:

        full_text = "\n".join(txt.get_text() for txt in stat_texts)

        for txt in stat_texts:
            txt.remove()

        lines = full_text.split("\n")
        order = ["NES", "Pval", "FDR"]
        flags = {"NES": show_NES, "Pval": show_pval, "FDR": show_FDR}

        selected_lines = []
        for key in order:
            if flags[key]:
                for l in lines:
                    if key.lower() in l.lower():
                        selected_lines.append(l)
                        break

        stats_str = "\n".join(selected_lines)

        if stats_str:

            bbox_kwargs = None
            if stats_box:
                face_rgba = mcolors.to_rgba(stats_box_facecolor, stats_box_alpha)
                bbox_kwargs = dict(
                    boxstyle="round,pad=0.3",
                    facecolor=face_rgba,
                    edgecolor=stats_box_edgecolor,
                    linewidth=stats_box_linewidth
                )

            ax_es.annotate(
                stats_str,
                xy=(stats_x, stats_y),
                xycoords="axes fraction",
                ha=stats_ha,
                va=stats_va,
                fontsize=stats_text_size,
                bbox=bbox_kwargs
            )
    else:
        for txt in stat_texts:
            txt.remove()

    # -------------------------------------------------
    # Title
    # -------------------------------------------------
    fig.suptitle("")
    if show_title:
        final_title = term if title_text is None else title_text
        weight = "bold" if title_bold else "normal"
        fig.suptitle(final_title, fontsize=title_size, weight=weight)

    # -------------------------------------------------
    # Labels
    # -------------------------------------------------
    if show_es_ylabel:
        if es_ylabel is not None:
            ax_es.set_ylabel(es_ylabel)
        ax_es.yaxis.label.set_size(es_ylabel_size)
        ax_es.yaxis.label.set_weight(es_ylabel_weight)
    else:
        ax_es.set_ylabel("")

    if show_rank_ylabel:
        if rank_ylabel is not None:
            ax_rank.set_ylabel(rank_ylabel)
        ax_rank.yaxis.label.set_size(rank_ylabel_size)
        ax_rank.yaxis.label.set_weight(rank_ylabel_weight)
    else:
        ax_rank.set_ylabel("")

    if show_rank_xlabel:
        if rank_xlabel is not None:
            ax_rank.set_xlabel(rank_xlabel)
        ax_rank.xaxis.label.set_size(rank_xlabel_size)
        ax_rank.xaxis.label.set_weight(rank_xlabel_weight)
    else:
        ax_rank.set_xlabel("")

    # -------------------------------------------------
    # Ticks
    # -------------------------------------------------
    if show_es_ticks:
        ax_es.tick_params(axis="both", labelsize=es_tick_label_size,
                          width=tick_width, length=tick_length)
    else:
        ax_es.set_xticks([])
        ax_es.set_yticks([])

    if show_rank_ticks:
        ax_rank.tick_params(axis="both", labelsize=rank_tick_label_size,
                            width=tick_width, length=tick_length)
    else:
        ax_rank.set_xticks([])
        ax_rank.set_yticks([])

    # -------------------------------------------------
    # Highlight genes
    # -------------------------------------------------
    for gene, params in genes_dict.items():

        if gene not in ranking.index:
            continue

        pos = ranking.index.get_loc(gene)

        # Lines
        line_color = params.get("line_color", default_line_color)
        line_style = params.get("line_style", default_line_style)
        line_width = params.get("line_width", default_line_width)
        line_alpha = params.get("line_alpha", default_line_alpha)

        # Text
        if sync_label_color_with_line:
            text_color = params.get("text_color", line_color)
        else:
            text_color = params.get("text_color", gene_label_color)
        
        text_size = params.get("text_size", gene_label_size)
        xoffset = params.get("xoffset_pts", gene_label_offset_pts)
        text_alpha = params.get("text_alpha", default_label_alpha)
        text_weight = params.get("text_weight", default_label_weight)
        text_style = params.get("text_style", default_label_style)

        if show_lines:
            ax_es.axvline(
                pos,
                color=line_color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=line_alpha
            )

        if show_gene_labels:
            ax_es.annotate(
                gene,
                xy=(pos, gene_label_y),
                xytext=(xoffset, 0),
                textcoords="offset points",
                rotation=gene_label_rotation,
                rotation_mode="anchor",
                fontsize=text_size,
                color=text_color,
                fontweight=text_weight,
                fontstyle=text_style,
                alpha=text_alpha,
                ha="left" if xoffset >= 0 else "right",
                va=gene_label_va
            )

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
    if show_legend and genes_dict:

        handles = []

        if gene_sets:
            for set_name, gene_list in gene_sets.items():

                if not gene_list:
                    continue

                first_gene = gene_list[0]
                params = genes_dict.get(first_gene, {})

                handles.append(
                    mlines.Line2D(
                        [], [],
                        color=params.get("line_color", default_line_color),
                        linestyle=params.get("line_style", default_line_style),
                        linewidth=params.get("line_width", default_line_width),
                        label=set_name
                    )
                )
        else:
            for gene, params in genes_dict.items():
                handles.append(
                    mlines.Line2D(
                        [], [],
                        color=params.get("line_color", default_line_color),
                        linestyle=params.get("line_style", default_line_style),
                        linewidth=params.get("line_width", default_line_width),
                        label=gene
                    )
                )

        ax_es.legend(
            handles=handles,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=9
        )

    # -------------------------------------------------
    # Figure saving and export
    # -------------------------------------------------
    fig.subplots_adjust(left=0.1, right=0.85,
                        top=0.9, bottom=0.15, hspace=0.3)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()