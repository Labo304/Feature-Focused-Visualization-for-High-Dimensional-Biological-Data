import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
import pandas as pd
import numpy as np

from matplotlib_venn import venn2

def plot_barplot(
    # =========================
    # Data
    # =========================
    df,
    x,
    y,
    hue=None,
    palette=None,
    use_seaborn=True,

    # =========================
    # Horizontal line at 0
    # =========================
    show_zero_line=True,
    zero_line_color="black",
    zero_line_width=1.5,
    zero_line_style="-",

    # =========================
    # Axis labels
    # =========================
    xlabel=None,
    ylabel=None,
    show_xlabel=True,
    show_ylabel=True,
    label_size=9,
    label_fontweight="normal",      

    # =========================
    # Title
    # =========================
    title=None,
    show_title=True,
    title_size=10,
    title_fontweight="normal",      

    # =========================
    # Bars (line art)
    # =========================
    show_bar_edge=True,
    bar_edgecolor="black",
    bar_linewidth=2,
    bar_width=0.8,
    dodge=True,

    # =========================
    # Ticks
    # =========================
    show_xticks=True,
    show_yticks=True,
    tick_label_size=8,
    tick_fontweight="normal",      
    tick_width=2,
    tick_length=6,
    xtick_label_rotation=None,
    
    # =========================
    # Spines (frame)
    # =========================
    show_spines=True,
    spine_width=2,

    # =========================
    # Legend
    # =========================
    show_legend=True,
    legend_loc="best",             
    legend_bbox_to_anchor=None,     
    legend_fontsize=8,
    legend_fontweight="normal",
    legend_fontstyle="normal",
    legend_title=None,
    legend_title_size=None,
    legend_title_fontweight="normal",
    legend_title_fontstyle="normal",
    legend_show_box=True,
    legend_edgecolor="black",
    legend_edgewidth=1,
    legend_facecolor="white",
    legend_alpha=1.0,
       
    # =========================
    # Figure and export
    # =========================
    figsize=(3.35, 3.0),
    dpi=600,
    save_path=None
):
    """
        Generate a highly customizable barplot with publication-quality aesthetics.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing the data.
        x : str
            Column name for x-axis categories.
        y : str
            Column name for y-axis values.
        hue : str, optional
            Column used for grouping.
        palette : str or list, optional
            Color palette for bars.
        use_seaborn : bool, default=True
            Whether to use seaborn backend for plotting.
        
        Zero line
        ---------
        show_zero_line : bool, default=True
            If True, draws horizontal line at y=0.
        zero_line_color : str, default="black"
            Color of zero reference line.
        zero_line_width : float, default=1.5
            Line width of zero reference line.
        zero_line_style : str, default="-"
            Line style of zero reference line.
        
        Axis labels
        -----------
        xlabel, ylabel : str, optional
            Axis labels.
        show_xlabel, show_ylabel : bool, default=True
            Toggle visibility of axis labels.
        label_size : int, default=9
            Font size of axis labels.
        label_fontweight : str, default="normal"
            Font weight of axis labels.
        
        Title
        -----
        title : str, optional
            Plot title.
        show_title : bool, default=True
            Whether to display title.
        title_size : int, default=10
            Title font size.
        title_fontweight : str, default="normal"
            Title font weight.
        
        Bars
        ----
        show_bar_edge : bool, default=True
            Whether to draw bar borders.
        bar_edgecolor : str, default="black"
            Color of bar borders.
        bar_linewidth : float, default=2
            Width of bar borders.
        bar_width : float, default=0.8
            Width of bars.
        dodge : bool, default=True
            Separate bars when hue is used.
        
        Ticks
        -----
        show_xticks, show_yticks : bool, default=True
            Toggle tick visibility.
        tick_label_size : int, default=8
            Font size of tick labels.
        tick_fontweight : str, default="normal"
            Font weight of tick labels.
        tick_width : float, default=2
            Tick line width.
        tick_length : float, default=6
            Tick length.
        xtick_label_rotation : float, optional
            Rotation angle for x tick labels.
        
        Spines
        ------
        show_spines : bool, default=True
            Whether to display plot frame.
        spine_width : float, default=2
            Width of spines.
        
        Legend
        ------
        show_legend : bool, default=True
            Whether to display legend.
        legend_loc : str, default="best"
            Legend location.
        legend_bbox_to_anchor : tuple, optional
            Advanced legend positioning.
        legend_fontsize : int, default=8
            Legend text size.
        legend_fontweight : str, default="normal"
            Legend text weight.
        legend_fontstyle : str, default="normal"
            Legend text style.
        legend_title : str, optional
            Legend title.
        legend_title_size : int, optional
            Legend title size.
        legend_title_fontweight : str, default="normal"
            Legend title weight.
        legend_title_fontstyle : str, default="normal"
            Legend title style.
        legend_show_box : bool, default=True
            Whether to draw legend frame.
        legend_edgecolor : str, default="black"
            Legend frame edge color.
        legend_edgewidth : float, default=1
            Legend frame line width.
        legend_facecolor : str, default="white"
            Legend background color.
        legend_alpha : float, default=1.0
            Legend transparency.
        
        Figure and export
        -----------------
        figsize : tuple, default=(3.35, 3.0)
            Figure size in inches.
        dpi : int, default=600
            Output resolution.
        save_path : str, optional
            If provided, saves figure to this path.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the plot.
        
        Notes
        -----
        Designed for high-control scientific visualization and publication-ready figures.
    """

    fig, ax = plt.subplots(figsize=figsize)
    
    # =========================
    # Input validation
    # =========================
    
    # type check
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame")
    
    # column checks
    if x not in df.columns:
        raise ValueError(f"{x} not found in dataframe columns")
    
    if y not in df.columns:
        raise ValueError(f"{y} not found in dataframe columns")
    
    if hue is not None and hue not in df.columns:
        raise ValueError(f"{hue} not found in dataframe columns")

    # -------------------------
    # Automatic pallet
    # -------------------------
    if palette is None and hue is not None and use_seaborn:
        n_colors = df[hue].nunique()
        palette = sns.color_palette("bwr", n_colors=n_colors)

    # =========================
    # Barplot
    # =========================
    if use_seaborn:
        sns_bar = sns.barplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            width=bar_width,
            dodge=dodge,
            ax=ax
        )
        bars = ax.patches
    
    else:
        values = df[y].values
        labels = df[x].values
    
        if hue is None:
            # Caso simple
            colors = (
                ["steelblue" if v < 0 else "indianred" for v in values]
                if palette is None else palette
            )
    
            bars = ax.bar(
                labels,
                values,
                color=colors,
                width=bar_width
            )
    
        else:           
                
            unique_groups = df[hue].unique()
            n_groups = len(unique_groups)
    
            x_pos = np.arange(len(df[x].unique()))
    
            total_width = bar_width
            single_width = total_width / n_groups
    
            bars = []
            for i, group in enumerate(unique_groups):
                subset = df[df[hue] == group]
                offset = (i - n_groups / 2) * single_width + single_width / 2
    
                b = ax.bar(
                    x_pos + offset,
                    subset[y].values,
                    width=single_width,
                    label=group,
                    color=palette[i] if palette is not None else None
                )
                bars.extend(b)
    
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df[x].unique())
            
    # =========================
    # Bar borders
    # =========================
    if show_bar_edge and bar_linewidth > 0:
        for bar in bars:
            try:
                bar.set_edgecolor(bar_edgecolor)
                bar.set_linewidth(bar_linewidth)
            except Exception:
                pass
    else:
        for bar in bars:
            try:
                bar.set_linewidth(0)
            except Exception:
                pass
    # =========================
    # X-axis ticks
    # =========================
    if show_xticks:
        ax.tick_params(
            axis="x",
            labelsize=tick_label_size,
            width=tick_width,
            length=tick_length
        )
    
        for label in ax.get_xticklabels():
            label.set_fontweight(tick_fontweight)
    
            if xtick_label_rotation is not None:
                label.set_rotation(xtick_label_rotation)   
               
            # Automatic alignment recommended
                if xtick_label_rotation == 0:
                    label.set_ha("center")
                elif xtick_label_rotation > 0:
                    label.set_ha("right")
                else:
                    label.set_ha("left")
    else:
        ax.set_xticks([])
    
    # =========================
    # Etiqueta eje Y
    # =========================
    if show_ylabel:
        ax.set_ylabel(
            ylabel if ylabel else y,
            fontsize=label_size,
            fontweight=label_fontweight
        )
    else:
        ax.set_ylabel("")
        
    # =========================
    # Y axis label
    # =========================
    if show_xlabel:
        ax.set_xlabel(
            xlabel if xlabel else x,
            fontsize=label_size,
            fontweight=label_fontweight
        )
    else:
        ax.set_xlabel(None)

    # =========================
    # Ticks
    # =========================
   
    if show_yticks:
        ax.tick_params(axis="y",
                       labelsize=tick_label_size,
                       width=tick_width,
                       length=tick_length)
        for label in ax.get_yticklabels():
            label.set_fontweight(tick_fontweight)
    else:
        ax.set_yticks([])

    # =========================
    # Spines
    # =========================
    if show_spines:
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(spine_width)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # =========================
    # Legend
    # =========================
    if show_legend and hue is not None:
        leg = ax.legend(
            title=legend_title if legend_title else "",
            fontsize=legend_fontsize,
            title_fontsize=legend_title_size if legend_title_size else legend_fontsize,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=legend_show_box
        )

        # Control fontweight texts
        for text in leg.get_texts():
            text.set_fontweight(legend_fontweight)
            text.set_fontstyle(legend_fontstyle)

        if leg.get_title():
            leg.get_title().set_fontweight(legend_title_fontweight)
            leg.get_title().set_fontstyle(legend_title_fontstyle)
            
        if legend_show_box:
            frame = leg.get_frame()
            frame.set_facecolor(to_rgba(legend_facecolor, legend_alpha))
            frame.set_edgecolor(legend_edgecolor)
            frame.set_linewidth(legend_edgewidth)
    else:
        if ax.get_legend():
            ax.get_legend().remove()

    # =========================
    # Title
    # =========================
    if show_title:

        # Define automatic title
        if title is None:
            default_title = f"{y} vs {x}" if x and y else "Barplot"
            title_text = default_title
        else:
            title_text = title
    
        ax.set_title(
            title_text,
            fontsize=title_size,
            fontweight=title_fontweight
        )
        
    # =========================
    # Horizontal line at 0
    # =========================
    if show_zero_line:
        ax.axhline(
            y=0,
            color=zero_line_color,
            linewidth=zero_line_width,
            linestyle=zero_line_style,
            zorder=0  
        )
        
    # =========================
    # Figure saving
    # =========================
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()





def plot_venn2(
    # =========================
    # Data
    # =========================
    set1,
    set2,

    # =========================
    # Labels
    # =========================
    labels=("Set 1", "Set 2"),
    show_labels=True,
    font_size=9,
    label_fontweight="normal",
    label_fontstyle="normal",

    # =========================
    # Title
    # =========================
    title=None,
    show_title=True,
    fontsize_title=10,
    title_fontweight="normal",
    title_fontstyle="normal",

    # =========================
    # Venn diagram aesthetics
    # =========================
    colors=None,
    intersection_color=None,
    dibujar_bordes=True,
    line_width=2,
    alpha=0.6,

    # =========================
    # Axes
    # =========================
    show_axes=False,
    show_spines=False,
    spine_width=2,

    # =========================
    # Figure saving and export
    # =========================
    figsize=(3.35, 3.35),
    dpi=600,
    save_path=None
):
    """
        Plots a 2-set Venn diagram with customizable aesthetics, labels, axes, and figure export options.
    
        This function is designed to visually compare two sets, highlighting their unique and shared elements.
        It works with any iterable data (lists, tuples, sets, pandas Series, etc.), not restricted to transcriptomic data.
    
        Parameters
        ----------
        Data ----------
        set1 : iterable
            First dataset or group of elements.
        set2 : iterable
            Second dataset or group of elements.
    
        Labels ----------
        labels : tuple of str, default=("Set 1", "Set 2")
            Labels for the two sets.
        show_labels : bool, default=True
            Whether to display the set labels on the diagram.
        font_size : int or float, default=9
            Font size for set labels and subset counts.
        label_fontweight : str, default="normal"
            Font weight for set labels.
        label_fontstyle : str, default="normal"
            Font style for set labels (e.g., "normal", "italic").
    
        Title ----------
        title : str, optional
            Title of the diagram. If None, defaults to "Venn Diagram".
        show_title : bool, default=True
            Whether to display the title.
        fontsize_title : int or float, default=10
            Font size of the title.
        title_fontweight : str, default="normal"
            Font weight of the title.
        title_fontstyle : str, default="normal"
            Font style of the title.
    
        Venn diagram aesthetics ----------
        colors : list or tuple of str, optional
            Colors for the two sets (length 2). If None, default matplotlib colors are used.
        intersection_color : str, optional
            Color for the intersection region. Defaults to automatic coloring.
        dibujar_bordes : bool, default=True
            Whether to draw black borders around each set.
        line_width : int or float, default=2
            Width of the borders.
        alpha : float, default=0.6
            Transparency of set areas (0 fully transparent, 1 fully opaque).
    
        Axes ----------
        show_axes : bool, default=False
            Whether to show axes.
        show_spines : bool, default=False
            Whether to show spines around the axes.
        spine_width : float, default=2
            Width of the spines, if shown.
    
        Figure saving and export ----------
        figsize : tuple of float, default=(3.35, 3.35)
            Figure size (width, height) in inches.
        dpi : int or float, default=600
            Resolution of the figure.
        save_path : str, optional
            File path to save the figure. If None, figure is not saved.
    
        Returns
        -------
        None
            The function directly displays the Venn diagram and optionally saves it to disk.
    
        Notes
        -----
        - Input sets are converted to Python sets internally to ensure unique elements.
        - This function is general-purpose and can be applied to any pair of categorical or list-like datasets,
          not only transcriptomic or genomic data.
        - The number of subsets is fixed to 2; for more than two sets, use `venn3` or other specialized packages.
    """
    # =========================
    # Input validations
    # =========================
    
    # Validate sets
    if not hasattr(set1, "__iter__") or not hasattr(set2, "__iter__"):
        raise TypeError("Both set1 and set2 must be iterable (list, set, tuple, etc.)")
    
    # Convert to set to ensure uniqueness
    set1 = set(set1)
    set2 = set(set2)
    
    # Validate labels
    if labels is not None:
        if not isinstance(labels, (list, tuple)) or len(labels) != 2:
            raise ValueError("labels must be a list or tuple of length 2")
        if not all(isinstance(l, str) for l in labels):
            raise TypeError("All items in labels must be strings")
    else:
        labels = ("Set 1", "Set 2")
    
    # Validate colors
    if colors is not None:
        if not isinstance(colors, (list, tuple)) or len(colors) != 2:
            raise ValueError("colors must be a list or tuple of length 2")
    if intersection_color is not None and not isinstance(intersection_color, str):
        raise TypeError("intersection_color must be a string representing a valid color")
    
    # Validate aesthetics
    if not isinstance(font_size, (int, float)):
        raise TypeError("font_size must be numeric")
    if not isinstance(line_width, (int, float)):
        raise TypeError("line_width must be numeric")
    if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
        raise ValueError("alpha must be a float between 0 and 1")
    
    # Validate figure and dpi
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must be a tuple/list of length 2 (width, height)")
    if not isinstance(dpi, (int, float)):
        raise TypeError("dpi must be numeric")
    
    # Validate save_path
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string")

    fig, ax = plt.subplots(figsize=figsize)

    # Show or hide labels
    if not show_labels:
        set_labels = ("", "")
    elif labels is None:
        set_labels = ("Set 1", "Set 2")  # default
    else:
        set_labels = labels


    v = venn2(
        subsets=(set(set1), set(set2)),
        set_labels=set_labels,
        ax=ax
    )

    # -------------------------
    # Labels and numbers text
    # -------------------------
    # Set labels
    if v.set_labels:
        for text in v.set_labels:
            if text is not None:
                text.set_fontsize(font_size)
                text.set_fontweight(label_fontweight)
                text.set_fontstyle(label_fontstyle)

    # Number of subsets
    if v.subset_labels:
        for text in v.subset_labels:
            if text is not None:
                text.set_fontsize(font_size)

    # -------------------------
    # Colors and borders
    # -------------------------
    for i, patch in enumerate(v.patches):
        if patch is not None:

            if colors is not None and i < 2:
                patch.set_facecolor(colors[i])

            if i == 2 and intersection_color is not None:
                patch.set_facecolor(intersection_color)

            if dibujar_bordes:
                patch.set_edgecolor("black")
                patch.set_linewidth(line_width)
            else:
                patch.set_edgecolor(None)
                patch.set_linewidth(0)

            patch.set_alpha(alpha)

    # -------------------------
    # Axes and spines
    # -------------------------
    ax.set_axis_on() if show_axes else ax.set_axis_off()

    if show_spines:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(spine_width)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # -------------------------
    # Title
    # -------------------------
    if show_title:
        if title is None:
            title_text = "Venn Diagram"
    else:
        title_text = title
    
    ax.set_title(
        title_text,
        fontsize=fontsize_title,
        fontweight=title_fontweight,
        fontstyle=title_fontstyle
        )

    plt.tight_layout()

    # -------------------------
    # Figure saving
    # -------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()