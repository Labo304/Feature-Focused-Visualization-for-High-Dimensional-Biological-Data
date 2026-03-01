from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA



def calculate_pca(
    df_vst,
    treatments,
    exclude_treatments=None,
    exclude_samples=None,
    n_components=2
):
    """
        Perform Principal Component Analysis (PCA) on a numeric matrix.
        
        This function computes PCA on a matrix where **columns represent observations/samples**
        and **rows represent variables/features**. While it is commonly used for gene expression
        matrices (genes x samples), it can handle any numeric dataset with the same structure, 
        such as proteomics, metabolomics, survey data, or other high-dimensional measurements.
        
        Parameters
        ----------
        df_vst : pandas.DataFrame
            Numeric matrix of features x samples (rows = variables, columns = observations).
        treatments : list or array-like
            Labels or groupings for each column in `df_vst`.
        exclude_treatments : list, optional
            Labels of treatments/groups to exclude from the analysis.
        exclude_samples : list, optional
            Names of specific columns/samples to exclude.
        n_components : int, default=2
            Number of principal components to compute.
        
        Returns
        -------
        pca_df : pandas.DataFrame
            DataFrame containing the computed principal components along with metadata columns:
            - PC1, PC2, ..., PCn
            - Treatment: corresponding treatment/group label
            - Sample: sample/column name
        pca : sklearn.decomposition.PCA
            Fitted PCA object from scikit-learn.
        
        Notes
        -----
        - Samples (columns) are treated as observations and features (rows) as variables.
          Therefore, the PCA is performed on the transposed matrix (`df_vst.T`).
        - The function can compute any number of components via `n_components`, which allows
          flexibility in downstream visualization (e.g., plotting PC1 vs PC2).
        - Works for any numeric dataset, not limited to gene expression.
        - Ensure that `treatments` aligns with the columns of `df_vst`.
        
        Example
        -------
        >>> df = pd.DataFrame(np.random.rand(100, 6), index=[f"feat{i}" for i in range(100)])
        >>> treatments = ["A", "A", "B", "B", "C", "C"]
        >>> pca_df, pca = calculate_pca(df, treatments)
    """
    # =========================
    # Input validation
    # =========================
    if not isinstance(df_vst, pd.DataFrame):
        raise TypeError("df_vst must be a pandas DataFrame")

    if len(treatments) != df_vst.shape[1]:
        raise ValueError(
            f"'treatments' must have length {df_vst.shape[1]}, got {len(treatments)}"
        )
  
    # ================================   
    # Map samples to treatments
    # ================================  
    df_trat = pd.DataFrame({
        'sample': df_vst.columns,
        'treatment': treatments
    })

    # ========================  
    # Exclusions
    # ======================== 
    if exclude_treatments:
        df_trat = df_trat[~df_trat['treatment'].isin(exclude_treatments)]

    if exclude_samples:
        df_trat = df_trat[~df_trat['sample'].isin(exclude_samples)]
        
    # ========================    
    # Reset index 
    # ======================== 
    df_trat = df_trat.reset_index(drop=True)
    
    # ======================== 
    # Filtering the dataframe
    # ======================== 
    end_columns = df_trat['sample'].tolist()
    df_filtered = df_vst[end_columns]
    
    # ======================== 
    # PCA
    # ======================== 
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df_filtered.T)
    
    # ======================== 
    # Results
    # ======================== 
    pca_df = pd.DataFrame(
        pcs,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df["Treatment"] = df_trat["treatment"].values
    pca_df["Sample"] = df_trat["sample"].values

    return pca_df, pca

def plot_pca(
    # =========================
    # PCA data
    # =========================
    pca_df,
    pca,
    pc_x=1,
    pc_y=2,
    group_col="Treatment",

    # =========================
    # Dot style
    # =========================
    palette="Set2",
    point_size=100,
    label_samples=True,
    label_offset=(0.02, 0.02),
    fontsize_points=8,
    fontweight_points="bold",
    rename_samples=None,  # diccionario {'Sample1':'NuevoNombre1', ...}
    markers=None,
    alpha=1,
    
    # =========================
    # Axis labels
    # =========================
    xlabel=None,
    ylabel=None,
    show_xlabel=True,
    show_ylabel=True,
    fontsize_labels=9,
    fontweight_labels="normal",
    

    # =========================
    # title
    # =========================
    title=None,
    show_title=True,
    fontsize_title=10,
    fontweight_title="normal",

    # =========================
    # Legend
    # =========================
    show_legend=True,
    legend_title=None,
    legend_loc="lower right",
    legend_frame=True,       
    legend_facecolor="white",
    legend_edgecolor="black",
    legend_edgewidth=1,
    legend_alpha=1.0,
    fontsize_legend=8,
    fontsize_legend_title=9,
    fontweight_legend="normal",
    fontweight_legend_title="normal",

    # =========================
    # Ticks
    # =========================
    show_xticks=True,
    show_yticks=True,
    fontsize_ticks=8,
    tick_width=2,
    tick_length=6,
    fontweight_ticks="normal",

    # =========================
    # Spines (frame)
    # =========================
    show_spines=True,
    spine_width=2,

    # =========================
    # Grid
    # =========================
    grid=False,
    grid_color="grey",
    grid_alpha=0.5,
    grid_width=1,

    # =========================
    # Figure and export
    # =========================
    figsize=(8, 6),
    dpi=600,
    save_path=None,
    
    # -------------------------
    # Optional return
    # -------------------------
    return_fig=False
):
    """
        Plot a customizable PCA scatter plot with publication-quality styling.
        
        Parameters
        ----------
        
        # =========================
        # Data
        # =========================
        pca_df : pandas.DataFrame
            DataFrame with PCA coordinates and metadata.
        pca : sklearn.decomposition.PCA
            Fitted PCA object.
        
        # =========================
        # Components
        # =========================
        pc_x : int, default=1
        pc_y : int, default=2
        
        # =========================
        # Grouping
        # =========================
        group_col : str, default="Treatment"
        palette : str | list | dict, default="Set2"
        markers : dict | list | None, default=None
        alpha : float, default=1
        
        # =========================
        # Points / Labels
        # =========================
        point_size : int, default=100
        label_samples : bool, default=True
        label_offset : tuple, default=(0.02, 0.02)
        fontsize_points : int, default=8
        fontweight_points : str, default="bold"
        rename_samples : dict | None, default=None
        
        # =========================
        # Axis labels
        # =========================
        xlabel, ylabel : str | None
        show_xlabel, show_ylabel : bool, default=True
        fontsize_labels : int, default=9
        fontweight_labels : str, default="normal"
        
        # =========================
        # Title
        # =========================
        title : str | None
        show_title : bool, default=True
        fontsize_title : int, default=10
        fontweight_title : str, default="normal"
        
        # =========================
        # Legend
        # =========================
        show_legend : bool, default=True
        legend_title : str | None
        legend_loc : str, default="lower right"
        legend_frame : bool, default=True
        legend_facecolor : str, default="white"
        legend_edgecolor : str, default="black"
        legend_edgewidth : float, default=1
        legend_alpha : float, default=1.0
        fontsize_legend : int, default=8
        fontsize_legend_title : int, default=9
        fontweight_legend : str, default="normal"
        fontweight_legend_title : str, default="normal"
        
        # =========================
        # Ticks
        # =========================
        show_xticks, show_yticks : bool, default=True
        fontsize_ticks : int, default=8
        tick_width : float, default=2
        tick_length : float, default=6
        fontweight_ticks : str, default="normal"
        
        # =========================
        # Spines
        # =========================
        show_spines : bool, default=True
        spine_width : float, default=2
        
        # =========================
        # Grid
        # =========================
        grid : bool, default=False
        grid_color : str, default="grey"
        grid_alpha : float, default=0.5
        grid_width : float, default=1
        
        # =========================
        # Figure
        # =========================
        figsize : tuple, default=(8, 6)
        dpi : int, default=600
        save_path : str | None
        return_fig : bool, default=False
        
        Returns
        -------
        fig, ax : tuple(matplotlib Figure, Axes)
            Only returned if return_fig=True.
        
        Notes
        -----
        Samples are plotted as points, grouped by `group_col`.
        Designed for publication-ready figures.
        
        Only two principal components can be plotted at a time
        (controlled by `pc_x` and `pc_y`). For more components,
        call the function multiple times with different PC pairs.
    """
    
    # =========================
    # Input validation
    # =========================
    
    # type checks
    if not isinstance(pca_df, pd.DataFrame):
        raise TypeError("pca_df must be a pandas DataFrame")
    
    if not hasattr(pca, "explained_variance_ratio_"):
        raise TypeError("pca must be a fitted sklearn PCA object")
    
    # column checks
    required_cols = {f"PC{pc_x}", f"PC{pc_y}", group_col, "Sample"}
    missing = required_cols - set(pca_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in pca_df: {missing}")
    
    # logical checks
    if pc_x == pc_y:
        raise ValueError("pc_x and pc_y must be different")
    
    # PCA bounds check
    max_pc = pca.n_components_
    if pc_x > max_pc or pc_y > max_pc:
        raise ValueError(
            f"PCA only has {max_pc} components. Requested PC{pc_x}, PC{pc_y}."
        )
    
    # markers check
    if markers is not None and not isinstance(markers, (dict, list)):
        raise TypeError("markers must be dict, list or None")
    
    x = f"PC{pc_x}"
    y = f"PC{pc_y}"

    # -------------------------
    # Figure
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------
    # Scatter PCA
    # -------------------------
    sns.scatterplot(
        data=pca_df,
        x=x,
        y=y,
        hue=group_col,
        style=group_col,
        palette=palette,
        s=point_size,
        markers=markers,
        alpha=alpha,
        ax=ax
    )

    # -------------------------
    # Sample labels
    # -------------------------
    if label_samples:
        for _, row in pca_df.iterrows():
            sample_name = row["Sample"]
            if rename_samples and sample_name in rename_samples:
                sample_name = rename_samples[sample_name]
            ax.text(
                row[x] + label_offset[0],
                row[y] + label_offset[1],
                sample_name,
                fontsize=fontsize_points,
                weight=fontweight_points
            )

    # -------------------------
    # Axis labels
    # -------------------------
    if show_xlabel:
        ax.set_xlabel(
            xlabel if xlabel else f"{x} ({pca.explained_variance_ratio_[pc_x-1]*100:.2f}%)",
            fontsize=fontsize_labels,
            fontweight=fontweight_labels
        )
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(
            ylabel if ylabel else f"{y} ({pca.explained_variance_ratio_[pc_y-1]*100:.2f}%)",
            fontsize=fontsize_labels,
            fontweight=fontweight_labels
        )
    else:
        ax.set_ylabel("")

    # -------------------------
    # Title
    # -------------------------
    if show_title:
        if not title:
            title = f"PCA: PC{pc_x} vs PC{pc_y}"
        ax.set_title(
            title,
            fontsize=fontsize_title,
            fontweight=fontweight_title
        )

    # -------------------------
    # Legend
    # -------------------------
    if show_legend:
        leg = ax.legend(
            title=legend_title if legend_title else group_col,
            fontsize=fontsize_legend,
            title_fontsize=fontsize_legend_title,
            frameon=legend_frame,
            loc=legend_loc
        )

        # Weight of the title of the legend
        leg.get_title().set_fontweight(fontweight_legend_title)

        # Weight of the legend text
        for text in leg.get_texts():
            text.set_fontweight(fontweight_legend)

        if legend_frame:
            frame = leg.get_frame()
            frame.set_facecolor(to_rgba(legend_facecolor, legend_alpha))
            frame.set_edgecolor(legend_edgecolor)
            frame.set_linewidth(legend_edgewidth)

    else:
        if ax.get_legend():
            ax.get_legend().remove()

    # -------------------------
    # Grid
    # -------------------------
    if grid:
        ax.grid(True, linewidth=grid_width, color=grid_color, alpha=grid_alpha)

    # -------------------------
    # Ticks
    # -------------------------
    if show_xticks:
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=fontsize_ticks,
            width=tick_width,
            length=tick_length
        )
        for label in ax.get_xticklabels():
            label.set_fontweight(fontweight_ticks)
    else:
        ax.set_xticks([])

    if show_yticks:
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=fontsize_ticks,
            width=tick_width,
            length=tick_length
        )
        for label in ax.get_yticklabels():
            label.set_fontweight(fontweight_ticks)
    else:
        ax.set_yticks([])

    # -------------------------
    # Spines
    # -------------------------
    if show_spines:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    # -------------------------
    # figure saving
    # -------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    plt.show()
    
    # -------------------------
    # Optional return
    # -------------------------
    if return_fig:
        return fig, ax
    else:
        plt.close(fig)


def obtain_top_variables(
    pca,
    filtered_data,
    metadata_df,
    name_column='variable_name',
    top_n=20,
    n_pcs=None
):
    """
    Extract top variables contributing to principal components (loadings) from a PCA.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    filtered_data : pandas.DataFrame
        Data matrix used for PCA (variables x samples). The index must match `metadata_df`.
    metadata_df : pandas.DataFrame
        DataFrame containing variable metadata. Must include a column specified by `name_column` 
        and have the same index as `filtered_data`.
    name_column : str, default='variable_name'
        Column name in `metadata_df` containing variable identifiers.
    top_n : int, default=20
        Number of top variables to return per component.
    n_pcs : int, optional
        Number of principal components to analyze. Defaults to all computed components.

    Returns
    -------
    dict or tuple of pandas.DataFrame
        If n_pcs > 3: returns a dictionary with PC names as keys and DataFrames as values:
            { 'PC1': df_top_PC1, 'PC2': df_top_PC2, ... }
        If n_pcs <= 3: returns a tuple of DataFrames in order (PC1, PC2, ...).

        Each DataFrame contains:
            - Variable identifier (column name defined by `name_column`)
            - Loading value for that PC

    Notes
    -----
    - The function ranks variables by the absolute value of their loading in each principal component.
    - Loadings indicate which variables contribute most to the variance captured by each PC.
    - This function is general and can be applied to genes, proteins, metabolites, or any other variables.
    - Indices of `filtered_data` and `metadata_df` must match exactly.
    """
    # =========================
    # Input validation
    # =========================
    if not hasattr(pca, "components_"):
        raise TypeError("pca must be a fitted sklearn PCA object with attribute 'components_'")

    if not isinstance(filtered_data, pd.DataFrame):
        raise TypeError("filtered_data must be a pandas DataFrame")

    if not isinstance(metadata_df, pd.DataFrame):
        raise TypeError("metadata_df must be a pandas DataFrame")

    if name_column not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain a column '{name_column}'")

    if not filtered_data.index.equals(metadata_df.index):
        raise ValueError("Indices of filtered_data and metadata_df must match")

    # =========================
    # Determine number of PCs
    # =========================
    total_components = pca.components_.shape[0]

    if n_pcs is None:
        n_pcs = total_components
    elif n_pcs > total_components:
        raise ValueError(f"PCA has only {total_components} components, but {n_pcs} were requested")

    pc_names = [f'PC{i+1}' for i in range(n_pcs)]

    # =========================
    # Compute loadings
    # =========================
    loadings = pd.DataFrame(
        pca.components_[:n_pcs, :],
        columns=filtered_data.index,
        index=pc_names
    ).T

    # Merge with variable names
    loadings = loadings.merge(metadata_df[[name_column]], left_index=True, right_index=True)

    # =========================
    # Top variables per PC
    # =========================
    top_variables = {}
    for pc in pc_names:
        top_variables[pc] = loadings.reindex(
            loadings[pc].abs().sort_values(ascending=False).index
        ).head(top_n)[[name_column, pc]]

    # =========================
    # Return
    # =========================
    if n_pcs <= 3:
        # Tuple for unpacking
        return tuple(top_variables[pc] for pc in pc_names)
    else:
        return top_variables


def plot_top_variables_pc(
    # =========================
    # Data
    # =========================
    top_PC,
    pc_col,
    name_column,
    pc_num=None,  

    # =========================
    # Bar style
    # =========================
    bar_color="skyblue",
    bar_edgecolor="black",
    bar_edge_width=2,

    # =========================
    # Title
    # =========================
    title=None,
    show_title=True,
    fontsize_title=10,
    fontweight_title="normal",

    # =========================
    # Axis labels
    # =========================
    xlabel=None,
    show_xlabel=True,
    ylabel=None,
    show_ylabel=True,
    fontsize_label=9,
    fontweight_label="normal",

    # =========================
    # Ticks
    # =========================
    show_xticks=True,
    show_yticks=True,
    fontsize_ticks=8,
    fontweight_ticks="normal",
    tick_width=2,
    tick_length=6,

    # =========================
    # Spines
    # =========================
    spine_width=2,

    # =========================
    # Figure and export
    # =========================
    figsize=(3.35, 4),
    dpi=600,
    save_path=None
):
    """
    Plot the top variables contributing to a specific principal component (PC) as a horizontal bar chart.
    
    Parameters
    ----------
    
    Data
    ----------
    top_PC : pandas.DataFrame
        DataFrame containing the top variables for a PC. Must include columns:
        - `name_column`: identifiers for the variables (e.g., genes, metabolites, proteins, features)
        - pc_col: numeric loadings for the PC to plot
    pc_col : str
        Column name in `top_PC` containing the loadings for the principal component.
    name_column : str
        Column name in `top_PC` containing variable identifiers.
    pc_num : int or None, optional
        PC number, used for default title. Default is None.
    
    
    Bar style
    ----------
    bar_color : str, default='skyblue'
        Fill color of the bars.
    bar_edgecolor : str, default='black'
        Edge color of the bars.
    bar_edge_width : float, default=2
        Width of the bar edges.
    
    
    Title
    ----------
    title : str or None, optional
        Plot title. If None, a default title is generated using `pc_num`.
    show_title : bool, default=True
        Whether to display the plot title.
    fontsize_title : int, default=10
        Font size of the title.
    fontweight_title : str, default='normal'
        Font weight of the title.
    
     
    Axis labels
    ----------
    xlabel : str or None, optional
        Label for the x-axis. Default is 'Loading'.
    show_xlabel : bool, default=True
        Whether to display the x-axis label.
    ylabel : str or None, optional
        Label for the y-axis. Default is the `name_column`.
    show_ylabel : bool, default=True
        Whether to display the y-axis label.
    fontsize_label : int, default=9
        Font size for axis labels.
    fontweight_label : str, default='normal'
        Font weight for axis labels.
    
    
    Ticks
    ----------
    show_xticks : bool, default=True
        Whether to show x-axis ticks.
    show_yticks : bool, default=True
        Whether to show y-axis ticks.
    fontsize_ticks : int, default=8
        Font size of tick labels.
    fontweight_ticks : str, default='normal'
        Font weight of tick labels.
    tick_width : float, default=2
        Width of tick marks.
    tick_length : float, default=6
        Length of tick marks.
    
    
    Spines
    ----------
    spine_width : float, default=2
        Width of the plot spines.
    
    
    Figure and export
    ----------
    figsize : tuple, default=(3.35, 4)
        Figure size (width, height) in inches.
    dpi : int, default=600
        Figure resolution.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    
    Returns
    -------
    None
        Displays the horizontal bar chart and optionally saves it.
    
    Notes
    -----
    - Variables are ranked by absolute loadings in the specified PC.
    - Works with any type of variables, not restricted to genes.
    - Designed for publication-quality visualization of top features per PC.
    - Can be used with output from a generalized `obtain_top_variables()` function.
    """
    
    #=========================
    # Input validation
    # =========================
    if not isinstance(top_PC, pd.DataFrame):
        raise TypeError("top_PC must be a pandas DataFrame")
    
    if top_PC.empty:
        raise ValueError("top_PC DataFrame is empty")
    
    if name_column not in top_PC.columns:
        raise ValueError(f"top_PC must contain the specified name_column '{name_column}'")
    
    if pc_col not in top_PC.columns:
        raise ValueError(f"top_PC must contain the specified pc_col '{pc_col}'")
    
    if pc_num is not None and not isinstance(pc_num, int):
        raise TypeError("pc_num must be an integer or None")
    
    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------
    # Bars
    # -------------------------
    ax.barh(
        top_PC[name_column],
        top_PC[pc_col],
        color=bar_color,
        edgecolor=bar_edgecolor,
        linewidth=bar_edge_width
    )

    # -------------------------
    # Title
    # -------------------------
    if show_title:
        if not title:
            title = f"Top variables PC{pc_num}" if pc_num else "Top variables"
        ax.set_title(
            title,
            fontsize=fontsize_title,
            fontweight=fontweight_title
        )

    # -------------------------
    # Axis labels
    # -------------------------
    if show_xlabel:
        if not xlabel:
            xlabel = "Loading"
        ax.set_xlabel(
            xlabel,
            fontsize=fontsize_label,
            fontweight=fontweight_label
        )

    if show_ylabel:
        if not ylabel:
            ylabel = name_column
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize_label,
            fontweight=fontweight_label
        )

    # -------------------------
    # Ticks
    # -------------------------
    if show_xticks:
        ax.tick_params(
            axis="x",
            labelsize=fontsize_ticks,
            width=tick_width,
            length=tick_length
        )
        for label in ax.get_xticklabels():
            label.set_fontweight(fontweight_ticks)
    else:
        ax.set_xticks([])

    if show_yticks:
        ax.tick_params(
            axis="y",
            labelsize=fontsize_ticks,
            width=tick_width,
            length=tick_length
        )
        for label in ax.get_yticklabels():
            label.set_fontweight(fontweight_ticks)
    else:
        ax.set_yticks([])

    # -------------------------
    # Spines
    # -------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    plt.tight_layout()

    # -------------------------
    # Figure saving
    # -------------------------
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

