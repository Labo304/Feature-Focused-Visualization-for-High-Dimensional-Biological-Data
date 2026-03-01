import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

def initialize_default_fontstyle(
    font_name="Arial",
    fallback_fonts=("Liberation Sans", "DejaVu Sans"),
    use_seaborn_theme=True,
    seaborn_style="whitegrid",
    seaborn_context="notebook",
    show_demo=True,
    show_diagnosis=True,
):
    
    """
        Initialize a global plotting style for matplotlib/Seaborn with portable font settings.
        
        This function sets a consistent visual style for plots, optionally using a Seaborn theme,
        and enforces a preferred font, falling back to alternative fonts if necessary.
        
        Parameters
        ----------
        font_name : str, default="Arial"
            Preferred font to apply for plots. If not available, a font from `fallback_fonts`
            or 'DejaVu Sans' will be used.
            
        fallback_fonts : iterable of str, default=("Liberation Sans", "DejaVu Sans")
            Fonts to try if `font_name` is not available on the system. The first available
            font in the list will be applied.
            
        use_seaborn_theme : bool, default=True
            If True, applies a Seaborn theme with the specified `seaborn_style` and `seaborn_context`.
            If False, matplotlib's default style is restored.
            
        seaborn_style : str, default="whitegrid"
            Seaborn style to apply. Must be one of: "darkgrid", "whitegrid", "dark", "white", "ticks".
            
        seaborn_context : str, default="notebook"
            Seaborn context to apply. Must be one of: "notebook", "paper", "talk", "poster".
            
        show_demo : bool, default=True
            If True, displays a small example plot to visually verify the font selection.
            
        show_diagnosis : bool, default=True
            If True, prints diagnostic messages including which font was applied and the
            status of the Seaborn theme.
            
        Returns
        -------
        selected_font : str
            The actual font that was applied to matplotlib's rcParams.
            Could be `font_name`, a font from `fallback_fonts`, or 'DejaVu Sans' as a last resort.
            
        Notes
        -----
        - The font is enforced after setting the Seaborn theme to avoid being overwritten.
        - The function checks system-installed fonts; some fonts may not be available on all systems.
        - The visual demo uses matplotlib and may fail in headless environments (e.g., some servers).
          In such cases, set `show_demo=False`.
        - The function updates `mpl.rcParams` globally, affecting all subsequent plots.
    """
    
    # ===============================
    # ✅ Validation block
    # ===============================
    # 1️⃣ Basic types
    if not isinstance(font_name, str):
        raise TypeError(f"font_name must be a string, not {type(font_name).__name__}")
    
    if not hasattr(fallback_fonts, "__iter__") or isinstance(fallback_fonts, str):
        raise TypeError("fallback_fonts must be an iterable of strings (not a single str)")
    
    if not isinstance(use_seaborn_theme, bool):
        raise TypeError("use_seaborn_theme must be a bool")
    
    if not isinstance(seaborn_style, str):
        raise TypeError("seaborn_style must be a string")
    
    if not isinstance(seaborn_context, str):
        raise TypeError("seaborn_context must be a string")
    
    if not isinstance(show_demo, bool):
        raise TypeError("show_demo must be a bool")
    
    if not isinstance(show_diagnosis, bool):
        raise TypeError("show_diagnosis must be a bool")
    
    # 2️⃣ Allowed values for seaborn_style and seaborn_context
    valid_styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
    valid_contexts = ["notebook", "paper", "talk", "poster"]
    
    if seaborn_style not in valid_styles:
        raise ValueError(f"seaborn_style must be one of {valid_styles}, not '{seaborn_style}'")
    
    if seaborn_context not in valid_contexts:
        raise ValueError(f"seaborn_context must be one of {valid_contexts}, not '{seaborn_context}'")

    # ===============================
    # 1️⃣ Full style control
    # ===============================
    import seaborn as sns
    
    if use_seaborn_theme:
        sns.set_theme(style=seaborn_style, context=seaborn_context)
    else:
        sns.reset_orig()   

    # ===============================
    # 2️⃣ Detect available sources
    # ===============================
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    if font_name in available_fonts:
        selected_font = font_name
        found = True
    else:
        found = False
        selected_font = None
        for fb in fallback_fonts:
            if fb in available_fonts:
                selected_font = fb
                break

        if selected_font is None:
            selected_font = "DejaVu Sans"

    # ===============================
    # 3️⃣ FORCE font (AFTER theme)
    # ===============================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected_font],
        "mathtext.fontset": "dejavusans",  
    })

    # ===============================
    # 4️⃣ Diagnosis
    # ===============================
    if show_diagnosis:
        if found:
            print(f"✔ Font '{font_name}' applied.")
        else:
            print(f"⚠ Font '{font_name}' not available.")
            print(f"→ Using fallback: '{selected_font}'")

        if use_seaborn_theme:
            print(f"Seaborn theme active: {seaborn_style} | context: {seaborn_context}")
        else:
            print("Seaborn disabled.")

        print(f"Final font in rcParams: {selected_font}")

    # ===============================
    # 5️⃣ Visual verification
    # ===============================
    if show_demo:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.set_title("Font Verification")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.text(0.5, 0.5, f"{selected_font}",
                ha="center", va="center",
                transform=ax.transAxes)
        plt.tight_layout()
        plt.show()

        real_font = ax.title.get_fontproperties().get_name()
        if show_diagnosis:
            print("Font actually rendered:", real_font)

    return selected_font