"""functions for setting LaTex fonts and figure dimensions for plotting."""

def get_tex_fonts(document):
    """Get LaTex fonts for plotting.

    Args:
        document (string): 'beamer' or 'thesis'

    Returns:
        dict: LaTex font settings
    """
    
    if document == 'beamer':
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "sans-serif",
            "text.latex.preamble" : (r"\usepackage{amsmath}"
                                     r"\usepackage{sansmath}"
                                     r"\sansmath"),
            # Use 11pt font in plots, to match 11pt font in document
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "font.size": 11,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        }
        
    elif document == 'thesis':
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble" : (r"\usepackage[scaled=1.0]{XCharter}"
                                     r"\usepackage{amsmath}"
                                     r"\usepackage{amsfonts}"
                                     r"\usepackage{amssymb}"),
            # Use 11pt font in plots, to match 11pt font in document
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "font.size": 11,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        }
    return tex_fonts


def get_size(width, fraction=1, subplots=(1, 1), ratio='golden'):
    """Get figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
            
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    
    if width == 'thesis':
        width_pt = 432.48195 # A4 thesis
    elif width == 'beamer':
        # width_pt = 342.2953 # for 4:3 aspect ratio
        width_pt = 433.34402 # for 16:10 aspect ratio
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio == 'golden':
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5**.5 - 1) / 2
    elif ratio == 'equal':
        ratio = 1

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

