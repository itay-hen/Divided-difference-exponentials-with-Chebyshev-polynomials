"""plot_style.py -- shared matplotlib style for the accuracy experiments
(serif / Computer Modern mathtext, 9 pt, colourblind-safe Okabe--Ito palette).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe & Ito (2008) colourblind-safe palette.
PALETTE = ["#0072B2",  # blue
           "#E69F00",  # orange
           "#009E73",  # bluish green
           "#D55E00",  # vermillion
           "#CC79A7",  # reddish purple
           "#56B4E9",  # sky blue
           "#F0E442",  # yellow
           "#000000"]  # black

FULL_WIDTH = 6.5   # inches
SINGLE_WIDTH = 3.2


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,  # cmr10 has no Unicode minus
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })
