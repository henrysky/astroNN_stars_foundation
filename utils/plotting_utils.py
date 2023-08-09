import copy
import matplotlib
import pylab as plt
from astropy.stats import mad_std
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from utils.gaia_utils import xp_sampling_grid


def top_cbar(ax, mappable, text=None, labelpad=None, ticksformat=None):
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal", format=ticksformat)
    cbar.set_label(text, labelpad=labelpad)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    return cbar


def density_cmap():
    """
    colormap for matplotlib with white as the lowest value
    """
    custom_cmap = copy.copy(plt.get_cmap("viridis"))
    custom_cmap.set_under(color="white")
    return custom_cmap


def plot_kiel_uncertainty(
    val_labels_pd,
    pred_df,
    suptitle=None,
    figsize=(12, 5.2),
    plot_one_to_one=True,
    plot_topbar=True,
    density=False,
    fig=None,
):
    if fig is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    else:
        ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    if density:
        mappable = ax1.hexbin(
            val_labels_pd["teff"],
            pred_df["teff"],
            bins="log",
            extent=[3000, 7000, 3000, 7000],
            cmap=density_cmap(),
            vmin=1.1,
            rasterized=True,
        )
    else:
        mappable = ax1.scatter(
            val_labels_pd["teff"],
            pred_df["teff"],
            s=0.1,
            c=pred_df["teff_error"],
            vmin=50,
            vmax=150,
            cmap="plasma",
            rasterized=True,
        )
    if plot_one_to_one:
        ax1.plot([3000, 7000], [3000, 7000], lw=2, ls="--", c="r")
    ax1.set_xlim(3000, 7000)
    ax1.set_ylim(3000, 7000)
    ax1.set_yticklabels([])
    ax1.set_xlabel("APOGEE $T_\mathrm{eff}$ (K)")
    ax1.set_ylabel("NN $T_\mathrm{eff}$ (K)")
    ax1.set_aspect("equal", "box")
    ax1.annotate(
        f"$\\sigma=${mad_std(val_labels_pd['teff'] - pred_df['teff'], ignore_nan=True):.2f} K",
        xy=(0.05, 0.95),
        xycoords=ax1,
        fontsize=20,
        ha="left",
        va="top",
    )
    if plot_topbar:
        if density:
            top_cbar(ax1, mappable, "log$_{10}$ N", labelpad=10)
        else:
            top_cbar(ax1, mappable, "$\sigma_\mathrm{model}$ (K)", labelpad=10)

    if density:
        mappable = ax2.hexbin(
            val_labels_pd["logg"],
            pred_df["logg"],
            bins="log",
            extent=[0.0, 5.0, 0.0, 5.0],
            cmap=density_cmap(),
            vmin=1.1,
            rasterized=True,
        )
    else:
        mappable = ax2.scatter(
            val_labels_pd["logg"],
            pred_df["logg"],
            s=0.1,
            c=pred_df["logg_error"],
            vmin=0.05,
            vmax=0.3,
            cmap="plasma",
            rasterized=True,
        )
    if plot_one_to_one:
        ax2.plot([0.0, 5.0], [0.0, 5.0], lw=2, ls="--", c="r")
    ax2.set_xlim(0.0, 5.0)
    ax2.set_ylim(0.0, 5.0)
    ax2.set_yticklabels([])
    ax2.set_xlabel("APOGEE $\log{g}$ (dex)")
    ax2.set_ylabel("NN $\log{g}$ (dex)")
    ax2.set_aspect("equal", "box")
    ax2.annotate(
        f"$\\sigma=${mad_std(val_labels_pd['logg'] - pred_df['logg'], ignore_nan=True):.2f} dex",
        xy=(0.05, 0.95),
        xycoords=ax2,
        fontsize=20,
        ha="left",
        va="top",
    )
    if plot_topbar:
        if density:
            top_cbar(ax2, mappable, "log$_{10}$ N", labelpad=10)
        else:
            top_cbar(ax2, mappable, "$\sigma_\mathrm{model}$ (dex)", labelpad=10)

    if density:
        mappable = ax3.hexbin(
            val_labels_pd["m_h"],
            pred_df["m_h"],
            bins="log",
            extent=[-1.9, 0.5, -1.9, 0.5],
            cmap=density_cmap(),
            vmin=1.1,
            rasterized=True,
        )
    else:
        mappable = ax3.scatter(
            val_labels_pd["m_h"],
            pred_df["m_h"],
            s=0.1,
            c=pred_df["m_h_error"],
            vmin=0.05,
            vmax=0.3,
            cmap="plasma",
            rasterized=True,
        )
    if plot_one_to_one:
        ax3.plot([-2.5, 0.5], [-2.5, 0.5], lw=2, ls="--", c="r")
    ax3.set_xlim(-2.5, 0.5)
    ax3.set_ylim(-2.5, 0.5)
    ax3.set_yticklabels([])
    ax3.set_xlabel("APOGEE [M/H] (dex)")
    ax3.set_ylabel("NN [M/H] (dex)")
    ax3.set_aspect("equal")
    ax3.annotate(
        f"$\\sigma=${mad_std(val_labels_pd['m_h'] - pred_df['m_h'], ignore_nan=True):.2f} dex",
        xy=(0.05, 0.95),
        xycoords=ax3,
        fontsize=20,
        ha="left",
        va="top",
    )
    if plot_topbar:
        if density:
            top_cbar(ax3, mappable, "log$_{10}$ N", labelpad=10)
        else:
            top_cbar(ax3, mappable, "$\sigma_\mathrm{model}$ (dex)", labelpad=10)

    if suptitle is not None:
        fig.suptitle(suptitle, x=0.04, y=1.05, ha="left")


def setup_xp_physical_plot(
    ax,
    specs_ls,
    c,
    cmap,
    vmin,
    vmax,
    cbar_title,
    lines_list=False,
    logscale=0,
):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i, c in zip(specs_ls, c):
        ax.plot(
            xp_sampling_grid,
            i / 10**logscale,
            lw=2,
            color=matplotlib.colors.to_hex(cmap(norm(c))),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    top_cbar(ax, sm, cbar_title)

    ax.set_xlabel("Wavelength ($nm$)")
    scale_text = "" if logscale == 0 else f"10^{{{logscale}}} "
    ax.set_ylabel(f"Flux at 10 $pc$ (${scale_text} W nm^{{{-1}}} m^{{{-2}}}$)")
    ax.set_xlim(392, 992)

    if lines_list:
        # [M/H] lines
        ax.axvline(422.672, lw=2, alpha=0.5, c="C5")  # Ca
        ax.axvline(430.774, lw=2, alpha=0.5, c="C5")  # Ca
        ax.axvline(430.790, lw=2, alpha=0.5, c="C5", label="[M/H]")  # Fe
        ax.axvline(438.355, lw=2, alpha=0.5, c="C5")  # Fe, Mg
        ax.axvline(516.891, lw=2, alpha=0.5, c="C5")  # Fe

        # Balmer series
        ax.axvline(410.175, lw=2, alpha=0.5, ls="--", c="C2", label="Balmer series")
        ax.axvline(486.134, lw=2, alpha=0.5, ls="--", c="C2")
        ax.axvline(656.281, lw=2, alpha=0.5, ls="--", c="C2")

        # calcium triplet
        ax.axvspan(850, 867, alpha=0.2, lw=0, color="C4", label="Calcium Triplet")
