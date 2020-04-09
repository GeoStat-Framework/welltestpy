# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing plotting routines.

.. currentmodule:: welltestpy.tools.plotter

The following classes and functions are provided

.. autosummary::
   campaign_plot
   campaign_well_plot
   plot_pump_test
   plot_well_pos
   fadeline
   plotfit_transient
   plotfit_steady
   plotparainteract
   plotparatrace
   plotsensitivity
"""
# pylint: disable=C0103
import copy
import warnings
import functools as ft

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _get_fig_ax(
    fig=None,
    ax=None,
    ax_name="rectilinear",
    sub_args=None,
    sub_kwargs=None,
    **fig_kwargs
):  # pragma: no cover
    # ax_case: 0->None (create one) or given, 1->False, 2->True
    ax_case = 1 + int(ax) if isinstance(ax, bool) else 0
    sub_args = (111,) if sub_args is None else sub_args
    sub_kwargs = {} if sub_kwargs is None else sub_kwargs
    sub_kwargs["projection"] = ax_name
    if ax_case == 0:
        if fig is None:
            fig = plt.figure(**fig_kwargs) if ax is None else ax.get_figure()
        if ax is None:
            ax = fig.add_subplot(*sub_args, **sub_kwargs)
        assert ax.name == ax_name
        assert ax.get_figure() is fig
        return fig, ax
    # if ax=False we only want a figure
    if ax_case == 1:
        return plt.figure(**fig_kwargs) if fig is None else fig
    # if ax=True we want the current axis of the given figure
    assert fig is not None
    return fig, fig.gca()


def _sort_lgd(ax, **kwargs):
    """Show legend and sort it by names."""
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, **kwargs)


def dashes(i=1, max_n=12, width=1):
    """
    Dashes for matplotlib.

    Parameters
    ----------
    i : int, optional
        Number of dots. The default is 1.
    max_n : int, optional
        Maximal Number of dots. The default is 12.
    width : float, optional
        Linewidth. The default is 1.

    Returns
    -------
    list
        dashes list for matplotlib.

    """
    return i * [width, width] + [max_n * 2 * width - 2 * i * width, width]


def fadeline(ax, x, y, label=None, color=None, steps=20, **kwargs):
    """Fading line for matplotlib.

    This is a workaround to produce a fading line.

    Parameters
    ----------
    ax : axis
        Axis to plot on.
    x : :class:`list`
        start and end value of x components of the line
    y : :class:`list`
        start and end value of y components of the line
    label : :class:`str`, optional
        label for the legend.
        Default: ``None``
    color : MPL color, optional
        color of the line
        Default: ``None``
    steps : :class:`int`, optional
        steps of fading
        Default: ``20``
    **kwargs
        keyword arguments that are forwarded to `plt.plot`
    """
    xarr = np.linspace(x[0], x[1], steps + 1)
    yarr = np.linspace(y[0], y[1], steps + 1)

    kwargs.pop("label", None)
    kwargs.pop("alpha", None)
    kwargs["color"] = color
    kwargs["solid_capstyle"] = "butt"

    for i in range(steps):
        kwargs["label"] = label if i == 0 else None
        kwargs["alpha"] = (steps - i) * (1.0 / steps) * 0.9 + 0.1
        ax.plot([xarr[i], xarr[i + 1]], [yarr[i], yarr[i + 1]], **kwargs)


def campaign_plot(campaign, select_test=None, fig=None, style="WTP", **kwargs):
    """
    Plot an overview of the tests within the campaign.

    Parameters
    ----------
    campaign : :class:`Campaign`
        The campaign to be plotted.
    select_test : dict, optional
        The selected tests to be added to the plot. The default is None.
    fig : Figure, optional
        Matplotlib figure to plot on. The default is None.
    style : str, optional
        Plot stlye. The default is "WTP".
    **kwargs : TYPE
        Keyword arguments forwarded to the tests plotting routines.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    """
    if select_test is None:
        tests = list(campaign.tests.keys())
    else:
        tests = select_test

    tests.sort()
    nroftests = len(tests)
    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig = _get_fig_ax(fig, ax=False, dpi=75, figsize=[8, 3 * nroftests])

        for n, t in enumerate(tests):
            ax = fig.add_subplot(nroftests, 1, n + 1)
            # call the plotting routine of the test
            campaign.tests[t].plot(wells=campaign.wells, ax=ax, **kwargs)

        fig.tight_layout()
        fig.show()
    return fig


def campaign_well_plot(
    campaign, plot_tests=True, plot_well_names=True, fig=None, style="WTP"
):
    """
    Plot of the well constellation within the campaign.

    Parameters
    ----------
    campaign : :class:`Campaign`
        The campaign to be plotted.
    plot_tests : bool, optional
        DESCRIPTION. The default is True.
    plot_well_names : TYPE, optional
        DESCRIPTION. The default is True.
    fig : Figure, optional
        Matplotlib figure to plot on. The default is None.
    style : str, optional
        Plot stlye. The default is "WTP".

    Returns
    -------
    ax : Axes
        The created matplotlib axes.

    """
    well_const0 = []
    names = []

    for w in campaign.wells:
        well_const0.append(
            [campaign.wells[w].pos[0], campaign.wells[w].pos[1]]
        )
        names.append(w)
    well_const = [well_const0]

    fig = plot_well_pos(
        well_const,
        names,
        plot_well_names=plot_well_names,
        fig=fig,
        style=style,
    )

    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clr_n = len(clrs)

        fig, ax = _get_fig_ax(fig, ax=True)

        if plot_tests:
            testlist = list(campaign.tests.keys())
            testlist.sort()
            for i, t in enumerate(testlist):
                p_well = campaign.tests[t].pumpingwell
                for j, obs in enumerate(campaign.tests[t].observations):
                    x0 = campaign.wells[p_well].pos[0]
                    y0 = campaign.wells[p_well].pos[1]
                    x1 = campaign.wells[obs].pos[0]
                    y1 = campaign.wells[obs].pos[1]
                    label = "'{}'".format(t) if j == 0 else None
                    fadeline(
                        ax=ax,
                        x=[x0, x1],
                        y=[y0, y1],
                        label=label,
                        color=clrs[(i + 2) % clr_n],
                        linewidth=3,
                        zorder=10,
                    )
        # get equal axis (for realism)
        ax.axis("equal")
        ax.legend(title="Tests", loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.show()
    return ax


def plot_pump_test(
    pump_test, wells, exclude=None, fig=None, ax=None, style="WTP", **kwargs
):
    """Plot a pumping test.

    Parameters
    ----------
    pump_test: :class:`PumpingTest`
        Pumping test class that should be plotted.
    wells : :class:`dict`
        Dictonary containing the well classes sorted by name.
    exclude: :class:`list`, optional
        List of wells that should be excluded from the plot.
        Default: ``None``
    fig : Figure, optional
        Matplotlib figure to plot on. The default is None.
    ax : :class:`Axes`
        Matplotlib axes to plot on. The default is None.
    style : str, optional
        Plot stlye. The default is "WTP".

    Returns
    -------
    ax : Axes
        The created matplotlib axes.

    Notes
    -----
    This will be used by the Campaign class.
    """
    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clr_n = len(clrs)
        fig, ax = _get_fig_ax(fig, ax)
        exclude = set() if exclude is None else set(exclude)
        well_set = set(wells)
        test_wells = set(pump_test.observationwells)
        plot_wells = list((well_set & test_wells) - exclude)
        # sort by radius
        plot_wells.sort(key=lambda x: wells[x] - wells[pump_test.pumpingwell])
        state = pump_test.state(wells=plot_wells)
        steady_guide_x = []
        steady_guide_y = []
        # label for absolute values
        abslab = "abs. " if ("abs_val" in kwargs and kwargs["abs_val"]) else ""
        if state == "mixed":
            ax1 = ax
            ax2 = ax1.twiny()
        elif state == "transient":
            ax1 = ax
            ax2 = None
        elif state == "steady":
            ax1 = None
            ax2 = ax
        else:
            raise ValueError("plot_pump_test: unknow state of pumping test.")
        for i, k in enumerate(plot_wells):
            if k != pump_test.pumpingwell:
                dist = wells[k] - wells[pump_test.pumpingwell]
            else:
                dist = wells[pump_test.pumpingwell].radius
            if pump_test.observations[k].state == "transient":
                if abslab:
                    displace = np.abs(pump_test.observations[k].value[0])
                else:
                    displace = pump_test.observations[k].value[0]
                ax1.plot(
                    pump_test.observations[k].value[1],
                    displace,
                    linewidth=2,
                    color=clrs[i % clr_n],
                    label=(
                        pump_test.observations[k].name
                        + " r={:1.2f}".format(dist)
                    ),
                )
                ax1.set_xlabel(pump_test.observations[k].labels[0])
                ax1.set_ylabel(
                    abslab + "{}".format(pump_test.observations[k].labels[1])
                )
            else:
                if abslab:
                    displace = np.abs(pump_test.observations[k].value)
                else:
                    displace = pump_test.observations[k].value
                steady_guide_x.append(dist)
                steady_guide_y.append(displace)
                label = pump_test.observations[k].name + " r={:1.2f}".format(
                    dist
                )
                color = "C{}".format(i % 10)
                ax2.scatter(
                    dist, displace, color=color, label=label,
                )
                ax2.set_xlabel("r in {}".format(wells[k].coordinates.units))
                ax2.set_ylabel(
                    abslab + "{}".format(pump_test.observations[k].labels)
                )

        if state != "transient":
            steady_guide_x = np.array(steady_guide_x, dtype=float)
            steady_guide_y = np.array(steady_guide_y, dtype=float)
            arg = np.argsort(steady_guide_x)
            steady_guide_x = steady_guide_x[arg]
            steady_guide_y = steady_guide_y[arg]
            ax2.plot(steady_guide_x, steady_guide_y, color="k", alpha=0.1)

        if "title" not in kwargs or not kwargs["title"] is False:
            ax.set_title(repr(pump_test))
        if "xscale" in kwargs:
            ax.set_xscale(kwargs["xscale"])
        if "yscale" in kwargs:
            ax.set_yscale(kwargs["yscale"])

        ax.legend(
            title="Pumping test '{}'".format(pump_test.name),
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        if state == "mixed":  # add a second legend
            ax2.legend(loc="upper right", fancybox=True, framealpha=0.75)
    return ax


####


def plot_well_pos(
    well_const,
    names=None,
    title="",
    filename=None,
    plot_well_names=True,
    ticks_set="auto",
    fig=None,
    style="WTP",
):
    """
    Plot all well constellations and label the points with the names.

    Parameters
    ----------
    well_const : list
        List of well constellations.
    names : list of str, optional
        Names for the wells. The default is None.
    title : str, optional
        Plot title. The default is "".
    filename : str, optional
        Filename if the result should be saved. The default is None.
    plot_well_names : bool, optional
        Whether to plot the well-names. The default is True.
    ticks_set : int or str, optional
        Tick spacing in the plot. The default is "auto".
    fig : Figure, optional
        Matplotlib figure to plot on. The default is None.
    style : str, optional
        Plot stlye. The default is "WTP".

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    """
    # calculate Column- and Row-count for quadratic shape of the plot
    # total number of plots
    total_n = len(well_const)
    # columns near the square-root but tendentially wider than tall
    col_n = int(np.ceil(np.sqrt(total_n)))
    # enough rows to catch all plots
    row_n = int(np.ceil(total_n / col_n))
    # Possition numbers as array
    pos_tuple = np.arange(total_n) + 1

    # generate names for points if undefined
    if names is None:
        names = []
        for i in range(len(well_const[0])):
            names.append("p" + str(i))

    # genearte commen borders for all plots
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf

    for i in well_const:
        for j in i:
            xmax = max(j[0], xmax)
            xmin = min(j[0], xmin)
            ymax = max(j[1], ymax)
            ymin = min(j[1], ymin)

    # add some space around the points in the plot
    space = 0.1 * max(abs(xmax - xmin), abs(ymax - ymin))
    xspace = yspace = space

    if ticks_set == "auto":
        # bit hacky auto-ticking to be more pleasant for the eyes
        tick_list = [1, 2, 5, 10]
        tk_space = space * 10 / 7  # assume about 7 ticks
        scaling = np.log10(tk_space)
        if np.log10(0.4) < scaling < 1:
            # if space is less 10, choose nearest value in tick_list (by log)
            ticks_set = min(tick_list, key=lambda x: abs(np.log(x / tk_space)))
        else:
            # k * 10 ** n as ticks (0.1, 0.2, ..., 10, 20, ..., 100, 200, ...)
            space_pot = 10 ** int(np.floor(scaling))
            ticks_set = space_pot * int(np.around(tk_space / space_pot))

    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig = _get_fig_ax(
            fig, ax=False, dpi=100, figsize=[9 * col_n, 5 * row_n]
        )

        for i, wells in enumerate(well_const):
            ax = fig.add_subplot(row_n, col_n, pos_tuple[i])
            ax.set_xlim([xmin - xspace, xmax + xspace])
            ax.set_ylim([ymin - yspace, ymax + yspace])
            ax.set_aspect("equal")

            for j, name in enumerate(names):
                ax.scatter(wells[j][0], wells[j][1], color="k", zorder=100)
                if plot_well_names:
                    ax.annotate(
                        "  " + name, (wells[j][0], wells[j][1]), zorder=100
                    )
            ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks_set))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks_set))
            ax.set_xlabel("x distance in $[m]$")
            ax.set_ylabel("y distance in $[m]$")
            if total_n > 1:
                ax.set_title("Result {}".format(i))

        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if filename is not None:
            fig.savefig(filename, format="pdf")

    return fig


######


# Estimation plotting


def plotfit_transient(
    setup,
    data,
    para,
    rad,
    time,
    radnames,
    extra,
    plotname=None,
    fig=None,
    ax=None,
    style="WTP",
):
    """Plot of transient estimation fitting."""
    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style1 = "ggplot"
        style2 = "default"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    else:
        style1 = style2 = style
    with plt.style.context(style1):
        clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clr_n = len(clrs)
    with plt.style.context(style2):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig, ax = _get_fig_ax(fig, ax, ax_name=Axes3D.name, figsize=(12, 8))
        val_fix = setup.val_fix
        for kwarg in ["time", "rad"]:
            val_fix.pop(extra[kwarg], None)

        para_ordered = np.empty(len(setup.para_names))
        for i, name in enumerate(setup.para_names):
            para_ordered[i] = para[name]
        para_kw = setup.get_sim_kwargs(para_ordered)
        val_fix.update(para_kw)

        plot_f = ft.partial(setup.func, **val_fix)

        radarr = np.linspace(rad.min(), rad.max(), 100)
        timarr = np.linspace(time.min(), time.max(), 100)

        t_gen = np.ones_like(radarr)
        r_gen = np.ones_like(time)
        r_gen1 = np.ones_like(timarr)
        xydir = np.zeros_like(time)
        test_name = list(np.unique(radnames[:, 0]))
        test_name.sort()
        __, rad_un_idx = np.unique(rad, return_index=True)
        for ri, re in enumerate(rad):
            r1 = re * r_gen
            r11 = re * r_gen1
            h = plot_f(**{extra["time"]: time, extra["rad"]: re}).reshape(-1)
            h1 = data[:, ri]
            h2 = plot_f(**{extra["time"]: timarr, extra["rad"]: re}).reshape(
                -1
            )
            color = clrs[(test_name.index(radnames[ri, 0]) + 2) % clr_n]
            alpha = 0.3 * (1 - (re - min(rad)) / (max(rad) - min(rad))) + 0.3
            zord = 100 * (len(rad) - ri)

            if radnames[ri, 0] == radnames[ri, 1]:
                label = "test at '{}'".format(radnames[ri, 0])
                label_eff = "fitted type curve"
                eff_zord = zord + 100  # first line should be on top
            else:
                label = None
                label_eff = None
                eff_zord = 1
            if ri in rad_un_idx:
                ax.plot(
                    r11,
                    timarr,
                    h2,
                    zorder=eff_zord,
                    color="k",
                    alpha=alpha,
                    label=label_eff,
                )
            ax.quiver(
                r1,
                time,
                h,
                xydir,
                xydir,
                h1 - h,
                alpha=0.6,
                arrow_length_ratio=0.0,
                color=color,
                zorder=zord + 30,
            )
            ax.scatter(
                r1,
                time,
                h1,
                depthshade=False,
                zorder=zord + 60,
                color=color,
                label=label,
            )

        for te in time:
            t11 = te * t_gen
            h = plot_f(**{extra["time"]: te, extra["rad"]: radarr}).reshape(-1)
            ax.plot(radarr, t11, h, color="k", alpha=0.1, linestyle="--")

        ax.view_init(elev=40, azim=125)
        ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
        ax.set_ylabel(r"$t$ in $\left[\mathrm{s}\right]$")
        ax.set_zlabel(r"$\tilde{h}$ in $\left[\mathrm{m}\right]$")
        _sort_lgd(ax, loc="upper right", markerscale=2)
        fig.tight_layout()
        if plotname is not None:
            fig.savefig(plotname, format="pdf")

    return ax


def plotfit_steady(
    setup,
    data,
    para,
    rad,
    radnames,
    extra,
    plotname=None,
    ax_ins=True,
    fig=None,
    ax=None,
    style="WTP",
):
    """Plot of steady estimation fitting."""
    val_fix = setup.val_fix
    val_fix.pop(extra["rad"], None)

    para_ordered = np.empty(len(setup.para_names))
    for i, name in enumerate(setup.para_names):
        para_ordered[i] = para[name]
    para_kw = setup.get_sim_kwargs(para_ordered)
    val_fix.update(para_kw)

    plot_f = ft.partial(setup.func, **val_fix)
    radarr = np.linspace(rad.min(), rad.max(), 100)

    test_name = list(np.unique(radnames[:, 0]))
    test_name.sort()

    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clr_n = len(clrs)
        fig, ax = _get_fig_ax(fig, ax, figsize=(9, 6))
        if ax_ins:
            axins = ax.inset_axes([0.4, 0.07, 0.57, 0.5])
            axins.plot(
                radarr,
                plot_f(**{extra["rad"]: radarr}),
                alpha=0.6,
                color="k",
                zorder=200,
            )
            axins.set_xscale("log")
            axins.set_facecolor("w")
            axins.text(
                0.975,
                0.025,
                "log-radius plot",
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round", ec="k", fc="w"),
                transform=axins.transAxes,
            )
        for ri, re in enumerate(rad):
            h = plot_f(**{extra["rad"]: re}).reshape(-1)
            h1 = data[ri]
            color = clrs[(test_name.index(radnames[ri, 0]) + 2) % clr_n]
            if radnames[ri, 0] == radnames[ri, 1]:
                label = "test at '{}'".format(radnames[ri, 0])
            else:
                label = None
            ax.plot([re, re], [h, h1], alpha=0.6, color=color, zorder=100)
            ax.scatter(re, data[ri], color=color, label=label, zorder=300)
            if ax_ins:
                axins.plot(
                    [re, re], [h, h1], alpha=0.6, color=color, zorder=100
                )
                axins.scatter(re, data[ri], color=color, zorder=300)

        ax.plot(
            radarr,
            plot_f(**{extra["rad"]: radarr}),
            alpha=0.6,
            color="k",
            zorder=200,
            label="fitted type curve",
        )
        ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
        ax.set_ylabel(r"$\tilde{h}$ in $\left[\mathrm{m}\right]$")
        _sort_lgd(ax, loc="upper left", bbox_to_anchor=(1, 1), markerscale=2)
        fig.tight_layout()
        if plotname is not None:
            fig.savefig(plotname, format="pdf")

    return ax


def plotparainteract(result, paranames, plotname=None, fig=None, style="WTP"):
    """Plot of parameter interaction."""
    import pandas as pd

    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "default"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig, ax = _get_fig_ax(fig, ax=None, figsize=(12, 12))
        fields = [par for par in result.dtype.names if par.startswith("par")]
        parameterdistribtion = result[fields]
        df = pd.DataFrame(
            np.asarray(parameterdistribtion).T.tolist(), columns=paranames
        )
        with warnings.catch_warnings():
            # We know that fig is resetted, but we need to give ax to set fig
            warnings.simplefilter("ignore", UserWarning)
            if len(paranames) > 1:
                pd.plotting.scatter_matrix(
                    df, alpha=0.2, ax=ax, diagonal="kde"
                )
            else:
                df.plot.kde(ax=ax)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.1)
        if plotname is not None:
            fig.savefig(plotname, format="pdf")
    return fig


def plotparatrace(
    result,
    parameternames=None,
    parameterlabels=None,
    xticks=None,
    stdvalues=None,
    plotname=None,
    fig=None,
    style="WTP",
):
    """Plot of parameter trace."""
    rep = len(result)
    rows = len(parameternames)
    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig = _get_fig_ax(fig, ax=False, figsize=(15, 3 * rows))

        for j in range(rows):
            ax = fig.add_subplot(rows, 1, 1 + j)
            data = result["par" + parameternames[j]]

            ax.plot(data, "-", color=clrs[0])

            if stdvalues is not None:
                ax.plot(
                    [stdvalues[parameternames[j]]] * rep,
                    "--",
                    label="best value: {:04.2f}".format(
                        stdvalues[parameternames[j]]
                    ),
                    color="k",
                    alpha=0.7,
                )
                ax.legend()

            if xticks is None:
                xticks = np.linspace(0, 1, 11) * len(data)

            ax.set_xlim(0, rep)
            ax.set_ylim(
                np.min(data) - 0.1 * np.max(abs(data)),
                np.max(data) + 0.1 * np.max(abs(data)),
            )
            ax.xaxis.set_ticks(xticks)
            ax.set_ylabel(
                parameterlabels[j], rotation=0, fontsize="large", labelpad=10
            )

        fig.tight_layout()
        if plotname is not None:
            fig.savefig(plotname, format="pdf", bbox_inches="tight")
    return fig


def plotsensitivity(
    paralabels, sensitivities, plotname=None, fig=None, ax=None, style="WTP"
):
    """Plot of sensitivity results."""
    style = copy.deepcopy(plt.rcParams) if style is None else style
    keep_fs = False
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig, ax = _get_fig_ax(fig, ax)
        w_props = {"linewidth": 1, "edgecolor": "w", "width": 0.5}
        wedges, __ = ax.pie(
            sensitivities["ST"], wedgeprops=w_props, startangle=90
        )
        lgd = ax.legend(
            wedges,
            paralabels,
            title="Parameters",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        ax.axis("equal")
        fig.suptitle("FAST total sensitivity shares", fontsize="large")
        fig.tight_layout()
        if plotname is not None:
            fig.savefig(
                plotname,
                format="pdf",
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
    return ax
