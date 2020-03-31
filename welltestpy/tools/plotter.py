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
from __future__ import absolute_import, division, print_function

import warnings
import functools as ft

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# use the ggplot style like R
plt.style.use("ggplot")


def _get_fig_ax(
    fig=None,
    ax=None,
    ax_name="rectilinear",
    sub_args=None,
    sub_kwargs=None,
    **fig_kwargs
):  # pragma: no cover
    # 0->None or given, 1->False, 2->True
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
    elif ax_case == 1:
        return plt.figure(**fig_kwargs) if fig is None else fig
    # if ax=True we want the current axis of the given figure
    assert fig is not None
    return fig, fig.gca()


def fadeline(ax, x, y, label=None, color=None, steps=20, **kwargs):
    """Fading line for matplotlib.

    This is a workaround to produce a fading line.

    Parameters
    ----------
    ax : axis
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
        if i == 0:
            label0 = label
        else:
            label0 = None
        ax.plot(
            [xarr[i], xarr[i + 1]],
            [yarr[i], yarr[i + 1]],
            label=label0,
            alpha=(steps - i) * (1.0 / steps) * 0.9 + 0.1,
            **kwargs
        )


def campaign_plot(campaign, select_test=None, fig=None, **kwargs):
    """Plot an overview of the tests within the campaign."""
    if select_test is None:
        tests = list(campaign.tests.keys())
    else:
        tests = select_test

    tests.sort()
    nroftests = len(tests)
    fig = _get_fig_ax(fig, ax=False, dpi=75, figsize=[8, 3 * nroftests])

    for n, t in enumerate(tests):
        ax = fig.add_subplot(nroftests, 1, n + 1)
        # call the plotting routine of the test
        campaign.tests[t].plot(wells=campaign.wells, ax=ax)

    if "xscale" in kwargs:
        ax.set_xscale(kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(kwargs["yscale"])

    fig.tight_layout()
    fig.show()
    return fig


def campaign_well_plot(
    campaign, plot_tests=True, plot_well_names=True, fig=None
):
    """Plot of the well constellation within the campaign."""
    well_const0 = []
    names = []

    for w in campaign.wells:
        well_const0.append(
            [
                campaign.wells[w].coordinates[0],
                campaign.wells[w].coordinates[1],
            ]
        )
        names.append(w)
    well_const = [well_const0]

    fig = plot_well_pos(
        well_const,
        names,
        campaign.name,
        plot_well_names=plot_well_names,
        fig=fig,
    )

    fig, ax = _get_fig_ax(fig, ax=True)

    if plot_tests:
        testlist = list(campaign.tests.keys())
        testlist.sort()
        for i, t in enumerate(testlist):
            for j, obs in enumerate(campaign.tests[t].observations):
                x0 = campaign.wells[campaign.tests[t].pumpingwell].coordinates[
                    0
                ]
                y0 = campaign.wells[campaign.tests[t].pumpingwell].coordinates[
                    1
                ]
                x1 = campaign.wells[obs].coordinates[0]
                y1 = campaign.wells[obs].coordinates[1]
                if j == 0:
                    label = "test at " + t
                else:
                    label = None
                fadeline(
                    ax,
                    [x0, x1],
                    [y0, y1],
                    label,
                    "C" + str((i + 2) % 10),
                    linewidth=3,
                )
    # get equal axis (for realism)
    ax.axis("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.show()
    return ax


def plot_pump_test(pump_test, wells, exclude=None, fig=None, ax=None):
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
    ax : :class:`Axes`
        Axes where the plot should be done.

    Notes
    -----
    This will be used by the Campaign class.
    """
    fig, ax = _get_fig_ax(fig, ax)
    exclude = set() if exclude is None else set(exclude)
    well_set = set(wells)
    test_wells = set(pump_test.observationwells)
    plot_wells = list((well_set & test_wells) - exclude)
    plot_wells.sort()  # sort by name
    state = pump_test.state(wells=plot_wells)
    steady_guide_x = []
    steady_guide_y = []
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
        return
    for i, k in enumerate(plot_wells):
        if k != pump_test.pumpingwell:
            dist = wells[k] - wells[pump_test.pumpingwell]
        else:
            dist = wells[pump_test.pumpingwell].radius
        if pump_test.observations[k].state == "transient":
            if pump_test.pumpingrate > 0:
                displace = np.maximum(pump_test.observations[k].value[1], 0.0)
            else:
                displace = np.minimum(pump_test.observations[k].value[1], 0.0)
            ax1.plot(
                pump_test.observations[k].value[0],
                displace,
                linewidth=2,
                color="C{}".format(i % 10),
                label=(
                    pump_test.observations[k].name + " r={:1.2f}".format(dist)
                ),
            )
            ax1.set_xlabel(pump_test.observations[k].labels[0])
            ax1.set_ylabel(pump_test.observations[k].labels[1])
        else:
            steady_guide_x.append(dist)
            steady_guide_y.append(pump_test.observations[k].value)
            label = pump_test.observations[k].name + " r={:1.2f}".format(dist)
            color = "C{}".format(i % 10)
            ax2.scatter(
                dist, pump_test.observations[k].value, color=color, label=label
            )
            ax2.set_xlabel("r in {}".format(wells[k]._coordinates.units))
            ax2.set_ylabel(pump_test.observations[k].labels)

    if state != "transient":
        steady_guide_x = np.array(steady_guide_x, dtype=float)
        steady_guide_y = np.array(steady_guide_y, dtype=float)
        arg = np.argsort(steady_guide_x)
        steady_guide_x = steady_guide_x[arg]
        steady_guide_y = steady_guide_y[arg]
        ax2.plot(steady_guide_x, steady_guide_y, color="k", alpha=0.1)

    ax.set_title(repr(pump_test))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fancybox=True,
        framealpha=0.75,
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
    fig=None,
):
    """Plot all well constellations and label the points with the names."""
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

    fig = _get_fig_ax(fig, ax=False, dpi=75, figsize=[9 * col_n, 5 * row_n])

    for i, wells in enumerate(well_const):
        ax = fig.add_subplot(row_n, col_n, pos_tuple[i])
        ax.set_xlim([xmin - xspace, xmax + xspace])
        ax.set_ylim([ymin - yspace, ymax + yspace])
        ax.set_aspect("equal")

        for j, name in enumerate(names):
            ax.scatter(wells[j][0], wells[j][1], color="k", zorder=10)
            if plot_well_names:
                ax.annotate("  " + name, (wells[j][0], wells[j][1]))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_xlabel("x distance in $[m]$")  # , fontsize=16)
        ax.set_ylabel("y distance in $[m]$")  # , fontsize=16)
        if total_n > 1:
            ax.set_title("Result {}".format(i))

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
):
    """Plot of transient estimation fitting."""
    plt.style.use("default")
    fig, ax = _get_fig_ax(fig, ax, ax_name=Axes3D.name, figsize=(12, 8))
    val_fix = setup.val_fix
    for kwarg in ["time", "rad"]:
        val_fix.pop(extra[kwarg], None)

    para_kw = setup.get_sim_kwargs(para)
    val_fix.update(para_kw)

    plot_f = ft.partial(setup.func, **val_fix)

    radarr = np.linspace(rad.min(), rad.max(), 100)
    timarr = np.linspace(time.min(), time.max(), 100)

    t_gen = np.ones_like(radarr)
    r_gen = np.ones_like(time)
    r_gen1 = np.ones_like(timarr)
    xydir = np.zeros_like(time)

    for ri, re in enumerate(rad):
        r1 = re * r_gen
        r11 = re * r_gen1

        h = plot_f(**{extra["time"]: time, extra["rad"]: re}).reshape(-1)
        h1 = data[:, ri]
        h2 = plot_f(**{extra["time"]: timarr, extra["rad"]: re}).reshape(-1)

        zord = 1000 * (len(rad) - ri)

        ax.plot(
            r11,
            timarr,
            h2,
            label=radnames[ri] + " r={:04.2f}".format(re),
            zorder=zord,
        )
        ax.quiver(
            r1,
            time,
            h,
            xydir,
            xydir,
            h1 - h,
            alpha=0.5,
            arrow_length_ratio=0.0,
            color="C" + str(ri % 10),
            zorder=zord,
        )
        ax.scatter(r1, time, h1, depthshade=False, zorder=zord)

    for te in time:
        t11 = te * t_gen
        h = plot_f(**{extra["time"]: te, extra["rad"]: radarr}).reshape(-1)
        ax.plot(radarr, t11, h, color="k", alpha=0.1, linestyle="--")

    ax.view_init(elev=45, azim=155)
    ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
    ax.set_ylabel(r"$t$ in $\left[\mathrm{s}\right]$")
    ax.set_zlabel(r"$\tilde{h}$ in $\left[\mathrm{m}\right]$")
    ax.legend(loc="lower left", fontsize="small")
    fig.tight_layout()
    if plotname is not None:
        fig.savefig(plotname, format="pdf")
    plt.style.use("ggplot")
    return ax


def plotfit_steady(
    setup, data, para, rad, radnames, extra, plotname=None, fig=None, ax=None
):
    """Plot of steady estimation fitting."""
    val_fix = setup.val_fix
    val_fix.pop(extra["rad"], None)

    para_kw = setup.get_sim_kwargs(para)
    val_fix.update(para_kw)

    plot_f = ft.partial(setup.func, **val_fix)
    radarr = np.linspace(rad.min(), rad.max(), 100)

    fig, ax = _get_fig_ax(fig, ax, figsize=(9, 6))

    for ri, re in enumerate(rad):
        ax.scatter(re, data[ri], label=radnames[ri])

    ax.plot(radarr, plot_f(**{extra["rad"]: radarr}), "--", color="k")
    ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
    ax.set_ylabel(r"$h/|Q|$ in $\left[\mathrm{m}\right]$")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if plotname is not None:
        fig.savefig(plotname, format="pdf")
    return ax


def plotparainteract(result, paranames, plotname=None, fig=None):
    """Plot of parameter interaction."""
    import pandas as pd

    plt.style.use("default")
    fig, ax = _get_fig_ax(fig, ax=None, figsize=(12, 12))
    fields = [word for word in result.dtype.names if word.startswith("par")]
    parameterdistribtion = result[fields]
    df = pd.DataFrame(
        np.asarray(parameterdistribtion).T.tolist(), columns=paranames
    )
    with warnings.catch_warnings():
        # We know that the fig is resetted, but we need to give ax to pass fig
        warnings.simplefilter("ignore", UserWarning)
        pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax, diagonal="kde")

    fig.tight_layout()
    if plotname is not None:
        fig.savefig(plotname, format="pdf")
    plt.style.use("ggplot")
    return fig


def plotparatrace(
    result,
    parameternames=None,
    parameterlabels=None,
    xticks=None,
    stdvalues=None,
    plotname=None,
    fig=None,
):
    """Plot of parameter trace."""
    rep = len(result)
    rows = len(parameternames)
    fig = _get_fig_ax(fig, ax=False, figsize=(15, 3 * rows))

    for j in range(rows):
        ax = fig.add_subplot(rows, 1, 1 + j)
        data = result["par" + parameternames[j]]

        ax.plot(data, "-", color="C0")

        if stdvalues is not None:
            ax.plot(
                [stdvalues[j]] * rep,
                "--",
                label="best value: {:04.2f}".format(stdvalues[j]),
                color="C1",
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
    paralabels, sensitivities, plotname=None, fig=None, ax=None
):
    """Plot of sensitivity results."""
    plt.style.use("ggplot")
    fig, ax = _get_fig_ax(fig, ax)
    explode = np.full_like(sensitivities["ST"], 0.1)
    wedges, __ = ax.pie(
        sensitivities["ST"], explode=explode, shadow=True, startangle=90
    )
    lgd = ax.legend(
        wedges,
        paralabels,
        title="Parameter",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax.axis("equal")
    fig.tight_layout()
    fig.suptitle("FAST total sensitivity shares", fontsize="large")
    if plotname is not None:
        fig.savefig(
            plotname,
            format="pdf",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
    return ax
