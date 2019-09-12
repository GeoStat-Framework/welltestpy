# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing plotting routines.

.. currentmodule:: welltestpy.tools.plotter

The following classes and functions are provided

.. autosummary::
   Editor
   CampaignPlot
   fadeline
   plotres
   WellPlot
   plotfit_transient
   plotfitting3D
   plotfitting3Dtheis
   plotparainteract
   plotparatrace
   plotsensitivity
"""
from __future__ import absolute_import, division, print_function

import functools as ft

import numpy as np
import anaflow as ana

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# use the ggplot style like R
plt.style.use("ggplot")


def CampaignPlot(campaign, select_test=None, **kwargs):
    """Plotting an overview of the tests within the campaign."""
    if select_test is None:
        tests = list(campaign.tests.keys())
    else:
        tests = select_test

    tests.sort()
    nroftests = len(tests)
    fig = plt.figure(dpi=75, figsize=[8, 3 * nroftests])

    for n, t in enumerate(tests):
        ax = fig.add_subplot(nroftests, 1, n + 1)
        campaign.tests[t]._addplot(ax, campaign.wells)

    if "xscale" in kwargs:
        ax.set_xscale(kwargs["xscale"])
    if "yscale" in kwargs:
        ax.set_yscale(kwargs["yscale"])

    fig.tight_layout()
    plt.show()


####


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

    for i in range(steps):
        if i == 0:
            label0 = label
        else:
            label0 = None
        ax.plot(
            [xarr[i], xarr[i + 1]],
            [yarr[i], yarr[i + 1]],
            label=label0,
            color=color,
            alpha=(steps - i) * (1.0 / steps),
            **kwargs
        )


def plotres(res, names=None, title="", filename=None, plot_well_names=True):
    """Plots all solutions in res and label the points with the names."""
    # calculate Column- and Row-count for quadratic shape of the plot
    # total number of plots
    Tot = len(res)
    # columns near the square-root but tendentially wider than tall
    Cols = int(np.ceil(np.sqrt(Tot)))
    # enough rows to catch all plots
    Rows = int(np.ceil(Tot / Cols))
    # Possition numbers as array
    Pos = np.arange(Tot) + 1

    # generate names for points if undefined
    if names is None:
        names = []
        for i in range(len(res[0])):
            names.append("p" + str(i))

    # genearte commen borders for all plots
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf

    for i in res:
        for j in i:
            xmax = max(j[0], xmax)
            xmin = min(j[0], xmin)
            ymax = max(j[1], ymax)
            ymin = min(j[1], ymin)

    # add some space around the points in the plot
    space = 0.1 * max(abs(xmax - xmin), abs(ymax - ymin))
    xspace = yspace = space

    fig = plt.figure(dpi=75, figsize=[9 * Cols, 5 * Rows])
    # fig.suptitle("well locations and pumping tests at " + title, fontsize=18)

    for i, result in enumerate(res):
        ax = fig.add_subplot(Rows, Cols, Pos[i])
        ax.set_xlim([xmin - xspace, xmax + xspace])
        ax.set_ylim([ymin - yspace, ymax + yspace])
        ax.set_aspect("equal")

        for j, name in enumerate(names):
            ax.scatter(result[j][0], result[j][1], color="k", zorder=10)
            if plot_well_names:
                ax.annotate("  " + name, (result[j][0], result[j][1]))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_xlabel("x distance in $[m]$")  # , fontsize=16)
        ax.set_ylabel("y distance in $[m]$")  # , fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if filename is not None:
        plt.savefig(filename, format="pdf")

    return fig, ax


######


def WellPlot(campaign, plot_tests=True, plot_well_names=True):
    """Plotting of the wellconstellation within the campaign."""
    res0 = []
    names = []

    for w in campaign.wells:
        res0.append(
            [
                campaign.wells[w].coordinates[0],
                campaign.wells[w].coordinates[1],
            ]
        )
        names.append(w)
    res = [res0]

    fig, ax = plotres(
        res, names, campaign.name, plot_well_names=plot_well_names
    )

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
                    linestyle=":",
                )
    # get equal axis (for realism)
    ax.axis("equal")
    ax.legend()
    plt.show()
    return fig, ax


# Estimation plotting


def plotfit_transient(setup, data, para, rad, time, radnames, plotname, extra):
    """Plot of transient estimation fitting."""
    val_fix = setup.val_fix
    for kwarg in ["time", "rad"]:
        val_fix.pop(extra[kwarg], None)

    para_kw = setup.get_sim_kwargs(para)
    val_fix.update(para_kw)

    plot_f = ft.partial(setup.func, **val_fix)

    radarr = np.linspace(rad.min(), rad.max(), 100)
    timarr = np.linspace(time.min(), time.max(), 100)

    plt.style.use("default")

    t_gen = np.ones_like(radarr)
    r_gen = np.ones_like(time)
    r_gen1 = np.ones_like(timarr)
    xydir = np.zeros_like(time)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection=Axes3D.name)

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
    ax.set_zlabel(r"$h/|Q|$ in $\left[\mathrm{m}\right]$")
    ax.legend(loc="lower left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(plotname, format="pdf")


def plotfitting3D(
    data, para, rad, time, radnames, prate, plotname, rwell=0.0, rinf=np.inf
):
    """Plot of estimation fitting."""
    radarr = np.linspace(rad.min(), rad.max(), 100)
    timarr = np.linspace(time.min(), time.max(), 100)

    plt.style.use("default")

    t_gen = np.ones_like(radarr)
    r_gen = np.ones_like(time)
    r_gen1 = np.ones_like(timarr)
    xydir = np.zeros_like(time)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection=Axes3D.name)

    for ri, re in enumerate(rad):
        r1 = re * r_gen
        r11 = re * r_gen1

        h = ana.ext_theis2D(
            time=time,
            rad=re,
            TG=np.exp(para[0]),
            sig2=para[1],
            corr=para[2],
            S=np.exp(para[3]),
            Qw=prate,
            rwell=rwell,
            rinf=rinf,
        ).reshape(-1)
        h1 = data[:, ri]
        h2 = ana.ext_theis2D(
            time=timarr,
            rad=re,
            TG=np.exp(para[0]),
            sig2=para[1],
            corr=para[2],
            S=np.exp(para[3]),
            Qw=prate,
            rwell=rwell,
            rinf=rinf,
        ).reshape(-1)

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
        h = ana.ext_theis2D(
            time=te,
            rad=radarr,
            TG=np.exp(para[0]),
            sig2=para[1],
            corr=para[2],
            S=np.exp(para[3]),
            Qw=prate,
            rwell=rwell,
            rinf=rinf,
        ).reshape(-1)
        ax.plot(radarr, t11, h, color="k", alpha=0.1, linestyle="--")

    ax.view_init(elev=45, azim=155)
    ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
    ax.set_ylabel(r"$t$ in $\left[\mathrm{s}\right]$")
    ax.set_zlabel(r"$h/|Q|$ in $\left[\mathrm{m}\right]$")
    ax.legend(loc="lower left", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(plotname, format="pdf")


def plotfitting3Dtheis(data, para, rad, time, radnames, prate, plotname):
    """Plot of estimation fitting with theis."""
    radarr = np.linspace(rad.min(), rad.max(), 100)
    timarr = np.linspace(time.min(), time.max(), 100)

    plt.style.use("default")

    t_gen = np.ones_like(radarr)
    r_gen = np.ones_like(time)
    r_gen1 = np.ones_like(timarr)
    xydir = np.zeros_like(time)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection="3d")

    for ri, re in enumerate(rad):
        r1 = re * r_gen
        r11 = re * r_gen1

        h = ana.theis(
            time=time, rad=re, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
        ).reshape(-1)
        h1 = data[:, ri]
        h2 = ana.theis(
            time=timarr, rad=re, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
        ).reshape(-1)

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
        h = ana.theis(
            time=te, rad=radarr, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
        ).reshape(-1)
        ax.plot(radarr, t11, h, color="k", alpha=0.1, linestyle="--")

    ax.view_init(elev=45, azim=155)
    ax.set_xlabel(r"$r$ in $\left[\mathrm{m}\right]$")
    ax.set_ylabel(r"$t$ in $\left[\mathrm{s}\right]$")
    ax.set_zlabel(r"$h/|Q|$ in $\left[\mathrm{m}\right]$")
    ax.legend(loc="lower left")
    #    plt.tight_layout()
    plt.savefig(plotname, format="pdf")


def plotparainteract(result, paranames, plotname):
    """Plot of parameter interaction."""
    import pandas as pd

    fields = [word for word in result.dtype.names if word.startswith("par")]
    parameterdistribtion = result[fields]
    df = pd.DataFrame(
        np.asarray(parameterdistribtion).T.tolist(), columns=paranames
    )
    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal="kde")
    plt.savefig(plotname, format="pdf")


def plotparatrace(
    result,
    parameternames=None,
    parameterlabels=None,
    xticks=None,
    stdvalues=None,
    filename="test.pdf",
):
    """Plot of parameter trace."""
    rep = len(result)
    rows = len(parameternames)
    fig = plt.figure(figsize=(15, 3 * rows))

    for j in range(rows):
        ax = plt.subplot(rows, 1, 1 + j)
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
        ax.set_ylabel(parameterlabels[j], rotation=0, fontsize=16, labelpad=10)

    plt.tight_layout()
    fig.savefig(filename, format="pdf", bbox_inches="tight")


def plotsensitivity(paralabels, sensitivities, plotname):
    """Plot of sensitivity results."""
    __, ax = plt.subplots()
    ax.bar(
        range(len(paralabels)),
        sensitivities["ST"],
        color="C1",
        alpha=0.8,
        align="center",
    )
    ax.set_ylabel(r"FAST total-sensitivity")
    ax.set_ylim([-0.1, 1.1])
    plt.xticks(range(len(paralabels)), paralabels)
    plt.title("Sensitivity", fontsize=16)
    plt.savefig(plotname, format="pdf")
    return ax
