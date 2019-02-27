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
   plotfitting3D
   plotfitting3Dtheis
   plotparainteract
   plotparatrace
   plotsensitivity
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import anaflow as ana

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# from matplotlib.widgets import TextBox

# suppress the standard keybinding 's' for saving
plt.rcParams["keymap.save"] = ""
# use the ggplot style like R
plt.style.use("ggplot")


class Editor(object):
    """A 2D line editor

    Mouse-bindings
    --------------
      'left/right-click'
          select vertex
      'left-click-n-drag'
          move vertex vertically
      'right-click-n-drag'
          move vertex freely

    Key-bindings
    ------------
        'a'
            reset all vertices
        'r'
            reset the selected vertex
        'd'
            delete the selected vertex
        'm'
            set the selected vertex into middle of its neighbors
        'n'
            set the selected vertex vertically in line with its neighbors
        's'
            save the xy-data

    Text-input
    ----------
        'Enter' :
            save the selected observation to the given file name.
    """

    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, *observ):
        """Editor initialisation.

        Parameters
        ----------
        *observ
            list of observations of type :class:`welltestpy.data.Observation`
        """

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(observ[0].labels[0])
        self.ax.set_ylabel(observ[0].labels[1])
        #        plt.subplots_adjust(bottom=0.2)
        #        self.axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        #        self.text_box = TextBox(self.axbox, 'save-path', initial="observ")
        #        self.text_box.on_submit(self.submit)
        # self.linesave = []

        self.linesave = observ

        self.linesori = []
        for obs in self.linesave:
            self.linesori.append([obs.time, obs.observation])

        # this is just to use the atomatic dimensioning of matplotlib
        for x, y in self.linesori:
            self.ax.plot(x, y, visible=False)

        self.canvas = self.fig.canvas

        self.lines = []

        for i, xy in enumerate(self.linesori):
            # cycle through the colors
            c = "C" + str(i)
            # create line_objects
            self.lines.append(
                Line2D(xy[0], xy[1], marker="o", color=c, animated=True)
            )
            # add lines to Axes
            self.ax.add_line(self.lines[-1])

        # generate the selector Point
        self.selector = Line2D(
            [0.0],
            [0.0],
            markersize=10,
            marker="o",
            color="k",
            alpha=0.5,
            animated=True,
            visible=False,
        )
        self.ax.add_line(self.selector)

        self._indl = None  # the active line
        self._indv = None  # the active vert

        # connect the interactive functions to the canvas
        self.canvas.mpl_connect("draw_event", self.draw_callback)
        self.canvas.mpl_connect("key_press_event", self.key_press_callback)
        self.canvas.mpl_connect(
            "button_press_event", self.button_press_callback
        )
        self.canvas.mpl_connect(
            "button_release_event", self.button_release_callback
        )
        self.canvas.mpl_connect(
            "motion_notify_event", self.motion_notify_callback
        )
        #        plt.show(block=False)
        plt.show()

    def submit(self, text):
        """submit action for the text-box"""
        if self._indl is not None:
            self.save_xy()
            self.linesave[self._indl].save(name=text)
            print("saved observation no {}".format(self._indl))
        else:
            print("no observation selected")

    def save_xy(self):
        """save the actual data"""
        print("saving...")
        for i, obs in enumerate(self.linesave):
            obs.time = self.lines[i]._x
            obs.observation = self.lines[i]._y

    def draw_callback(self, event):
        """draw everything"""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for line in self.lines:
            self.ax.draw_artist(line)
        self.ax.draw_artist(self.selector)

    def key_press_callback(self, event):
        """whenever a key is pressed"""
        # save all
        if event.key == "s":
            self.save_xy()

        # reset all
        if event.key == "a":
            print("reseting all...")
            self.selector.set_xdata([0.0])
            self.selector.set_ydata([0.0])
            self.selector.set_visible(False)
            for i, xy in enumerate(self.linesori):
                self.lines[i].set_xdata(xy[0])
                self.lines[i].set_ydata(xy[1])
            self.canvas.draw()

        # reset selected
        if event.key == "r" and self._indv is not None:
            print("reseting vertex...")
            self.lines[self._indl]._x[self._indv] = self.linesori[self._indl][
                0
            ][self._indv]
            self.lines[self._indl]._y[self._indv] = self.linesori[self._indl][
                1
            ][self._indv]
            self.selector.set_xdata([self.lines[self._indl]._x[self._indv]])
            self.selector.set_ydata([self.lines[self._indl]._y[self._indv]])
            self.canvas.draw()

        # delete selected
        elif event.key == "d":
            print("deleting vertex...")
            if self._indv is not None and len(self.lines[self._indl]._y) > 1:
                self.lines[self._indl].set_xdata(
                    [
                        tup
                        for i, tup in enumerate(
                            self.lines[self._indl].get_xdata()
                        )
                        if i != self._indv
                    ]
                )
                self.lines[self._indl].set_ydata(
                    [
                        tup
                        for i, tup in enumerate(
                            self.lines[self._indl].get_ydata()
                        )
                        if i != self._indv
                    ]
                )
                self.selector.set_visible(False)
                self.canvas.draw()

        # set selected to middle of neighbors
        if event.key == "m":
            print("averaging vertex...")
            if self.selector.get_visible():
                length = len(self.lines[self._indl]._y)
                if 0 < self._indv < length - 1:
                    self.lines[self._indl]._x[self._indv] = 0.5 * (
                        self.lines[self._indl]._x[self._indv - 1]
                        + self.lines[self._indl]._x[self._indv + 1]
                    )
                    self.lines[self._indl]._y[self._indv] = 0.5 * (
                        self.lines[self._indl]._y[self._indv - 1]
                        + self.lines[self._indl]._y[self._indv + 1]
                    )
                    self.selector.set_xdata(
                        self.lines[self._indl]._x[self._indv]
                    )
                    self.selector.set_ydata(
                        self.lines[self._indl]._y[self._indv]
                    )
                    self.canvas.draw()

        # set selected vertically on line with neighbors
        if event.key == "n":
            print("setting vertex vertically in line...")
            if self.selector.get_visible():
                length = len(self.lines[self._indl]._y)
                if 0 < self._indv < length - 1:
                    self.lines[self._indl]._y[self._indv] = (
                        self.lines[self._indl]._x[self._indv]
                        - self.lines[self._indl]._x[self._indv - 1]
                    ) / (
                        self.lines[self._indl]._x[self._indv + 1]
                        - self.lines[self._indl]._x[self._indv - 1]
                    ) * (
                        self.lines[self._indl]._y[self._indv + 1]
                        - self.lines[self._indl]._y[self._indv - 1]
                    ) + (
                        self.lines[self._indl]._y[self._indv - 1]
                    )
                elif self._indv == 0 and length > 2:
                    self.lines[self._indl]._y[0] = (
                        self.lines[self._indl]._x[0]
                        - self.lines[self._indl]._x[1]
                    ) / (
                        self.lines[self._indl]._x[2]
                        - self.lines[self._indl]._x[1]
                    ) * (
                        self.lines[self._indl]._y[2]
                        - self.lines[self._indl]._y[1]
                    ) + (
                        self.lines[self._indl]._y[1]
                    )
                elif self._indv == length - 1 and length > 2:
                    self.lines[self._indl]._y[self._indv] = (
                        self.lines[self._indl]._x[self._indv]
                        - self.lines[self._indl]._x[self._indv - 2]
                    ) / (
                        self.lines[self._indl]._x[self._indv - 1]
                        - self.lines[self._indl]._x[self._indv - 2]
                    ) * (
                        self.lines[self._indl]._y[self._indv - 1]
                        - self.lines[self._indl]._y[self._indv - 2]
                    ) + (
                        self.lines[self._indl]._y[self._indv - 2]
                    )
                self.selector.set_ydata(self.lines[self._indl]._y[self._indv])
                self.canvas.draw()

    def get_ind_under_point(self, event):
        """get the index of the vertex under
        point if within epsilon tolerance"""
        xy = []
        xyt = []
        xt = []
        yt = []
        d = []
        indseq = []
        ind = []
        # display coords
        for line in self.lines:
            xy.append(np.asarray(line._xy))
            xyt.append(line.get_transform().transform(xy[-1]))
            xt.append(xyt[-1][:, 0])
            yt.append(xyt[-1][:, 1])
            d.append(
                np.sqrt((xt[-1] - event.x) ** 2 + (yt[-1] - event.y) ** 2)
            )
            # select the first nearest vertex of the actual line
            indseq.append(np.nonzero(np.equal(d[-1], np.amin(d[-1])))[0])
            ind.append(indseq[-1][0])
            if d[-1][ind[-1]] >= self.epsilon:
                ind[-1] = None

        # serch for the supreme line of selected indices
        for indl, indv in reversed(list(enumerate(ind))):
            if indv is not None:
                break

        return indl, indv

    def button_press_callback(self, event):
        """whenever a mouse button is pressed"""
        if event.inaxes is None:
            return
        if event.button != 1 and event.button != 3:
            self.selector.set_visible(False)
            self.draw_callback(None)
            self.canvas.draw()
            return
        if event.button == 1 or event.button == 3:
            self._indl, self._indv = self.get_ind_under_point(event)
            if not (self._indl is None or self._indv is None):
                # show the selector
                self.selector.set_xdata(self.lines[self._indl]._x[self._indv])
                self.selector.set_ydata(self.lines[self._indl]._y[self._indv])
                self.selector.set_visible(True)
                self.canvas.draw()
            else:
                # hide the selector
                self.selector.set_xdata(0.0)
                self.selector.set_ydata(0.0)
                self.selector.set_visible(False)
                self.canvas.draw()

    def button_release_callback(self, event):
        """whenever a mouse button is released"""
        return

    def motion_notify_callback(self, event):
        """on mouse movement"""
        if self._indv is None:
            return
        if event.inaxes is None:
            return
        if event.button == 1:
            # just vertical movement with left click
            y = event.ydata

            self.lines[self._indl]._y[self._indv] = y
            self.selector.set_ydata([y])

            # erase actual picture
            self.canvas.restore_region(self.background)
            # draw the updated line
            for line in self.lines:
                self.ax.draw_artist(line)
            self.ax.draw_artist(self.selector)
            # update the canvas with the axis content
            self.canvas.blit(self.ax.bbox)

        if event.button == 3:
            # freely movement with right click
            x, y = event.xdata, event.ydata

            self.lines[self._indl]._xy[self._indv] = x, y
            self.selector.set_xdata([x])
            self.selector.set_ydata([y])

            # erase actual picture
            self.canvas.restore_region(self.background)
            # draw the updated line
            for line in self.lines:
                self.ax.draw_artist(line)
            self.ax.draw_artist(self.selector)
            # update the canvas with the axes content
            self.canvas.blit(self.ax.bbox)


#####


def CampaignPlot(campaign, select_test=None, **kwargs):
    """plotting an overview of the tests within the campaign"""
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
    """Fading line for matplotlib

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


def plotres(res, names=None, title="", filename=None):
    """
    plots all solutions in res and label the points with the names in names
    """

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
    fig.suptitle("well locations and pumping tests at " + title, fontsize=18)

    for i, result in enumerate(res):
        ax = fig.add_subplot(Rows, Cols, Pos[i])
        ax.set_xlim([xmin - xspace, xmax + xspace])
        ax.set_ylim([ymin - yspace, ymax + yspace])
        ax.set_aspect("equal")

        for j, name in enumerate(names):
            ax.scatter(result[j][0], result[j][1], color="k", zorder=10)
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


def WellPlot(campaign, plot_tests=True):
    """Plotting of the wellconstellation within the campaign"""
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

    __, ax = plotres(res, names, campaign.name)

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


# Estimation plotting


def plotfitting3D(
    data, para, rad, time, radnames, prate, plotname, rwell=0.0, rinf=np.inf
):
    """plot of estimation fitting"""
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
            re,
            time,
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
            re,
            timarr,
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
            radarr,
            te,
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
    #    ax.legend(loc="lower left", fontsize='x-small')
    plt.tight_layout()
    plt.savefig(plotname, format="pdf")


def plotfitting3Dtheis(data, para, rad, time, radnames, prate, plotname):
    """plot of estimation fitting with theis"""
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
            re, time, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
        ).reshape(-1)
        h1 = data[:, ri]
        h2 = ana.theis(
            re, timarr, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
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
            radarr, te, T=np.exp(para[0]), S=np.exp(para[1]), Qw=prate
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
    """plot of parameter interaction"""
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
    """plot of parameter trace"""
    rep = len(result)
    rows = len(parameternames)
    fig = plt.figure(figsize=(15, 3 * rows))

    for j in range(rows):
        ax = plt.subplot(rows, 1, 1 + j)
        data = result["par" + parameternames[j]]

        ax.plot(data, "-")

        if stdvalues is None:
            ax.plot([1] * rep, "--")
        else:
            ax.plot([stdvalues[j]] * rep, "--")

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
    """plot of sensitivity results"""
    __, ax = plt.subplots()
    ax.bar(
        range(len(paralabels)),
        sensitivities["ST"],
        color="C1",
        alpha=0.8,
        align="center",
    )
    ax.set_ylabel(r"FAST total-sensitivity")
    plt.xticks(range(len(paralabels)), paralabels)
    plt.title("Sensitivity", fontsize=16)
    plt.savefig(plotname, format="pdf")
