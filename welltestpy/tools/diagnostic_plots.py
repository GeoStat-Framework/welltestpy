"""
welltestpy subpackage to make diagnostic plots.

.. currentmodule:: welltestpy.tools.diagnostic_plots

The following classes and functions are provided

.. autosummary::
   diagnostic_plot_pump_test

"""
# pylint: disable=C0103


import numpy as np

from ..process import processlib
from . import plotter

import matplotlib.pyplot as plt


def diagnostic_plot_pump_test(
    observation,
    rate,
    method="bourdet",
    linthresh_time=1.0,
    linthresh_head=1e-5,
    fig=None,
    ax=None,
    plotname=None,
    style="WTP",
):
    """plot the derivative with the original data.

    Parameters
    ----------
    observation : :class:`welltestpy.data.Observation`
        The observation to calculate the derivative.
    rate : :class:`float`
        Pumping rate.
    method : :class:`str`, optional
        Method to calculate the time derivative.
        Default: "bourdet"
    linthresh_time : :class: 'float'
        Range of time around 0 that behaves linear.
        Default: 1
    linthresh_head : :class: 'float'
        Range of head values around 0 that behaves linear.
        Default: 1e-5
    fig : Figure, optional
        Matplotlib figure to plot on.
        Default: None.
    ax : :class:`Axes`
        Matplotlib axes to plot on.
        Default: None.
    plotname : str, optional
        Plot name if the result should be saved.
        Default: None.
    style : str, optional
        Plot style.
        Default: "WTP".

     Returns
     ---------
     Diagnostic plot
    """
    head, time = observation()
    head = np.array(head, dtype=float).reshape(-1)
    time = np.array(time, dtype=float).reshape(-1)
    if rate < 0:
        head = head * -1
    derivative = processlib.smoothing_derivative(
        head=head, time=time, method=method
    )
    # setting variables
    dx = time[1:-1]
    dy = derivative[1:-1]

    # plotting
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig, ax = plotter._get_fig_ax(fig, ax)
        ax.scatter(time, head, color="C0", label="drawdown")
        ax.plot(dx, dy, color="C1", label="time derivative")
        ax.set_xscale("symlog", linthresh=linthresh_time)
        ax.set_yscale("symlog", linthresh=linthresh_head)
        ax.set_xlabel("$t$ in [s]", fontsize=16)
        ax.set_ylabel("$h$ and $dh/dx$ in [m]", fontsize=16)
        lgd = ax.legend(loc="upper left", facecolor="w")
        min_v = min(np.min(head), np.min(dy))
        max_v = max(np.max(head), np.max(dy))
        max_e = int(np.ceil(np.log10(max_v)))
        if min_v < linthresh_head:
            min_e = -np.inf
        else:
            min_e = int(np.floor(np.log10(min_v)))
        ax.set_ylim(10.0 ** min_e, 10.0 ** max_e)
        yticks = [0 if min_v < linthresh_head else 10.0 ** min_e]
        thresh_e = int(np.floor(np.log10(linthresh_head)))
        first_e = thresh_e if min_v < linthresh_head else (min_e + 1)
        yticks += list(10.0 ** np.arange(first_e, max_e + 1))
        ax.set_yticks(yticks)
        fig.tight_layout()
        if plotname is not None:
            fig.savefig(
                plotname,
                format="pdf",
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
    return ax
