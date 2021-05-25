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


def diagnostic_plot_pump_test(observation, fig=None, ax=None,plotname=None,style="WTP"):
    """plot the derivative with the original data.

               Parameters
               ----------
               observation : :class:`welltestpy.data.Observation`
                    The observation to calculate the derivative.



                Returns
                ---------
                Diagnostic plot
          """
    derivative = processlib.smoothing_derivative(observation)
    head, time = observation()
    head = np.array(head, dtype=float).reshape(-1)
    time = np.array(time, dtype=float).reshape(-1)


    # setting variables
    x = time
    y = head
    sx = time
    sy = head
    dx = time
    dy = derivative

    # plotting
    if style == "WTP":
        style = "ggplot"
        font_size = plt.rcParams.get("font.size", 10.0)
        keep_fs = True
    with plt.style.context(style):
        if keep_fs:
            plt.rcParams.update({"font.size": font_size})
        fig, ax = plotter._get_fig_ax(fig, ax)

    ax.scatter(x, y, marker="+", color="red", label="drawdown")
    ax.plot(sx, sy, c="red")
    ax.plot(dx, dy, c = "black",linestyle='dashed', label="time derivative")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim([1, len(time)])
    ax.set_xlabel("$t$ in [s]", fontsize=16)
    ax.set_ylabel("$h$ and $dh/dx$ in [m]", fontsize=16)
    fig.tight_layout()
    lgd = ax.legend(
        title="Legend",
        loc="upper right",
    )
    if plotname is not None:
        fig.savefig(
            plotname,
            format="pdf",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
    return ax


