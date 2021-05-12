"""
welltestpy subpackage to make diagnostic plots.

.. currentmodule:: welltestpy.tools.diagnostic_plots

The following classes and functions are provided

.. autosummary::
   smoothing_derivative
   plot_diagnostic
"""
# pylint: disable=C0103
import copy
import warnings
import functools as ft

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




def smoothing_derivative(observation):
    """Calculate the derivative of the drawdown curve.

            Parameters
            ----------
            observation : :class:`welltestpy.data.Observation`
                The observation to calculate the derivative.


            Returns
            ---------
            the derivative of the observed heads.

           """
    # create arrays for the input of head and time.
    head, time = observation()
    head = np.array(head, dtype=float).reshape(-1)
    time = np.array(time, dtype=float).reshape(-1)
    dhead = np.zeros(len(head))
    t = np.arange(len(time))
    for i in t:
        if i == 0:
            continue
        elif i == t[-1]:
            continue
        else:
            # derivative approximation by Bourdet (1989)
            dh = (((head[i]-head[i-1])/(time[i]-time[i-1]) * (time[i + 1]-time[i])) + ((head[i + 1]-head[i])/(time[i + 1]-time[i])* (time[i]-time[i-1])))/((time[i]-time[i-1]) + (time[i + 1]-time[i]))
            dhead[i] = dh

    observation(observation=dhead)
    return observation


def plot_diagnostic(observation, derivative):
    """plot the derivative with the original data.

               Parameters
               ----------
               smoothed_observation
                    the smoothed data

               derivative
                    the calculated derivative

                Returns
                ---------
                Diagnostic plot
          """

    # setting variables
    x = observation[1]
    y = observation[0]
    sx = observation[1]
    sy = observation[0]
    dx = derivative[1]
    dy = derivative[0]

    # plotting
    fig, ax = plt.subplots(dpi=75, figsize=[7.5, 5.5])
    ax.scatter(x, y, marker=".", color="C1", label="datapoints")
    ax.plot(dx, dy, label="observed data")
    ax.plot(sx, sy, color="k", label="time derivative")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim([1, 1e5])
    ax.set_xlabel("$t$ in [s]", fontsize=16)
    ax.set_ylabel("$h$ and $dh/dx$ in [m]", fontsize=16)
    ax.legend(loc="upper left")
    fig.tight_layout()

    fig.show()