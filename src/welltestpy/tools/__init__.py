"""
welltestpy subpackage providing miscellaneous tools.

.. currentmodule:: welltestpy.tools

Included functions
^^^^^^^^^^^^^^^^^^

The following functions are provided for point triangulation

.. autosummary::
   :toctree:

   triangulate
   sym

The following plotting routines are provided

.. autosummary::
   :toctree:

   campaign_plot
   fadeline
   plot_well_pos
   campaign_well_plot
   plotfit_transient
   plotfit_steady
   plotparainteract
   plotparatrace
   plotsensitivity
   diagnostic_plot_pump_test
"""
from . import diagnostic_plots, plotter, trilib
from .diagnostic_plots import diagnostic_plot_pump_test
from .plotter import (
    campaign_plot,
    campaign_well_plot,
    fadeline,
    plot_well_pos,
    plotfit_steady,
    plotfit_transient,
    plotparainteract,
    plotparatrace,
    plotsensitivity,
)
from .trilib import sym, triangulate

__all__ = [
    "triangulate",
    "sym",
    "campaign_plot",
    "fadeline",
    "plot_well_pos",
    "campaign_well_plot",
    "plotfit_transient",
    "plotfit_steady",
    "plotparainteract",
    "plotparatrace",
    "plotsensitivity",
    "diagnostic_plot_pump_test",
]
__all__ += ["plotter", "trilib", "diagnostic_plots"]
