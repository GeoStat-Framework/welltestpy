# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing miscellaneous tools.

.. currentmodule:: welltestpy.tools

Subpackages
^^^^^^^^^^^

The following subpackages are provided

.. autosummary::
    plotter
    trilib

Included functions
^^^^^^^^^^^^^^^^^^

The following classes and functions are provided

.. autosummary::
   triangulate
   sym
   campaign_plot
   fadeline
   plot_well_pos
   campaign_well_plot
   plotfit_transient
   plotfit_steady
   plotparainteract
   plotparatrace
   plotsensitivity
"""
from __future__ import absolute_import

try:
    import StringIO

    BytIO = StringIO.StringIO
except ImportError:
    import io

    BytIO = io.BytesIO

from welltestpy.tools import plotter, trilib

from welltestpy.tools.trilib import triangulate, sym

from welltestpy.tools.plotter import (
    campaign_plot,
    fadeline,
    plot_well_pos,
    campaign_well_plot,
    plotfit_transient,
    plotfit_steady,
    plotparainteract,
    plotparatrace,
    plotsensitivity,
)

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
    "plotter",
    "trilib",
    "BytIO",
]
