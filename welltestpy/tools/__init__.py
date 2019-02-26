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
from __future__ import absolute_import

from welltestpy.tools import plotter, trilib

from welltestpy.tools.trilib import triangulate

from welltestpy.tools.plotter import (
    CampaignPlot,
    fadeline,
    plotres,
    WellPlot,
    plotfitting3D,
    plotfitting3Dtheis,
    plotparainteract,
    plotparatrace,
    plotsensitivity,
)

__all__ = [
    "triangulate",
    "CampaignPlot",
    "fadeline",
    "plotres",
    "WellPlot",
    "plotfitting3D",
    "plotfitting3Dtheis",
    "plotparainteract",
    "plotparatrace",
    "plotsensitivity",
    "plotter",
    "trilib",
]
