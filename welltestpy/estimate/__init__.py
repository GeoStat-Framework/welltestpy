# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing routines to estimate pump test parameters.

.. currentmodule:: welltestpy.estimate

Subpackages
^^^^^^^^^^^

The following subpackages are provided

.. autosummary::
    estimators
    spotpylib
    steady_lib
    transient_lib

Estimators
^^^^^^^^^^

The following estimators are provided

.. autosummary::
    ExtTheis3D
    ExtTheis2D
    Neuman2004
    Theis
    ExtThiem3D
    ExtThiem2D
    Neuman2004Steady
    Thiem
"""
from . import estimators, spotpylib, steady_lib, transient_lib

from .estimators import (
    ExtTheis3D,
    ExtTheis2D,
    Neuman2004,
    Theis,
    ExtThiem3D,
    ExtThiem2D,
    Neuman2004Steady,
    Thiem,
)

__all__ = ["estimators", "spotpylib", "steady_lib", "transient_lib"]
__all__ += [
    "ExtTheis3D",
    "ExtTheis2D",
    "Neuman2004",
    "Theis",
    "ExtThiem3D",
    "ExtThiem2D",
    "Neuman2004Steady",
    "Thiem",
]
