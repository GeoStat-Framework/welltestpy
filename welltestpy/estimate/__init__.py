# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing routines to estimate pump test parameters.

.. currentmodule:: welltestpy.estimate

Subpackages
^^^^^^^^^^^

The following subpackages are provided

.. autosummary::
    estimatelib
    spotpy_classes

Estimation classes
^^^^^^^^^^^^^^^^^^

The following estimation classes are provided

.. autosummary::
    TransientPumping
    ExtTheis3D
    ExtTheis2D
    Neuman2004
    Theis
    ExtThiem3D
    ExtThiem2D
    Neuman2004Steady
    Thiem
    TypeCurve
"""
from welltestpy.estimate import estimatelib, spotpy_classes

from welltestpy.estimate.estimatelib import (
    TransientPumping,
    ExtTheis3D,
    ExtTheis2D,
    Neuman2004,
    Theis,
    ExtThiem3D,
    ExtThiem2D,
    Neuman2004Steady,
    Thiem,
)
from welltestpy.estimate.spotpy_classes import TypeCurve

__all__ = [
    "TransientPumping",
    "ExtTheis3D",
    "ExtTheis2D",
    "Neuman2004",
    "Theis",
    "ExtThiem3D",
    "ExtThiem2D",
    "Neuman2004Steady",
    "Thiem",
    "TypeCurve",
    "estimatelib",
    "spotpy_classes",
]
