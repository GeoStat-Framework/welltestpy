# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing routines to estimate subsurface parameters
from test data.

.. currentmodule:: welltestpy.estimate

Subpackages
^^^^^^^^^^^

The following subpackages are provided

.. autosummary::
    estimatelib
    spotpy_classes

Included classes
^^^^^^^^^^^^^^^^

The following classes are provided

.. autosummary::
    Stat2Dest
    Theisest
"""
from __future__ import absolute_import

from welltestpy.estimate import estimatelib, spotpy_classes

from welltestpy.estimate.estimatelib import Stat2Dest, Theisest

__all__ = ["Stat2Dest", "Theisest", "estimatelib", "spotpy_classes"]
