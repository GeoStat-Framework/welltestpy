"""
==========
welltestpy
==========

Contents
--------
WellTestPy provides a framework to handle and plot data from well based
field campaigns as well as a data interpretation module.

Subpackages
-----------
The following subpackages are provided

.. autosummary::
    data
    estimate
    process
    tools
"""
from __future__ import absolute_import

from welltestpy import (
    data,
    estimate,
    process,
    tools,
)

__all__ = [
    "data",
    "estimate",
    "process",
    "tools",
]

__version__ = '0.1.1'
