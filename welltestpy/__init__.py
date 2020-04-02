# -*- coding: utf-8 -*-
"""
Purpose
=======

WellTestPy provides a framework to handle and plot data from well based
field campaigns as well as a data interpretation module.

Subpackages
===========

.. autosummary::
    data
    estimate
    process
    tools
"""
from . import data, estimate, process, tools

try:
    from ._version import __version__
except ImportError:  # pragma: nocover
    # package is not installed
    __version__ = "0.0.0.dev0"

from .data.campaignlib import Campaign, FieldSite
from .data.testslib import PumpingTest
from .data.data_io import load_campaign

__all__ = ["__version__"]
__all__ += ["data", "estimate", "process", "tools"]
__all__ += ["Campaign", "FieldSite", "PumpingTest", "load_campaign"]
