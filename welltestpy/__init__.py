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
from __future__ import absolute_import

from welltestpy._version import __version__
from welltestpy import data, estimate, process, tools

__all__ = ["__version__"]
__all__ += ["data", "estimate", "process", "tools"]
