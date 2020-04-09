# -*- coding: utf-8 -*-
"""
Purpose
=======

welltestpy provides a framework to handle and plot data from well based
field campaigns as well as a parameter estimation module.

Subpackages
^^^^^^^^^^^

.. autosummary::
    data
    estimate
    process
    tools

Classes
^^^^^^^

Campaign classes
~~~~~~~~~~~~~~~~

.. currentmodule:: welltestpy.data.campaignlib

The following classes can be used to handle field campaigns.

.. autosummary::
    Campaign
    FieldSite

Field Test classes
~~~~~~~~~~~~~~~~~~

.. currentmodule:: welltestpy.data.testslib

The following classes can be used to handle field test within a campaign.

.. autosummary::
    PumpingTest

Loading routines
^^^^^^^^^^^^^^^^

.. currentmodule:: welltestpy.data.data_io

Campaign related loading routines

.. autosummary::
    load_campaign
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
