"""
welltestpy - a Python package to handle well-based Field-campaigns.

welltestpy provides a framework to handle and plot data from well based
field campaigns as well as a parameter estimation module.

Subpackages
^^^^^^^^^^^

.. autosummary::
   :toctree: api

   data
   estimate
   process
   tools

Classes
^^^^^^^

Campaign classes
~~~~~~~~~~~~~~~~

.. currentmodule:: welltestpy.data

The following classes can be used to handle field campaigns.

.. autosummary::
   Campaign
   FieldSite

Field Test classes
~~~~~~~~~~~~~~~~~~

The following classes can be used to handle field test within a campaign.

.. autosummary::
   PumpingTest

Loading routines
^^^^^^^^^^^^^^^^

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
from .data.data_io import load_campaign
from .data.testslib import PumpingTest

__all__ = ["__version__"]
__all__ += ["data", "estimate", "process", "tools"]
__all__ += ["Campaign", "FieldSite", "PumpingTest", "load_campaign"]
