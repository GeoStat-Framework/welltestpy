# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing datastructures.

Subpackages
^^^^^^^^^^^

.. currentmodule:: welltestpy.data

.. autosummary::
    data_io
    varlib
    testslib
    campaignlib

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

Variable classes
~~~~~~~~~~~~~~~~

.. currentmodule:: welltestpy.data.varlib

.. autosummary::
    Variable
    TimeVar
    HeadVar
    TemporalVar
    CoordinatesVar
    Observation
    StdyObs
    DrawdownObs
    StdyHeadObs
    Well

Routines
^^^^^^^^

Loading routines
~~~~~~~~~~~~~~~~

.. currentmodule:: welltestpy.data.data_io

Campaign related loading routines

.. autosummary::
    load_campaign
    load_fieldsite

Field test related loading routines

.. autosummary::
    load_test

Variable related loading routines

.. autosummary::
    load_var
    load_obs
    load_well
"""
from . import varlib, testslib, campaignlib, data_io

from .varlib import (
    Variable,
    TimeVar,
    HeadVar,
    TemporalVar,
    CoordinatesVar,
    Observation,
    StdyObs,
    DrawdownObs,
    StdyHeadObs,
    Well,
)
from .testslib import PumpingTest
from .campaignlib import (
    FieldSite,
    Campaign,
)
from .data_io import (
    load_var,
    load_obs,
    load_well,
    load_campaign,
    load_fieldsite,
    load_test,
)

__all__ = [
    "Variable",
    "TimeVar",
    "HeadVar",
    "TemporalVar",
    "CoordinatesVar",
    "Observation",
    "StdyObs",
    "DrawdownObs",
    "StdyHeadObs",
    "Well",
]
__all__ += [
    "PumpingTest",
]
__all__ += [
    "FieldSite",
    "Campaign",
]
__all__ += [
    "load_var",
    "load_obs",
    "load_well",
    "load_test",
    "load_fieldsite",
    "load_campaign",
]
__all__ += [
    "varlib",
    "testslib",
    "campaignlib",
    "data_io",
]
