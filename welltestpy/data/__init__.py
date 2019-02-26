# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing datastructures.

Subpackages
^^^^^^^^^^^

.. currentmodule:: welltestpy.data

.. autosummary::
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

.. currentmodule:: welltestpy.data.campaignlib

Campaign related loading routines

.. autosummary::
    load_campaign
    load_fieldsite

.. currentmodule:: welltestpy.data.testslib

Field test related loading routines

.. autosummary::
    load_test

.. currentmodule:: welltestpy.data.varlib

Variable related loading routines

.. autosummary::
    load_var
    load_obs
    load_well
"""
from __future__ import absolute_import

from welltestpy.data import varlib, testslib, campaignlib

from welltestpy.data.varlib import (
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
    load_var,
    load_obs,
    load_well,
)
from welltestpy.data.testslib import PumpingTest, load_test
from welltestpy.data.campaignlib import (
    FieldSite,
    Campaign,
    load_fieldsite,
    load_campaign,
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
    "PumpingTest",
    "FieldSite",
    "Campaign",
    "load_var",
    "load_obs",
    "load_well",
    "load_test",
    "load_fieldsite",
    "load_campaign",
    "varlib",
    "testslib",
    "campaignlib",
]
