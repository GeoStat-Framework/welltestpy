"""
welltestpy subpackage providing datastructures.

.. currentmodule:: welltestpy.data

Included classes
----------------
The following classes are provided

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
    PumpingTest
    FieldSite
    Campaign

Included functions
------------------
The following functions are provided

.. autosummary::
    load_var
    load_obs
    load_well
    load_test
    load_fieldsite
    load_campaign

Subpackages
-----------
The following subpackages are provided

.. autosummary::
    varlib
    testslib
    campaignlib
"""
from __future__ import absolute_import

from welltestpy.data import (
    varlib,
    testslib,
    campaignlib,
)

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
from welltestpy.data.testslib import (
    PumpingTest,
    load_test,
)
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
