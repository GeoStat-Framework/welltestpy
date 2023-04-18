"""
welltestpy subpackage providing datastructures.


Campaign classes
~~~~~~~~~~~~~~~~

The following classes can be used to handle field campaigns.

.. autosummary::
   :toctree:

    Campaign
    FieldSite

Field Test classes
~~~~~~~~~~~~~~~~~~

The following classes can be used to handle field test within a campaign.

.. autosummary::
   :toctree:

   PumpingTest
   Test

Variable classes
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree:

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

Campaign related loading routines

.. autosummary::
   :toctree:

   load_campaign
   load_fieldsite

Field test related loading routines

.. autosummary::
   :toctree:

   load_test

Variable related loading routines

.. autosummary::
   :toctree:

   load_var
   load_obs
   load_well
"""
from . import campaignlib, data_io, testslib, varlib
from .campaignlib import Campaign, FieldSite
from .data_io import (
    load_campaign,
    load_fieldsite,
    load_obs,
    load_test,
    load_var,
    load_well,
)
from .testslib import PumpingTest, Test
from .varlib import (
    CoordinatesVar,
    DrawdownObs,
    HeadVar,
    Observation,
    StdyHeadObs,
    StdyObs,
    TemporalVar,
    TimeVar,
    Variable,
    Well,
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
    "Test",
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
