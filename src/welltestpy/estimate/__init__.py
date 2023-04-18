"""
welltestpy subpackage providing routines to estimate pump test parameters.

.. currentmodule:: welltestpy.estimate

Estimators
^^^^^^^^^^

The following estimators are provided

.. autosummary::
   :toctree:

   ExtTheis3D
   ExtTheis2D
   Neuman2004
   Theis
   ExtThiem3D
   ExtThiem2D
   Neuman2004Steady
   Thiem

Base Classes
^^^^^^^^^^^^

Transient
~~~~~~~~~

All transient estimators are derived from the following class

.. autosummary::
   :toctree:

   TransientPumping

Steady Pumping
~~~~~~~~~~~~~~

All steady estimators are derived from the following class

.. autosummary::
   :toctree:

   SteadyPumping

Helper
^^^^^^

.. autosummary::
   :toctree:

   fast_rep
"""
from . import estimators, spotpylib, steady_lib, transient_lib
from .estimators import (
    ExtTheis2D,
    ExtTheis3D,
    ExtThiem2D,
    ExtThiem3D,
    Neuman2004,
    Neuman2004Steady,
    Theis,
    Thiem,
)
from .spotpylib import fast_rep
from .steady_lib import SteadyPumping
from .transient_lib import TransientPumping

__all__ = ["estimators", "spotpylib", "steady_lib", "transient_lib"]
__all__ += [
    "ExtTheis3D",
    "ExtTheis2D",
    "Neuman2004",
    "Theis",
    "ExtThiem3D",
    "ExtThiem2D",
    "Neuman2004Steady",
    "Thiem",
]
__all__ += ["TransientPumping", "SteadyPumping", "fast_rep"]
