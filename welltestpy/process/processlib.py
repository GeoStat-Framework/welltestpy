# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing functions to pre process data.

.. currentmodule:: welltestpy.process.processlib

The following classes are provided

.. autosummary::
   normpumptest
   combinepumptest
   filterdrawdown
"""
from copy import deepcopy as dcopy
import numpy as np
from scipy import signal

from ..data import testslib

__all__ = ["normpumptest", "combinepumptest", "filterdrawdown"]


def normpumptest(pumptest, pumpingrate=-1.0, factor=1.0):
    """Normalize the pumping rate of a pumping test.

    Parameters
    ----------
    pumpingrate : :class:`float`, optional
        Pumping rate. Default: ``-1.0``
    factor : :class:`float`, optional
        Scaling factor that can be used for unit conversion. Default: ``1.0``
    """
    if not isinstance(pumptest, testslib.PumpingTest):
        raise ValueError(str(pumptest) + " is no pumping test")

    if not pumptest.constant_rate:
        raise ValueError(str(pumptest) + " is no constant rate pumping test")

    oldprate = dcopy(pumptest.rate)
    pumptest.pumpingrate = pumpingrate

    for obs in pumptest.observations:
        pumptest.observations[obs].observation *= (
            factor * pumptest.rate / oldprate
        )


def combinepumptest(
    campaign,
    test1,
    test2,
    pumpingrate=None,
    finalname=None,
    factor1=1.0,
    factor2=1.0,
    infooftest1=True,
    replace=True,
):
    """Combine two pumping tests to one.

    They need to have the same pumping well.

    Parameters
    ----------
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used.
    test1 : :class:`str`
        Name of test 1.
    test2 : :class:`str`
        Name of test 2.
    pumpingrate : :class:`float`, optional
        Pumping rate. Default: ``-1.0``
    finalname : :class:`str`, optional
        Name of the final test. If `replace` is `True` and `finalname` is
        `None`, it will get the name of test 1. Else it will get a combined
        name of test 1 and test 2.
        Default: ``None``
    factor1 : :class:`float`, optional
        Scaling factor for test 1 that can be used for unit conversion.
        Default: ``1.0``
    factor2 : :class:`float`, optional
        Scaling factor for test 2 that can be used for unit conversion.
        Default: ``1.0``
    infooftest1 : :class:`bool`, optional
        State if the final test should take the information from test 1.
        Default: ``True``
    replace : :class:`bool`, optional
        State if the original tests should be erased.
        Default: ``True``
    """
    if test1 not in campaign.tests:
        raise ValueError(
            "combinepumptest: "
            + str(test1)
            + " not a test in "
            + "campaign "
            + str(campaign.name)
        )
    if test2 not in campaign.tests:
        raise ValueError(
            "combinepumptest: "
            + str(test2)
            + " not a test in "
            + "campaign "
            + str(campaign.name)
        )

    if finalname is None:
        if replace:
            finalname = test1
        else:
            finalname = test1 + "+" + test2

    if campaign.tests[test1].testtype != "PumpingTest":
        raise ValueError(
            "combinepumptest:" + str(test1) + " is no pumpingtest"
        )
    if campaign.tests[test2].testtype != "PumpingTest":
        raise ValueError(
            "combinepumptest:" + str(test2) + " is no pumpingtest"
        )

    if campaign.tests[test1].pumpingwell != campaign.tests[test2].pumpingwell:
        raise ValueError(
            "combinepumptest: The Pumpingtests do not have the "
            + "same pumping-well"
        )

    pwell = campaign.tests[test1].pumpingwell

    wellset1 = set(campaign.tests[test1].wells)
    wellset2 = set(campaign.tests[test2].wells)

    commonwells = wellset1 & wellset2

    if commonwells != {pwell} and commonwells != set():
        raise ValueError(
            "combinepumptest: The Pumpingtests shouldn't have "
            + "common observation-wells"
        )

    temptest1 = dcopy(campaign.tests[test1])
    temptest2 = dcopy(campaign.tests[test2])

    if pumpingrate is None:
        if infooftest1:
            pumpingrate = temptest1.rate
        else:
            pumpingrate = temptest2.rate

    normpumptest(temptest1, pumpingrate, factor1)
    normpumptest(temptest2, pumpingrate, factor2)

    prate = temptest1.rate

    if infooftest1:
        if pwell in temptest1.observations and pwell in temptest2.observations:
            temptest2.del_observations(pwell)
        aquiferdepth = temptest1.depth
        aquiferradius = temptest1.radius
        description = temptest1.description
        timeframe = temptest1.timeframe
    else:
        if pwell in temptest1.observations and pwell in temptest2.observations:
            temptest1.del_observations(pwell)
        aquiferdepth = temptest2.depth
        aquiferradius = temptest2.radius
        description = temptest2.description
        timeframe = temptest2.timeframe

    observations = dcopy(temptest1.observations)
    observations.update(temptest2.observations)

    if infooftest1:
        aquiferdepth = temptest1.depth
        aquiferradius = temptest1.radius
        description = temptest1.description
        timeframe = temptest1.timeframe
    else:
        aquiferdepth = temptest2.depth
        aquiferradius = temptest2.radius
        description = temptest2.description
        timeframe = temptest2.timeframe

    finalpt = testslib.PumpingTest(
        finalname,
        pwell,
        prate,
        observations,
        aquiferdepth,
        aquiferradius,
        description,
        timeframe,
    )

    campaign.addtests(finalpt)

    if replace:
        campaign.deltests([test1, test2])


def filterdrawdown(observation, tout=None, dxscale=2):
    """Smooth the drawdown data of an observation well.

    Parameters
    ----------
    observation : :class:`welltestpy.data.Observation`
        The observation to be smoothed.
    tout : :class:`numpy.ndarray`, optional
        Time points to evaluate the smoothed observation at. If ``None``,
        the original time points of the observation are taken.
        Default: ``None``
    dxscale : :class:`int`, optional
        Scale of time-steps used for smoothing.
        Default: ``2``
    """
    head, time = observation()
    head = np.array(head, dtype=float).reshape(-1)
    time = np.array(time, dtype=float).reshape(-1)

    if tout is None:
        tout = dcopy(time)
    tout = np.array(tout, dtype=float).reshape(-1)

    if len(time) == 1:
        return observation(time=tout, observation=np.full_like(tout, head[0]))

    # make the data equal-spaced to use filter with
    # a fraction of the minimal timestep
    dxv = dxscale * int((time[-1] - time[0]) / max(np.diff(time).min(), 1.0))
    tequal = np.linspace(time[0], time[-1], dxv)
    hequal = np.interp(tequal, time, head)
    # size = h.max() - h.min()

    try:
        para1, para2 = signal.butter(1, 0.025)  # size/10.)
        hfilt = signal.filtfilt(para1, para2, hequal, padlen=150)
        hout = np.interp(tout, tequal, hfilt)
    except ValueError:  # in this case there are to few data points
        hout = np.interp(tout, time, head)

    return observation(time=tout, observation=hout)
