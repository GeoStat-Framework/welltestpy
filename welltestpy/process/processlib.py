#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:43:13 2016

@author: Sebastian Mueller
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import numpy as np
from scipy import signal

from welltestpy.data.testslib import (PumpingTest)


def normpumptest(pumptest, pumpingrate=0.1, factor=1.0):
    if not isinstance(pumptest, PumpingTest):
        raise ValueError(str(pumptest)+" is no pumpingtest")

    oldprate = dcopy(pumptest.pumpingrate)
    pumptest.pumpingrate = pumpingrate

    for obs in pumptest.observations:
        pumptest.observations[obs].observation *= factor*pumpingrate/oldprate


def combinepumptest(campaign, test1, test2,
                    pumpingrate=None, finalname=None,
                    factor1=1.0, factor2=1.0,
                    infooftest1=True, replace=True):

    if test1 not in campaign.tests:
        raise ValueError("combinepumptest: "+str(test1)+" not a test in " +
                         "campaign "+str(campaign.name))
    if test2 not in campaign.tests:
        raise ValueError("combinepumptest: "+str(test2)+" not a test in " +
                         "campaign "+str(campaign.name))

    if finalname is None:
        if replace:
            finalname = test1
        else:
            finalname = test1+"+"+test2

    if campaign.tests[test1].testtype != "PumpingTest":
        raise ValueError("combinepumptest:"+str(test1)+" is no pumpingtest")
    if campaign.tests[test2].testtype != "PumpingTest":
        raise ValueError("combinepumptest:"+str(test2)+" is no pumpingtest")

    if campaign.tests[test1].pumpingwell != campaign.tests[test2].pumpingwell:
        raise ValueError("combinepumptest: The Pumpingtests do not have the " +
                         "same pumping-well")

    pwell = campaign.tests[test1].pumpingwell

    wellset1 = set(campaign.tests[test1].wells)
    wellset2 = set(campaign.tests[test2].wells)

    commonwells = wellset1 & wellset2

    if commonwells != {pwell} and commonwells != set():
        raise ValueError("combinepumptest: The Pumpingtests shouldn't have " +
                         "common observation-wells")

    temptest1 = dcopy(campaign.tests[test1])
    temptest2 = dcopy(campaign.tests[test2])

    if pumpingrate is None:
        if infooftest1:
            pumpingrate = dcopy(temptest1._pumpingrate)
        else:
            pumpingrate = dcopy(temptest2._pumpingrate)

    normpumptest(temptest1, pumpingrate, factor1)
    normpumptest(temptest2, pumpingrate, factor2)

    prate = dcopy(temptest1._pumpingrate)

    if infooftest1:
        if pwell in temptest1.observations and pwell in temptest2.observations:
            temptest2.delobservations(pwell)
        aquiferdepth = temptest1.aquiferdepth
        aquiferradius = temptest1.aquiferradius
        description = temptest1.description
        timeframe = temptest1.timeframe
    else:
        if pwell in temptest1.observations and pwell in temptest2.observations:
            temptest1.delobservations(pwell)
        aquiferdepth = temptest2.aquiferdepth
        aquiferradius = temptest2.aquiferradius
        description = temptest2.description
        timeframe = temptest2.timeframe

    observations = dcopy(temptest1.observations)
    observations.update(temptest2.observations)

    if infooftest1:
        aquiferdepth = temptest1.aquiferdepth
        aquiferradius = temptest1.aquiferradius
        description = temptest1.description
        timeframe = temptest1.timeframe
    else:
        aquiferdepth = temptest2.aquiferdepth
        aquiferradius = temptest2.aquiferradius
        description = temptest2.description
        timeframe = temptest2.timeframe

    finalpt = PumpingTest(finalname, pwell, prate, observations,
                          aquiferdepth, aquiferradius,
                          description, timeframe)

    campaign.addtests(finalpt)

    if replace:
        campaign.deltests([test1, test2])


def filterdrawdown(observation, tout=None, dxscale=2):
    t, h = observation()
    t = t.reshape(-1)
    h = h.reshape(-1)

    if tout is None:
        tout = dcopy(t)

    # make the data equal-spaced to use filter with
    # a fraction of the minimal timestep
    dxv = dxscale*int((t[-1]-t[0])/max(np.diff(t).min(), 1.))

    tequal = np.linspace(t[0], t[-1], dxv)
    hequal = np.interp(tequal, t, h)

    # size = h.max() - h.min()

    b, a = signal.butter(1, .025)  # size/10.)
    hfilt = signal.filtfilt(b, a, hequal, padlen=150)

    hout = np.interp(tout, tequal, hfilt)

    observation(time=tout, observation=hout)
