# -*- coding: utf-8 -*-
"""Spotpy classes for the estimation library from welltestpy

This module contanins spotpy classes to define estimation procedures
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import functools as ft

import numpy as np
import spotpy

import anaflow as ana


# functions for fitting

def ext_theis2D(rad, time, Qw=-.0001, rwell=0.0, rinf=np.inf):
    def function(mu, sig2, corr, lnS):
        return ana.ext_theis2D(rad=rad, time=time,
                               TG=np.exp(mu), sig2=sig2,
                               corr=corr, S=np.exp(lnS),
                               Qw=Qw, rwell=rwell, rinf=rinf,
                               parts=25, stehfestn=12)
    return function


def theis(rad, time, Qw=-.0001, rwell=0.0, rinf=np.inf):
    def function(mu, lnS):
        return ana.theis(rad=rad, time=time,
                         T=np.exp(mu), S=np.exp(lnS),
                         Qw=Qw, rwell=rwell, rinf=rinf, stehfestn=12)
    return function


# spotpy classes

class Stat2Dsetup(object):
    def __init__(self, rad, time, rtdata, Qw, bestvalues=None,
                 mu=None, sig2=None, corr=None, lnS=None,
                 murange=None, sig2range=None, corrrange=None, lnSrange=None):
        self.params = []
        self.kwargs = {}
        self.ranges = {}
        self.simkw = []
        self.simkwargs = {}
        self.mu = mu
        self.sig2 = sig2
        self.corr = corr
        self.lnS = lnS

        self.rad = rad
        self.time = time
        self.data = dcopy(rtdata)
        self.Qw = Qw

        # weights by time and radius
#        for ti, te in enumerate(self.time):
#            self.data[ti] /= np.log(te)
#        for ti, te in enumerate(self.time):
#            self.data[ti] /= np.log10(te)
#        for ri, re in enumerate(self.rad):
#            self.data[:,ri] *= 10. + re
#        for ri, re in enumerate(self.rad):
#            self.data[:,ri] /= 10. + re

        if murange is None:
#            self.ranges["mu"] = (-16., -1., 1., -9., -16., -1.)
            self.ranges["mu"] = (-10., -4., 1., -6., -10., -4.)
        else:
            self.ranges["mu"] = murange
        if sig2range is None:
#            self.ranges["sig2"] = (.01, 6., .5, 2.55, .01, 6.)
            self.ranges["sig2"] = (.01, 4., .5, 2, .01, 4.)
        else:
            self.ranges["sig2"] = sig2range
        if corrrange is None:
#            self.ranges["corr"] = (.5, 40., 2., 18., .5, 40.)
            self.ranges["corr"] = (.5, 15., 2., 4., .5, 15.)
        else:
            self.ranges["corr"] = corrrange
        if lnSrange is None:
#            self.ranges["lnS"] = (-16., -1., 1., -9., -16., -1.)
            self.ranges["lnS"] = (-10., -4., 1., -6., -10., -4.)
        else:
            self.ranges["lnS"] = lnSrange

        if self.mu is None:
            self.params.append(spotpy.parameter.Uniform('mu',
                                                        *self.ranges["mu"]))
            self.simkw.append("mu")
            self.simkwargs["mu"] = 0.
        else:
            self.kwargs["mu"] = self.mu

        if self.sig2 is None:
            self.params.append(spotpy.parameter.Uniform('sig2',
                                                        *self.ranges["sig2"]))
            self.simkw.append("sig2")
            self.simkwargs["sig2"] = 0.
        else:
            self.kwargs["sig2"] = self.sig2

        if self.corr is None:
            self.params.append(spotpy.parameter.Uniform('corr',
                                                        *self.ranges["corr"]))
            self.simkw.append("corr")
            self.simkwargs["corr"] = 0.
        else:
            self.kwargs["corr"] = self.corr

        if self.lnS is None:
            self.params.append(spotpy.parameter.Uniform('lnS',
                                                        *self.ranges["lnS"]))
            self.simkw.append("lnS")
            self.simkwargs["lnS"] = 0.
        else:
            self.kwargs["lnS"] = self.lnS

        if bestvalues is None:
            self.bestvalues = {}
            if self.mu is None:
                self.bestvalues["mu"] = self.ranges["mu"][3]
            if self.sig2 is None:
                self.bestvalues["sig2"] = self.ranges["sig2"][3]
            if self.corr is None:
                self.bestvalues["corr"] = self.ranges["corr"][3]
            if self.lnS is None:
                self.bestvalues["lnS"] = self.ranges["lnS"][3]
        else:
            self.bestvalues = bestvalues

        self.sim_raw = ext_theis2D(self.rad, self.time, self.Qw)

        self.sim = ft.partial(self.sim_raw, **self.kwargs)

    def parameters(self):
        ret = spotpy.parameter.generate(self.params)
        return ret

    def simulation(self, vector):
        x = np.array(vector)
        for i, m in enumerate(self.simkw):
            if not np.isfinite(x[i]):
                # if FAST-alg is producing nan-values, set the best value
                self.simkwargs[m] = self.bestvalues[m]
            else:
                self.simkwargs[m] = x[i]
        ret = self.sim(**self.simkwargs)

        # weights by time and radius
#        for ti, te in enumerate(self.time):
#            ret[ti] /= np.log(te)
#        for ti, te in enumerate(self.time):
#            ret[ti] /= np.log10(te)
#        for ri, re in enumerate(self.rad):
#            ret[:,ri] *= 10. + re
#        for ri, re in enumerate(self.rad):
#            ret[:,ri] /= 10. + re
        print("sim: ", np.min(ret), np.max(ret))
        print(self.simkwargs)
        return ret.reshape(-1)

    def evaluation(self):
        ret = np.squeeze(np.array(self.data).reshape(-1))
        print("eva: ", np.min(ret), np.max(ret))
        return ret

    def objectivefunction(self, simulation=simulation,
                          evaluation=evaluation):
        ret = -spotpy.objectivefunctions.rmse(evaluation=evaluation,
                                              simulation=simulation)
        return ret


class Theissetup(object):
    def __init__(self, rad, time, rtdata, Qw, bestvalues=None,
                 mu=None, lnS=None,
                 murange=None, lnSrange=None):
        self.params = []
        self.kwargs = {}
        self.ranges = {}
        self.simkw = []
        self.simkwargs = {}
        self.mu = mu
        self.lnS = lnS

        self.rad = rad
        self.time = time
        self.data = dcopy(rtdata)
        self.Qw = Qw

        # weights by time and radius
#        for ti, te in enumerate(self.time):
#            self.data[ti] /= np.log(te)
#        for ti, te in enumerate(self.time):
#            self.data[ti] /= np.log10(te)
#        for ri, re in enumerate(self.rad):
#            self.data[:,ri] *= 10. + re
#        for ri, re in enumerate(self.rad):
#            self.data[:,ri] /= 10. + re

        if murange is None:
            self.ranges["mu"] = (-16., -1., 1., -9., -16., -1.)
        else:
            self.ranges["mu"] = murange
        if lnSrange is None:
            self.ranges["lnS"] = (-16., -1., 1., -9., -16., -1.)
        else:
            self.ranges["lnS"] = lnSrange

        if self.mu is None:
            self.params.append(spotpy.parameter.Uniform('mu',
                                                        *self.ranges["mu"]))
            self.simkw.append("mu")
            self.simkwargs["mu"] = 0.
        else:
            self.kwargs["mu"] = self.mu

        if self.lnS is None:
            self.params.append(spotpy.parameter.Uniform('lnS',
                                                        *self.ranges["lnS"]))
            self.simkw.append("lnS")
            self.simkwargs["lnS"] = 0.
        else:
            self.kwargs["lnS"] = self.lnS

        if bestvalues is None:
            self.bestvalues = {}
            if self.mu is None:
                self.bestvalues["mu"] = self.ranges["mu"][3]
            if self.lnS is None:
                self.bestvalues["lnS"] = self.ranges["lnS"][3]
        else:
            self.bestvalues = bestvalues

        self.sim_raw = theis(self.rad, self.time, self.Qw)

        self.sim = ft.partial(self.sim_raw, **self.kwargs)

    def parameters(self):
        ret = spotpy.parameter.generate(self.params)
        return ret

    def simulation(self, vector):
        x = np.array(vector)
        for i, m in enumerate(self.simkw):
            if np.isnan(x[i]):
                # if FAST-alg is producing nan-values, set the best value
                self.simkwargs[m] = self.bestvalues[m]
            else:
                self.simkwargs[m] = x[i]
        ret = self.sim(**self.simkwargs)

        # weights by time and radius
#        for ti, te in enumerate(self.time):
#            ret[ti] /= np.log(te)
#        for ti, te in enumerate(self.time):
#            ret[ti] /= np.log10(te)
#        for ri, re in enumerate(self.rad):
#            ret[:,ri] *= 10. + re
#        for ri, re in enumerate(self.rad):
#            ret[:,ri] /= 10. + re

        return ret.reshape(-1)

    def evaluation(self):
        ret = np.squeeze(np.array(self.data).reshape(-1))
        return ret

    def objectivefunction(self, simulation=simulation,
                          evaluation=evaluation):
        ret = -spotpy.objectivefunctions.rmse(evaluation=evaluation,
                                              simulation=simulation)
        return ret
