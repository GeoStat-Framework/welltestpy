# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing classes for Spotpy classes for the
estimation library

.. currentmodule:: welltestpy.estimate.spotpy_classes

The following functions and classes are provided

.. autosummary::
   ext_theis2D
   theis
   Stat2Dsetup
   Theissetup
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import functools as ft

import numpy as np
import spotpy

import anaflow as ana

__all__ = ["ext_theis2D", "theis", "Stat2Dsetup", "Theissetup"]


# functions for fitting


def ranges(val_min, val_max, guess=None, stepsize=None):
    """Create sampling ranges fram value bounds"""
    if guess is None:
        guess = (val_max + val_min) / 2.0
    if stepsize is None:
        stepsize = (val_max - val_min) / 10.0
    return (val_min, val_max, stepsize, guess, val_min, val_max)


def ext_theis2D(rad, time, Qw=-0.0001, rwell=0.0, rinf=np.inf):
    """
    The extended Theis solution in 2D

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a 2D statistical framework, where the transmissivity distribution is
    following a log-normal distribution with a gaussian correlation function.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    Qw : :class:`float`
        Pumpingrate at the well
    rwell : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``

    Returns
    -------
    ext_theis2D : :any:`callable`
        callable function taking ``mu``, ``sig2``, ``corr`` and ``lnS``.
    """

    def function(mu, sig2, corr, lnS):
        """
        The extended Theis solution in 2D

        The extended Theis solution for transient flow under
        a pumping condition in a confined aquifer.
        The type curve is describing the effective drawdown
        in a 2D statistical framework, where the transmissivity distribution is
        following a log-normal distribution with a gaussian correlation
        function.

        Parameters
        ----------
        mu : :class:`float`
            Geometric-mean transmissivity (log val)
        sig2 : :class:`float`
            log-normal-variance of the transmissivity-distribution
        corr : :class:`float`
            corralation-length of transmissivity-distribution
        lnS : :class:`float`
            Given log-storativity of the aquifer

        Returns
        -------
        ext_theis2D : :class:`numpy.ndarray`
            Array with all heads at the given radii and time-points.
        """
        return ana.ext_theis2D(
            rad=rad,
            time=time,
            TG=np.exp(mu),
            sig2=sig2,
            corr=corr,
            S=np.exp(lnS),
            Qw=Qw,
            rwell=rwell,
            rinf=rinf,
            parts=25,
            stehfestn=12,
        )

    return function


def theis(rad, time, Qw=-0.0001, rwell=0.0, rinf=np.inf):
    """
    The Theis solution

    The Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in [Theis35]_.

    .. [Theis35] Theis, C.,
       ''The relation between the lowering of the piezometric surface and the
       rate and duration of discharge of a well using groundwater storage'',
       Trans. Am. Geophys. Union, 16, 519–524, 1935

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    Qw : :class:`float`
        Pumpingrate at the well
    rwell : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``

    Returns
    -------
    theis : :any:`callable`
        callable function taking ``mu`` and ``lnS``.
    """

    def function(mu, lnS):
        """
        The Theis solution

        Parameters
        ----------
        mu : :class:`float`
            Given log-transmissivity of the aquifer
        lnS : :class:`float`
            Given log-storativity of the aquifer

        Returns
        -------
        theis : :class:`numpy.ndarray`
            Array with all heads at the given radii and time-points.
        """
        return ana.theis(
            rad=rad,
            time=time,
            T=np.exp(mu),
            S=np.exp(lnS),
            Qw=Qw,
            rwell=rwell,
            rinf=rinf,
            stehfestn=12,
        )

    return function


# spotpy classes


class Stat2Dsetup(object):
    """Spotpy class for an estimation of stochastic subsurface parameters.

    This class uses the extended Theis solution in 2D to estimate parameters
    of heterogeneity of an aquifer from pumping test data.
    """

    def __init__(
        self,
        rad,
        time,
        rtdata,
        Qw,
        bestvalues=None,
        mu=None,
        sig2=None,
        corr=None,
        lnS=None,
        murange=(-16.0, -2.0),
        sig2range=(0.1, 10.0),
        corrrange=(1.0, 50.0),
        lnSrange=(-13.0, -1.0),
        rwell=0.0,
        rinf=np.inf,
    ):
        """Spotpy class initialisation.

        Parameters
        ----------
        rad : :class:`numpy.ndarray`
            Array of the radii apearing in the wellsetup
        time : :class:`numpy.ndarray`
            Array of time-points of the pumping test data
        rtdata : :class:`numpy.ndarray`
            Observed head data as array.
        Qw : :class:`float`
            Pumpingrate at the well
        bestvalues : :class:`dict`, optional
            Guessed best values by name ``mu``, ``sig2``, ``corr`` and ``lnS``.
            Default: ``None``
        mu : :class:`float`, optional
            Here you can fix the value for mean log-transmissivity ``mu``.
            Default: ``None``
        sig2 : :class:`float`, optional
            Here you can fix the value for variance of
            log-transmissivity ``sig2``.
            Default: ``None``
        corr : :class:`float`, optional
            Here you can fix the value for correlation length of
            log-transmissivity ``sig2``.
            Default: ``None``
        lnS : :class:`float`, optional
            Here you can fix the value for log-storativity ``lnS``.
            Default: ``None``
        murange : :class:`tuple`, optional
            Here you can specifiy the range of ``mu``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        sig2range : :class:`tuple`, optional
            Here you can specifiy the range of ``sig2``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        corrrange : :class:`tuple`, optional
            Here you can specifiy the range of ``corr``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        lnSrange : :class:`tuple`, optional
            Here you can specifiy the range of ``lnS``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        rwell : :class:`float`, optional
            Inner radius of the pumping-well. Default: ``0.0``
        rinf : :class:`float`, optional
            Radius of the outer boundary of the aquifer. Default: ``np.inf``
        """
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

        #        if murange is None:
        #            self.ranges["mu"] = (-16.0, -1.0, 1.0, -9.0, -16.0, -1.0)
        #        else:
        #            self.ranges["mu"] = murange
        #        if sig2range is None:
        #            self.ranges["sig2"] = (0.01, 6.0, 0.5, 2.55, 0.01, 6.0)
        #        else:
        #            self.ranges["sig2"] = sig2range
        #        if corrrange is None:
        #            self.ranges["corr"] = (0.5, 40.0, 2.0, 18.0, 0.5, 40.0)
        #        else:
        #            self.ranges["corr"] = corrrange
        #        if lnSrange is None:
        #            self.ranges["lnS"] = (-16.0, -1.0, 1.0, -9.0, -16.0, -1.0)
        #        else:
        #            self.ranges["lnS"] = lnSrange

        self.ranges["mu"] = ranges(*murange)
        self.ranges["sig2"] = ranges(*sig2range)
        self.ranges["corr"] = ranges(*corrrange)
        self.ranges["lnS"] = ranges(*lnSrange)

        if self.mu is None:
            self.params.append(
                spotpy.parameter.Uniform("mu", *self.ranges["mu"])
            )
            self.simkw.append("mu")
            self.simkwargs["mu"] = 0.0
        else:
            self.kwargs["mu"] = self.mu

        if self.sig2 is None:
            self.params.append(
                spotpy.parameter.Uniform("sig2", *self.ranges["sig2"])
            )
            self.simkw.append("sig2")
            self.simkwargs["sig2"] = 0.0
        else:
            self.kwargs["sig2"] = self.sig2

        if self.corr is None:
            self.params.append(
                spotpy.parameter.Uniform("corr", *self.ranges["corr"])
            )
            self.simkw.append("corr")
            self.simkwargs["corr"] = 0.0
        else:
            self.kwargs["corr"] = self.corr

        if self.lnS is None:
            self.params.append(
                spotpy.parameter.Uniform("lnS", *self.ranges["lnS"])
            )
            self.simkw.append("lnS")
            self.simkwargs["lnS"] = 0.0
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

        self.sim_raw = ext_theis2D(
            self.rad, self.time, self.Qw, rwell=rwell, rinf=rinf
        )

        self.sim = ft.partial(self.sim_raw, **self.kwargs)

    def parameters(self):
        """Generate a set of parameters"""
        ret = spotpy.parameter.generate(self.params)
        return ret

    def simulation(self, vector):
        """Run a simulation with the given parameters"""
        x = np.array(vector)
        for i, kw_i in enumerate(self.simkw):
            if not np.isfinite(x[i]):
                # if FAST-alg is producing nan-values, set the best value
                self.simkwargs[kw_i] = self.bestvalues[kw_i]
            else:
                self.simkwargs[kw_i] = x[i]
        ret = self.sim(**self.simkwargs)
        return ret.reshape(-1)

    def evaluation(self):
        """Accesss the drawdown data"""
        ret = np.squeeze(np.array(self.data).reshape(-1))
        return ret

    def objectivefunction(self, simulation=simulation, evaluation=evaluation):
        """Calculate the root mean square error between observation and
        simulation"""
        ret = -spotpy.objectivefunctions.rmse(
            evaluation=evaluation, simulation=simulation
        )
        return ret


class Theissetup(object):
    """Spotpy class for an estimation of subsurface parameters.

    This class uses the Theis solution to estimate parameters
    of an homogeneous aquifer from pumping test data.
    """

    def __init__(
        self,
        rad,
        time,
        rtdata,
        Qw,
        bestvalues=None,
        mu=None,
        lnS=None,
        murange=None,
        lnSrange=None,
        rwell=0.0,
        rinf=np.inf,
    ):
        """Spotpy class initialisation.

        Parameters
        ----------
        rad : :class:`numpy.ndarray`
            Array of the radii apearing in the wellsetup
        time : :class:`numpy.ndarray`
            Array of time-points of the pumping test data
        rtdata : :class:`numpy.ndarray`
            Observed head data as array.
        Qw : :class:`float`
            Pumpingrate at the well
        bestvalues : :class:`dict`, optional
            Guessed best values by name ``mu``, ``sig2``, ``corr`` and ``lnS``.
            Default: ``None``
        mu : :class:`float`, optional
            Here you can fix the value for log-transmissivity ``mu``.
            Default: ``None``
        lnS : :class:`float`, optional
            Here you can fix the value for log-storativity ``lnS``.
            Default: ``None``
        murange : :class:`tuple`, optional
            Here you can specifiy the range of ``mu``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        lnSrange : :class:`tuple`, optional
            Here you can specifiy the range of ``lnS``. It has the following
            structure:

                ``(min, max, step, start-value, min-value, max-value)``

            Default: ``None``
        rwell : :class:`float`, optional
            Inner radius of the pumping-well. Default: ``0.0``
        rinf : :class:`float`, optional
            Radius of the outer boundary of the aquifer. Default: ``np.inf``
        """
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

        if murange is None:
            self.ranges["mu"] = (-16.0, -1.0, 1.0, -9.0, -16.0, -1.0)
        else:
            self.ranges["mu"] = murange
        if lnSrange is None:
            self.ranges["lnS"] = (-16.0, -1.0, 1.0, -9.0, -16.0, -1.0)
        else:
            self.ranges["lnS"] = lnSrange

        if self.mu is None:
            self.params.append(
                spotpy.parameter.Uniform("mu", *self.ranges["mu"])
            )
            self.simkw.append("mu")
            self.simkwargs["mu"] = 0.0
        else:
            self.kwargs["mu"] = self.mu

        if self.lnS is None:
            self.params.append(
                spotpy.parameter.Uniform("lnS", *self.ranges["lnS"])
            )
            self.simkw.append("lnS")
            self.simkwargs["lnS"] = 0.0
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

        self.sim_raw = theis(
            self.rad, self.time, self.Qw, rwell=rwell, rinf=rinf
        )

        self.sim = ft.partial(self.sim_raw, **self.kwargs)

    def parameters(self):
        """Generate a set of parameters"""
        ret = spotpy.parameter.generate(self.params)
        return ret

    def simulation(self, vector):
        """Run a simulation with the given parameters"""
        x = np.array(vector)
        for i, kw_i in enumerate(self.simkw):
            if np.isnan(x[i]):
                # if FAST-alg is producing nan-values, set the best value
                self.simkwargs[kw_i] = self.bestvalues[kw_i]
            else:
                self.simkwargs[kw_i] = x[i]
        ret = self.sim(**self.simkwargs)
        return ret.reshape(-1)

    def evaluation(self):
        """Accesss the drawdown data"""
        ret = np.squeeze(np.array(self.data).reshape(-1))
        return ret

    def objectivefunction(self, simulation=simulation, evaluation=evaluation):
        """Calculate the root mean square error between observation and
        simulation"""
        ret = -spotpy.objectivefunctions.rmse(
            evaluation=evaluation, simulation=simulation
        )
        return ret
