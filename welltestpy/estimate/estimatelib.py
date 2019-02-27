# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing classes for parameter estimation.

.. currentmodule:: welltestpy.estimate.estimatelib

The following classes are provided

.. autosummary::
   Stat2Dest
   Theisest
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import time as timemodule

import numpy as np
import spotpy

from welltestpy.process.processlib import normpumptest, filterdrawdown
from welltestpy.estimate.spotpy_classes import Theissetup, Stat2Dsetup
from welltestpy.tools.plotter import (
    plotfitting3D,
    plotfitting3Dtheis,
    plotparainteract,
    plotparatrace,
    plotsensitivity,
)

__all__ = ["Stat2Dest", "Theisest"]


class Stat2Dest(object):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended theis solution in 2D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function.
    """

    def __init__(self, name, campaign, testinclude=None):
        """Estimation initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Estimation.
        campaign : :class:`welltestpy.data.Campaign`
            The pumping test campaign which should be used to estimate the
            paramters
        testinclude : :class:`dict`, optional
            dictonary of which tests should be included. If ``None`` is given,
            all available tests are included.
            Default: ``None``
        """
        self.name = name
        """:class:`str`: Name of the Estimation"""
        self.campaign_raw = dcopy(campaign)
        """:class:`welltestpy.data.Campaign`:\
        Copy of the original input campaign"""
        self.campaign = dcopy(campaign)
        """:class:`welltestpy.data.Campaign`:\
        Copy of the input campaign to be modified"""

        self.prate = None
        """:class:`float`: Pumpingrate at the pumping well"""

        self.time = None
        """:class:`numpy.ndarray`: time points of the observation"""
        self.rad = None
        """:class:`numpy.ndarray`: array of the radii from the wells"""
        self.data = None
        """:class:`numpy.ndarray`: observation data"""
        self.radnames = None
        """:class:`numpy.ndarray`: names of the radii well combination"""

        self.para = None
        """:class:`list` of :class:`float`: estimated parameters"""
        self.result = None
        """:class:`list`: result of the spotpy estimation"""
        self.sens = None
        """:class:`list`: result of the spotpy sensitivity analysis"""
        self.testinclude = {}
        """:class:`dict`: dictonary of which tests should be included"""

        if testinclude is None:
            wells = list(self.campaign.tests.keys())
            pumpdict = {}
            for wel in wells:
                pumpdict[wel] = list(
                    self.campaign.tests[wel].observations.keys()
                )
            self.testinclude = pumpdict
        else:
            self.testinclude = testinclude

        rwell_list = []
        rinf_list = []
        for wel in self.testinclude:
            rwell_list.append(self.campaign.wells[wel].radius)
            rinf_list.append(self.campaign.tests[wel].aquiferradius)

        self.rwell = min(rwell_list)
        """:class:`float`: radius of the pumping wells"""
        self.rinf = max(rinf_list)
        """:class:`float`: radius of the aquifer"""
        print("rwell", self.rwell)
        print("rinf", self.rinf)

    def setpumprate(self, prate=-1.0):
        """Set a uniform pumping rate at all pumpingwells wells.

        Parameters
        ----------
        prate : :class:`float`, optional
            Pumping rate. Default: ``-1.0``
        """
        for test in self.testinclude:
            normpumptest(self.campaign.tests[test], pumpingrate=prate)
        self.prate = prate

    def settime(self, time=None, tmin=10.0, tmax=np.inf, typ="quad", steps=10):
        """Set the uniform time points at which the observations should be
        evaluated.

        Parameters
        ----------
        time : :class:`numpy.ndarray`, optional
            Array of specified time points. If ``None`` is given, they will
            be determind by the observation data.
            Default: ``None``
        tmin : :class:`float`, optional
            Minimal time value. It will set a minimal value of 10s.
            Default: ``10``
        tmax : :class:`float`, optional
            Maximal time value.
            Default: ``inf``
        typ : :class:`str`, optional
            Typ of the time selection. You can select from:

                * ``quad``: Quadratically increasing time steps
                * ``geom``: Geometrically increasing time steps
                * ``exp``: Exponentially increasing time steps
                * ``lin``: Linear time steps

            Default: "quad"

        steps : :class:`int`, optional
            Number of generated time steps. Default: 10
        """
        if time is None:
            for test in self.testinclude:
                for obs in self.testinclude[test]:
                    temptime, _ = self.campaign.tests[test].observations[obs]()
                    tmin = max(tmin, temptime.min())
                    tmax = min(tmax, temptime.max())

            if typ == "exp":
                time = np.expm1(
                    np.linspace(np.log1p(tmin), np.log1p(tmax), 10)
                )
            elif typ == "geom":
                time = np.geomspace(tmin, tmax, 10)
            elif typ == "quad":
                time = np.power(
                    np.linspace(np.sqrt(tmin), np.sqrt(tmax), 10), 2
                )
            else:
                time = np.linspace(tmin, tmax, 10)

        for test in self.testinclude:
            for obs in self.testinclude[test]:
                filterdrawdown(
                    self.campaign.tests[test].observations[obs], tout=time
                )

        self.time = time

    def genrtdata(self):
        """Generate the observed drawdown at given time points.

        It will also generate an array containing all radii of all well
        combinations.
        """
        rad = np.array([])
        data = None

        radnames = np.array([])

        for test in self.testinclude:
            pwell = self.campaign.wells[self.campaign.tests[test].pumpingwell]
            for obs in self.testinclude[test]:
                _, temphead = self.campaign.tests[test].observations[obs]()
                temphead = temphead.reshape(-1)[np.newaxis].T

                if data is None:
                    data = dcopy(temphead)
                else:
                    data = np.hstack((data, temphead))

                owell = self.campaign.wells[obs]

                if pwell == owell:
                    temprad = pwell.radius
                else:
                    temprad = pwell - owell
                rad = np.hstack((rad, temprad))

                tempname = self.campaign.tests[test].pumpingwell + "-" + obs
                radnames = np.hstack((radnames, tempname))

        # sort everything by the radii
        idx = rad.argsort()

        self.rad = rad[idx]
        self.data = data[:, idx]
        self.radnames = radnames[idx]

    def run(
        self,
        rep=5000,
        parallel="seq",
        run=True,
        folder=None,
        dbname=None,
        plotname1=None,
        plotname2=None,
        plotname3=None,
        estname=None,
        mu=None,
        sig2=None,
        corr=None,
        lnS=None,
        murange=(-16.0, -2.0),
        sig2range=(0.1, 10.0),
        corrrange=(1.0, 50.0),
        lnSrange=(-13.0, -1.0),
        rwell=0.0,
        rinf=None,
    ):
        """Run the estimation.

        Parameters
        ----------
        rep : :class:`int`, optional
            The number of repetitions within the SCEua algorithm in spotpy.
            Default: ``5000``
        parallel : :class:`str`, optional
            State if the estimation should be run in parallel or not. Options:

                    * ``"seq"``: sequential on one CPU
                    * ``"mpi"``: use the mpi4py package

            Default: ``"seq"``
        run : :class:`bool`, optional
            State if the estimation should be executed. Otherwise all plots
            will be done with the previous results.
            Default: ``True``
        folder : :class:`str`, optional
            Path to the output folder. If ``None`` the CWD is used.
            Default: ``None``
        dbname : :class:`str`, optional
            File-name of the database of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_stat2D_db"``.
            Default: ``None``
        plotname1 : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_stat2D_plot_paratrace.pdf"``.
            Default: ``None``
        plotname2 : :class:`str`, optional
            File-name of the fitting plot of the estimation.
            If ``None``, it will be the actual time +
            ``"_stat2D_plot_fit.pdf"``.
            Default: ``None``
        plotname3 : :class:`str`, optional
            File-name of the parameter interaction plot
            of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_stat2D_plot_parainteract.pdf"``.
            Default: ``None``
        estname : :class:`str`, optional
            File-name of the results of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_estimate"``.
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
        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd() + "/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        os.makedirs(dire, exist_ok=True)

        if dbname is None:
            dbname = folder + act_time + "_stat2D_db"
        else:
            dbname = folder + dbname
        if plotname1 is None:
            plotname1 = folder + act_time + "_stat2D_plot_paratrace.pdf"
        else:
            plotname1 = folder + plotname1
        if plotname2 is None:
            plotname2 = folder + act_time + "_stat2D_plot_fit.pdf"
        else:
            plotname2 = folder + plotname2
        if plotname3 is None:
            plotname3 = folder + act_time + "_stat2D_plot_parainteract.pdf"
        else:
            plotname3 = folder + plotname3
        if estname is None:
            paraname = folder + act_time + "_estimate.txt"
        else:
            paraname = folder + estname

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r"$\mu$")
            paranames.append("mu")
        if sig2 is None:
            paralabels.append(r"$\sigma^{2}$")
            paranames.append("sig2")
        if corr is None:
            paralabels.append(r"$\ell$")
            paranames.append("corr")
        if lnS is None:
            paralabels.append(r"$\ln(S)$")
            paranames.append("lnS")

        if rwell is None:
            rwell = self.rwell
        if rinf is None:
            rinf = self.rinf

        if run:
            # generate the spotpy-setup
            setup = Stat2Dsetup(
                self.rad,
                self.time,
                self.data,
                Qw=self.prate,
                mu=mu,
                sig2=sig2,
                corr=corr,
                lnS=lnS,
                # to fix some parameters
                murange=murange,
                sig2range=sig2range,
                corrrange=corrrange,
                lnSrange=lnSrange,
                rwell=rwell,
                rinf=rinf,
            )
            # initialize the sampler
            sampler = spotpy.algorithms.sceua(
                setup,
                dbname=dbname,
                dbformat="csv",
                parallel=parallel,
                save_sim=False,
                #                alt_objfun=None,  # use -rmse for fitting
            )
            # start the estimation with the sce-ua algorithm
            sampler.sample(rep, ngs=10, kstop=100, pcento=1e-4, peps=1e-3)
            # save best parameter-set
            self.para = sampler.status.params
            np.savetxt(paraname, self.para)
            # save the results
            self.result = sampler.getdata()

        # plot the estimation-results
        plotparatrace(
            self.result,
            parameternames=paranames,
            parameterlabels=paralabels,
            stdvalues=self.para,
            filename=plotname1,
        )
        plotfitting3D(
            self.data,
            self.para,
            self.rad,
            self.time,
            self.radnames,
            self.prate,
            plotname2,
            rwell=rwell,
            rinf=rinf,
        )
        plotparainteract(self.result, paralabels, plotname3)

    def sensitivity(
        self,
        rep=5000,
        parallel="seq",
        folder=None,
        dbname=None,
        plotname=None,
        plotname1=None,
        mu=None,
        sig2=None,
        corr=None,
        lnS=None,
        murange=None,
        sig2range=None,
        corrrange=None,
        lnSrange=None,
        rwell=0.0,
        rinf=None,
    ):
        """Run the sensitivity analysis.

        Parameters
        ----------
        rep : :class:`int`, optional
            The number of repetitions within the FAST algorithm in spotpy.
            Default: ``5000``
        parallel : :class:`str`, optional
            State if the estimation should be run in parallel or not. Options:

                    * ``"seq"``: sequential on one CPU
                    * ``"mpi"``: use the mpi4py package

            Default: ``"seq"``
        folder : :class:`str`, optional
            Path to the output folder. If ``None`` the CWD is used.
            Default: ``None``
        dbname : :class:`str`, optional
            File-name of the database of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_sensitivity_db"``.
            Default: ``None``
        plotname : :class:`str`, optional
            File-name of the result plot of the sensitivity analysis.
            If ``None``, it will be the actual time +
            ``"_stat2D_plot_sensitivity.pdf"``.
            Default: ``None``
        plotname1 : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy sensitivity
            analysis.
            If ``None``, it will be the actual time +
            ``"_stat2D_plot_senstrace.pdf"``.
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
        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd() + "/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        os.makedirs(dire, exist_ok=True)

        if dbname is None:
            dbname = folder + act_time + "_sensitivity_db"
        else:
            dbname = folder + dbname
        if plotname is None:
            plotname = folder + act_time + "_stat2D_plot_sensitivity.pdf"
        else:
            plotname = folder + plotname
        if plotname1 is None:
            plotname1 = folder + act_time + "_stat2D_plot_senstrace.pdf"
        else:
            plotname1 = folder + plotname1
        sensname = folder + act_time + "_FAST_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r"$T^G$")
            paranames.append("mu")
        if sig2 is None:
            paralabels.append(r"$\sigma^{2}$")
            paranames.append("sig2")
        if corr is None:
            paralabels.append(r"$\ell$")
            paranames.append("corr")
        if lnS is None:
            paralabels.append(r"$S$")
            paranames.append("lnS")

        bestvalues = {}

        for par_i, par_e in enumerate(paranames):
            bestvalues[par_e] = self.para[par_i]

        if rwell is None:
            rwell = self.rwell
        if rinf is None:
            rinf = self.rinf

        # generate the spotpy-setup
        setup = Stat2Dsetup(
            self.rad,
            self.time,
            self.data,
            Qw=self.prate,
            bestvalues=bestvalues,
            mu=mu,
            sig2=sig2,
            corr=corr,
            lnS=lnS,
            # to fix some parameters
            murange=murange,
            sig2range=sig2range,
            corrrange=corrrange,
            lnSrange=lnSrange,
            rwell=rwell,
            rinf=rinf,
        )

        # initialize the sampler
        sampler = spotpy.algorithms.fast(
            setup,
            dbname=dbname,
            dbformat="csv",
            parallel=parallel,
            save_sim=True,
        )

        sampler.sample(rep)

        data = sampler.getdata()

        parmin = sampler.parameter()["minbound"]
        parmax = sampler.parameter()["maxbound"]

        bounds = list(zip(parmin, parmax))

        self.sens = sampler.analyze(
            bounds, np.nan_to_num(data["like1"]), len(self.para), paranames
        )

        np.savetxt(sensname, self.sens["ST"])

        plotsensitivity(paralabels, self.sens, plotname)
        plotparatrace(
            data,
            parameternames=paranames,
            parameterlabels=paralabels,
            stdvalues=self.para,
            filename=plotname1,
        )


# theis


class Theisest(object):
    """Class for an estimation of homogeneous subsurface parameters.

    With this class you can run an estimation of homogeneous subsurface
    parameters. It utilizes the theis solution.
    """

    def __init__(self, name, campaign, testinclude=None):
        """Estimation initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Estimation.
        campaign : :class:`Campaign`
            The pumping test campaign which should be used to estimate the
            paramters
        testinclude : :class:`dict`, optional
            dictonary of which tests should be included. If ``None`` is given,
            all available tests are included.
            Default: ``None``
        """
        self.name = name
        """:class:`str`: Name of the Estimation"""
        self.campaign_raw = dcopy(campaign)
        """:class:`welltestpy.data.Campaign`:\
        Copy of the original input campaign"""
        self.campaign = dcopy(campaign)
        """:class:`welltestpy.data.Campaign`:\
        Copy of the input campaign to be modified"""
        self.prate = None
        """:class:`float`: Pumpingrate at the pumping well"""
        self.time = None
        """:class:`numpy.ndarray`: time points of the observation"""
        self.rad = None
        """:class:`numpy.ndarray`: array of the radii from the wells"""
        self.data = None
        """:class:`numpy.ndarray`: observation data"""
        self.radnames = None
        """:class:`numpy.ndarray`: names of the radii well combination"""
        self.para = None
        """:class:`list` of :class:`float`: estimated parameters"""
        self.result = None
        """:class:`list`: result of the spotpy estimation"""
        self.sens = None
        """:class:`list`: result of the spotpy sensitivity analysis"""
        self.testinclude = {}
        """:class:`dict`: dictonary of which tests should be included"""

        if testinclude is None:
            wells = list(self.campaign.tests.keys())
            pumpdict = {}
            for wel in wells:
                pumpdict[wel] = list(
                    self.campaign.tests[wel].observations.keys()
                )
            self.testinclude = pumpdict
        else:
            self.testinclude = testinclude

    def setpumprate(self, prate=-1.0):
        """Set a uniform pumping rate at all pumpingwells wells.

        Parameters
        ----------
        prate : :class:`float`, optional
            Pumping rate. Default: ``-1.0``
        """
        for test in self.testinclude:
            normpumptest(self.campaign.tests[test], pumpingrate=prate)
        self.prate = prate

    def settime(self, time=None, tmin=10, tmax=np.inf):
        """Set the uniform time points at which the observations should be
        evaluated.

        Parameters
        ----------
        time : :class:`numpy.ndarray`, optional
            Array of specified time points. If ``None`` is given, they will
            be determind by the observation data.
            Default: ``None``
        tmin : :class:`float`, optional
            Minimal time value. It will set a minimal value of 10s.
            Default: ``-inf``
        tmax : :class:`float`, optional
            Maximal time value.
            Default: ``inf``
        """
        if time is None:
            for test in self.testinclude:
                for obs in self.testinclude[test]:
                    temptime, _ = self.campaign.tests[test].observations[obs]()
                    tmin = max(tmin, temptime.min())
                    tmax = min(tmax, temptime.max())

            time = np.power(np.linspace(np.sqrt(tmin), np.sqrt(tmax), 10), 2)

        for test in self.testinclude:
            for obs in self.testinclude[test]:
                filterdrawdown(
                    self.campaign.tests[test].observations[obs], tout=time
                )

        self.time = time

    def genrtdata(self):
        """Generate the observed drawdown at given time points.

        It will also generate an array containing all radii of all well
        combinations.
        """
        rad = np.array([])
        data = None

        radnames = np.array([])

        for test in self.testinclude:
            pwell = self.campaign.wells[self.campaign.tests[test].pumpingwell]
            for obs in self.testinclude[test]:
                _, temphead = self.campaign.tests[test].observations[obs]()
                temphead = temphead.reshape(-1)[np.newaxis].T

                if data is None:
                    data = dcopy(temphead)
                else:
                    data = np.hstack((data, temphead))

                owell = self.campaign.wells[obs]

                if pwell == owell:
                    temprad = pwell.radius
                else:
                    temprad = pwell - owell
                rad = np.hstack((rad, temprad))

                tempname = self.campaign.tests[test].pumpingwell + "-" + obs
                radnames = np.hstack((radnames, tempname))

        # sort everything by the radii
        idx = rad.argsort()

        self.rad = rad[idx]
        self.data = data[:, idx]
        self.radnames = radnames[idx]

    def run(
        self,
        rep=5000,
        parallel="seq",
        run=True,
        folder=None,
        dbname=None,
        plotname1=None,
        plotname2=None,
        plotname3=None,
        estname=None,
        mu=None,
        lnS=None,
        murange=None,
        lnSrange=None,
    ):
        """Run the estimation.

        Parameters
        ----------
        rep : :class:`int`, optional
            The number of repetitions within the SCEua algorithm in spotpy.
            Default: ``5000``
        parallel : :class:`str`, optional
            State if the estimation should be run in parallel or not. Options:

                    * ``"seq"``: sequential on one CPU
                    * ``"mpi"``: use the mpi4py package

            Default: ``"seq"``
        run : :class:`bool`, optional
            State if the estimation should be executed. Otherwise all plots
            will be done with the previous results.
            Default: ``True``
        folder : :class:`str`, optional
            Path to the output folder. If ``None`` the CWD is used.
            Default: ``None``
        dbname : :class:`str`, optional
            File-name of the database of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_db"``.
            Default: ``None``
        plotname1 : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_plot_paratrace.pdf"``.
            Default: ``None``
        plotname2 : :class:`str`, optional
            File-name of the fitting plot of the estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_plot_fit.pdf"``.
            Default: ``None``
        plotname3 : :class:`str`, optional
            File-name of the parameter interaction plot
            of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_plot_parainteract.pdf"``.
            Default: ``None``
        estname : :class:`str`, optional
            File-name of the results of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_estimate"``.
            Default: ``None``
        mu : :class:`float`, optional
            Here you can fix the value for mean log-transmissivity ``mu``.
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
        """
        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd() + "/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        os.makedirs(dire, exist_ok=True)

        if dbname is None:
            dbname = folder + act_time + "_Theis_db"
        else:
            dbname = folder + dbname
        if plotname1 is None:
            plotname1 = folder + act_time + "_Theis_plot_paratrace.pdf"
        else:
            plotname1 = folder + plotname1
        if plotname2 is None:
            plotname2 = folder + act_time + "_Theis_plot_fit.pdf"
        else:
            plotname2 = folder + plotname2
        if plotname3 is None:
            plotname3 = folder + act_time + "_Theis_plot_parainteract.pdf"
        else:
            plotname3 = folder + plotname3
        if estname is None:
            paraname = folder + act_time + "_Theis_estimate.txt"
        else:
            paraname = folder + estname

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r"$\mu$")
            paranames.append("mu")
        if lnS is None:
            paralabels.append(r"$\ln(S)$")
            paranames.append("lnS")

        if run:
            # generate the spotpy-setup
            setup = Theissetup(
                self.rad,
                self.time,
                self.data,
                Qw=self.prate,
                mu=mu,
                lnS=lnS,  # to fix some parameters
                murange=murange,
                lnSrange=lnSrange,
            )
            # initialize the sampler
            sampler = spotpy.algorithms.sceua(
                setup,
                dbname=dbname,
                dbformat="csv",
                parallel=parallel,
                save_sim=False,
                #                alt_objfun=None,  # use -rmse for fitting
            )
            # start the estimation with the sce-ua algorithm
            sampler.sample(rep, ngs=10, kstop=100, pcento=1e-4, peps=1e-3)
            # save best parameter-set
            self.para = sampler.status.params
            np.savetxt(paraname, self.para)
            # save the results
            self.result = sampler.getdata()

        # plot the estimation-results
        plotparatrace(
            self.result,
            parameternames=paranames,
            parameterlabels=paralabels,
            stdvalues=self.para,
            filename=plotname1,
        )
        plotfitting3Dtheis(
            self.data,
            self.para,
            self.rad,
            self.time,
            self.radnames,
            self.prate,
            plotname2,
        )
        plotparainteract(self.result, paralabels, plotname3)

    def sensitivity(
        self,
        rep=5000,
        parallel="seq",
        folder=None,
        dbname=None,
        plotname=None,
        mu=None,
        lnS=None,
        murange=None,
        lnSrange=None,
    ):
        """Run the sensitivity analysis.

        Parameters
        ----------
        rep : :class:`int`, optional
            The number of repetitions within the FAST algorithm in spotpy.
            Default: ``5000``
        parallel : :class:`str`, optional
            State if the estimation should be run in parallel or not. Options:

                    * ``"seq"``: sequential on one CPU
                    * ``"mpi"``: use the mpi4py package

            Default: ``"seq"``
        folder : :class:`str`, optional
            Path to the output folder. If ``None`` the CWD is used.
            Default: ``None``
        dbname : :class:`str`, optional
            File-name of the database of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_Theis_sensitivity_db"``.
            Default: ``None``
        plotname : :class:`str`, optional
            File-name of the sensitivity plot.
            If ``None``, it will be the actual time +
            ``"_Theis_plot_sensitivity.pdf"``.
            Default: ``None``
        mu : :class:`float`, optional
            Here you can fix the value for mean log-transmissivity ``mu``.
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
        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd() + "/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        os.makedirs(dire, exist_ok=True)

        if dbname is None:
            dbname = folder + act_time + "_Theis_sensitivity_db"
        else:
            dbname = folder + dbname
        if plotname is None:
            plotname = folder + act_time + "_Theis_plot_sensitivity.pdf"
        else:
            plotname = folder + plotname
        sensname = folder + act_time + "_Theis_FAST_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r"$T^G$")
            paranames.append("mu")
        if lnS is None:
            paralabels.append(r"$S$")
            paranames.append("lnS")

        bestvalues = {}

        for par_i, par_e in enumerate(paranames):
            bestvalues[par_e] = self.para[par_i]

        # generate the spotpy-setup
        setup = Theissetup(
            self.rad,
            self.time,
            self.data,
            Qw=self.prate,
            bestvalues=bestvalues,
            mu=mu,
            lnS=lnS,  # to fix some parameters
            murange=murange,
            lnSrange=lnSrange,
        )

        # initialize the sampler
        sampler = spotpy.algorithms.fast(
            setup,
            dbname=dbname,
            dbformat="csv",
            parallel=parallel,
            save_sim=True,
        )

        sampler.sample(rep)

        data = sampler.getdata()

        parmin = sampler.parameter()["minbound"]
        parmax = sampler.parameter()["maxbound"]

        bounds = list(zip(parmin, parmax))

        self.sens = sampler.analyze(
            bounds, np.nan_to_num(data["like1"]), len(self.para), paranames
        )

        np.savetxt(sensname, self.sens["ST"])

        plotsensitivity(paralabels, self.sens, plotname)
