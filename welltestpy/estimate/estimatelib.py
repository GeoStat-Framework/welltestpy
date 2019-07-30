# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing classes for parameter estimation.

.. currentmodule:: welltestpy.estimate.estimatelib

The following classes are provided

.. autosummary::
   TransientPumping
   ExtTheis3D
   ExtTheis2D
   Theis
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import time as timemodule

import numpy as np
import spotpy
import anaflow as ana

from welltestpy.process.processlib import normpumptest, filterdrawdown
from welltestpy.estimate.spotpy_classes import TypeCurve
from welltestpy.tools.plotter import (
    plotfit_transient,
    plotparainteract,
    plotparatrace,
    plotsensitivity,
)

__all__ = ["TransientPumping", "ExtTheis3D", "ExtTheis2D", "Theis"]


def fast_rep(para_no, infer_fac=4, freq_step=2):
    """Number of iterations needed for the FAST algorithm.

    Parameters
    ----------
    para_no : :class:`int`
        Number of parameters in the model.
    infer_fac : :class:`int`, optional
        The inference fractor. Default: 4
    freq_step : :class:`int`, optional
        The frequency step. Default: 2
    """
    return 2 * int(
        para_no * (1 + 4 * infer_fac ** 2 * (1 + (para_no - 2) * freq_step))
    )


class TransientPumping(object):
    """Class to estimate transient Type-Curve parameters.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        paramters
    type_curve : :any:`callable`
        The given type-curve. Output will be reshaped to flat array.
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Default: None
    fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting type for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        fit_type can be "lin", "log" (np.exp(val) will be used)
        or a callable function.
        By default, values will be fit linearly.
        Default: None
    val_kw_names : :class:`dict` or :any:`None`
        Dictionary containing keyword names in the type-curve for each value.

            {value-name: kwargs-name in type_curve}

        This is usefull if fitting is not done by linear values.
        By default, parameter names will be value names.
        Default: None
    val_plot_names : :class:`dict` or :any:`None`
        Dictionary containing keyword names in the type-curve for each value.

            {value-name: string for plot legend}

        This is usefull to get better plots.
        By default, parameter names will be value names.
        Default: None
    testinclude : :class:`dict`, optional
        dictonary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        type_curve,
        val_ranges,
        val_fix=None,
        fit_type=None,
        val_kw_names=None,
        val_plot_names=None,
        testinclude=None,
        generate=False,
    ):
        val_fix = {} if val_fix is None else val_fix
        fit_type = {} if fit_type is None else fit_type
        val_kw_names = {} if val_kw_names is None else val_kw_names
        val_plot_names = {} if val_plot_names is None else val_plot_names
        self.setup_kw = {
            "type_curve": type_curve,
            "val_ranges": val_ranges,
            "val_fix": val_fix,
            "fit_type": fit_type,
            "val_kw_names": val_kw_names,
            "val_plot_names": val_plot_names,
        }
        """:class:`dict`: TypeCurve Spotpy Setup definition"""
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
        """:class:`float`: radius of the furthest wells"""

        if generate:
            self.setpumprate()
            self.settime()
            self.gen_data()
            self.gen_setup()

    def setpumprate(self, prate=-1.0):
        """Set a uniform pumping rate at all pumpingwells wells.

        We assume linear scaling by the pumpingrate.

        Parameters
        ----------
        prate : :class:`float`, optional
            Pumping rate. Default: ``-1.0``
        """
        for test in self.testinclude:
            normpumptest(self.campaign.tests[test], pumpingrate=prate)
        self.prate = prate

    def settime(self, time=None, tmin=10.0, tmax=np.inf, typ="quad", steps=10):
        """Set uniform time points for the observations.

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
                    tmin = tmax if tmin > tmax else tmin

            if typ in ["geom", "exp", "exponential"]:
                time = np.geomspace(tmin, tmax, steps)
            elif typ == "quad":
                time = np.power(
                    np.linspace(np.sqrt(tmin), np.sqrt(tmax), steps), 2
                )
            else:
                time = np.linspace(tmin, tmax, steps)

        for test in self.testinclude:
            for obs in self.testinclude[test]:
                filterdrawdown(
                    self.campaign.tests[test].observations[obs], tout=time
                )

        self.time = time

    def gen_data(self):
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
                temphead = np.array(temphead).reshape(-1)[np.newaxis].T

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

    def gen_setup(self, prate_kw="Qw", rad_kw="rad", time_kw="time"):
        """Generate the Spotpy Setup.

        Parameters
        ----------
        prate_kw : :class:`str`, optional
            Keyword name for the pumping rate in the used type curve.
            Default: "Qw"
        rad_kw : :class:`str`, optional
            Keyword name for the radius in the used type curve.
            Default: "rad"
        prate_kw : :class:`str`, optional
            Keyword name for the time in the used type curve.
            Default: "time"
        """
        self.extra_kw_names = {
            "Qw": prate_kw,
            "rad": rad_kw,
            "time": time_kw,
        }
        self.setup_kw["val_fix"].setdefault(prate_kw, self.prate)
        self.setup_kw["val_fix"].setdefault(rad_kw, self.rad)
        self.setup_kw["val_fix"].setdefault(time_kw, self.time)
        self.setup_kw.setdefault("data", self.data)
        self.setup = TypeCurve(**self.setup_kw)

    def run(
        self,
        rep=5000,
        parallel="seq",
        run=True,
        folder=None,
        dbname=None,
        traceplotname=None,
        fittingplotname=None,
        interactplotname=None,
        estname=None,
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
            ``"_db"``.
            Default: ``None``
        traceplotname : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_paratrace.pdf"``.
            Default: ``None``
        fittingplotname : :class:`str`, optional
            File-name of the fitting plot of the estimation.
            If ``None``, it will be the actual time +
            ``"_fit.pdf"``.
            Default: ``None``
        interactplotname : :class:`str`, optional
            File-name of the parameter interaction plot
            of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_parainteract.pdf"``.
            Default: ``None``
        estname : :class:`str`, optional
            File-name of the results of the spotpy estimation.
            If ``None``, it will be the actual time +
            ``"_estimate"``.
            Default: ``None``
        """
        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.path.join(os.getcwd(), self.name)
        folder = os.path.abspath(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if dbname is None:
            dbname = os.path.join(folder, act_time + "_db")
        elif not os.path.isabs(dbname):
            dbname = os.path.join(folder, dbname)
        if traceplotname is None:
            traceplotname = os.path.join(folder, act_time + "_paratrace.pdf")
        elif not os.path.isabs(traceplotname):
            traceplotname = os.path.join(folder, traceplotname)
        if fittingplotname is None:
            fittingplotname = os.path.join(folder, act_time + "_fit.pdf")
        elif not os.path.isabs(fittingplotname):
            fittingplotname = os.path.join(folder, fittingplotname)
        if interactplotname is None:
            interactplotname = os.path.join(folder, act_time + "_interact.pdf")
        elif not os.path.isabs(interactplotname):
            interactplotname = os.path.join(folder, interactplotname)
        if estname is None:
            paraname = os.path.join(folder, act_time + "_estimate.txt")
        elif not os.path.isabs(estname):
            paraname = os.path.join(folder, estname)

        # generate the parameter-names for plotting
        paranames = self.setup.para_names
        paralabels = [self.setup.val_plot_names[name] for name in paranames]

        if parallel == "mpi":
            # send the dbname of rank0
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            if rank == 0:
                print(rank, "send dbname:", dbname)
                for i in range(1, size):
                    comm.send(dbname, dest=i, tag=0)
            else:
                dbname = comm.recv(source=0, tag=0)
                print(rank, "got dbname:", dbname)
        else:
            rank = 0

        if run:
            # initialize the sampler
            sampler = spotpy.algorithms.sceua(
                self.setup,
                dbname=dbname,
                dbformat="csv",
                parallel=parallel,
                save_sim=True,
                db_precision=np.float64,
            )
            # start the estimation with the sce-ua algorithm
            sampler.sample(rep, ngs=10, kstop=100, pcento=1e-4, peps=1e-3)

            if rank == 0:
                # save best parameter-set
                self.result = sampler.getdata()
                para_opt = spotpy.analyser.get_best_parameterset(
                    self.result, maximize=False
                )
                void_names = para_opt.dtype.names
                self.para = []
                for name in void_names:
                    self.para.append(para_opt[0][name])
                np.savetxt(paraname, self.para)

        if rank == 0:
            # plot the estimation-results
            plotparatrace(
                self.result,
                parameternames=paranames,
                parameterlabels=paralabels,
                stdvalues=self.para,
                filename=traceplotname,
            )
            plotfit_transient(
                self.setup,
                self.data,
                self.para,
                self.rad,
                self.time,
                self.radnames,
                fittingplotname,
                self.extra_kw_names,
            )
            plotparainteract(self.result, paralabels, interactplotname)

    def sensitivity(
        self,
        rep=None,
        parallel="seq",
        folder=None,
        dbname=None,
        plotname=None,
        traceplotname=None,
        sensname=None,
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
            ``"_sensitivity.pdf"``.
            Default: ``None``
        traceplotname : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy sensitivity
            analysis.
            If ``None``, it will be the actual time +
            ``"_senstrace.pdf"``.
            Default: ``None``
        sensname : :class:`str`, optional
            File-name of the results of the FAST estimation.
            If ``None``, it will be the actual time +
            ``"_estimate"``.
            Default: ``None``
        """
        if rep is None:
            rep = fast_rep(len(self.setup.para_names))

        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")
        # generate the filenames
        if folder is None:
            folder = os.path.join(os.getcwd(), self.name)
        folder = os.path.abspath(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if dbname is None:
            dbname = os.path.join(folder, act_time + "_sensitivity_db")
        elif not os.path.isabs(dbname):
            dbname = os.path.join(folder, dbname)
        if plotname is None:
            plotname = os.path.join(folder, act_time + "_sensitivity.pdf")
        elif not os.path.isabs(plotname):
            plotname = os.path.join(folder, plotname)
        if traceplotname is None:
            traceplotname = os.path.join(folder, act_time + "_senstrace.pdf")
        elif not os.path.isabs(traceplotname):
            traceplotname = os.path.join(folder, traceplotname)
        if sensname is None:
            sensname = os.path.join(folder, act_time + "_FAST_estimate.txt")
        elif not os.path.isabs(sensname):
            sensname = os.path.join(folder, sensname)

        # generate the parameter-names for plotting
        paranames = self.setup.para_names
        paralabels = [self.setup.val_plot_names[name] for name in paranames]

        if parallel == "mpi":
            # send the dbname of rank0
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            if rank == 0:
                print(rank, "send dbname:", dbname)
                for i in range(1, size):
                    comm.send(dbname, dest=i, tag=0)
            else:
                dbname = comm.recv(source=0, tag=0)
                print(rank, "got dbname:", dbname)
        else:
            rank = 0

        # initialize the sampler
        sampler = spotpy.algorithms.fast(
            self.setup,
            dbname=dbname,
            dbformat="csv",
            parallel=parallel,
            save_sim=True,
            db_precision=np.float64,
        )
        sampler.sample(rep)

        if rank == 0:
            data = sampler.getdata()
            parmin = sampler.parameter()["minbound"]
            parmax = sampler.parameter()["maxbound"]
            bounds = list(zip(parmin, parmax))
            self.sens = sampler.analyze(
                bounds, np.nan_to_num(data["like1"]), len(paranames), paranames
            )
            np.savetxt(sensname, self.sens["ST"])
            plotsensitivity(paralabels, self.sens, plotname)
            plotparatrace(
                data,
                parameternames=paranames,
                parameterlabels=paralabels,
                stdvalues=None,
                filename=traceplotname,
            )


# ext_theis_3D


class ExtTheis3D(TransientPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended theis solution in 3D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function and an anisotropy ratio 0 < e <= 1.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        paramters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Default: None
    testinclude : :class:`dict`, optional
        dictonary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "mu": (-16, -2),
            "sig2": (0, 10),
            "corr": (1, 50),
            "lnS": (-13, -1),
            "e": (0, 1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        val_fix = {"L": 1.0} if val_fix is None else val_fix
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        fit_type = {"mu": "log", "lnS": "log"}
        val_kw_names = {"mu": "KG", "lnS": "S"}
        val_plot_names = {
            "mu": r"$\mu$",
            "sig2": r"$\sigma^2$",
            "corr": r"$\ell$",
            "lnS": r"$\ln(S)$",
            "e": "$e$",
        }
        super(ExtTheis3D, self).__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.ext_theis3D,
            val_ranges=val_ranges,
            val_fix=val_fix,
            fit_type=fit_type,
            val_kw_names=val_kw_names,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )

# ext_theis_2D


class ExtTheis2D(TransientPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended theis solution in 2D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        paramters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Default: None
    testinclude : :class:`dict`, optional
        dictonary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "mu": (-16, -2),
            "sig2": (0, 10),
            "corr": (1, 50),
            "lnS": (-13, -1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        fit_type = {"mu": "log", "lnS": "log"}
        val_kw_names = {"mu": "TG", "lnS": "S"}
        val_plot_names = {
            "mu": r"$\mu$",
            "sig2": r"$\sigma^2$",
            "corr": r"$\ell$",
            "lnS": r"$\ln(S)$",
        }
        super(ExtTheis2D, self).__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.ext_theis2D,
            val_ranges=val_ranges,
            val_fix=val_fix,
            fit_type=fit_type,
            val_kw_names=val_kw_names,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# theis


class Theis(TransientPumping):
    """Class for an estimation of homogeneous subsurface parameters.

    With this class you can run an estimation of homogeneous subsurface
    parameters. It utilizes the theis solution.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        paramters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Default: None
    testinclude : :class:`dict`, optional
        dictonary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "mu": (-16, -2),
            "lnS": (-13, -1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        fit_type = {"mu": "log", "lnS": "log"}
        val_kw_names = {"mu": "T", "lnS": "S"}
        val_plot_names = {
            "mu": r"$\mu$",
            "lnS": r"$\ln(S)$",
        }
        super(Theis, self).__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.theis,
            val_ranges=val_ranges,
            val_fix=val_fix,
            fit_type=fit_type,
            val_kw_names=val_kw_names,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )
