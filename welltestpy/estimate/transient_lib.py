# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing base classe for transient estimations.

.. currentmodule:: welltestpy.estimate.transient_lib

The following classes are provided

.. autosummary::
   TransientPumping
"""
from copy import deepcopy as dcopy
import os
import time as timemodule

import numpy as np
import spotpy
import anaflow as ana

from ..data import testslib
from ..process import processlib
from . import spotpylib
from ..tools import plotter

__all__ = [
    "TransientPumping",
]


class TransientPumping:
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

        self.estimated_para = {}
        """:class:`dict`: estimated parameters by name"""
        self.result = None
        """:class:`list`: result of the spotpy estimation"""
        self.sens = None
        """:class:`dict`: result of the spotpy sensitivity analysis"""
        self.testinclude = {}
        """:class:`dict`: dictonary of which tests should be included"""

        if testinclude is None:
            tests = list(self.campaign.tests.keys())
            self.testinclude = {}
            for test in tests:
                self.testinclude[test] = self.campaign.tests[
                    test
                ].observationwells
        elif not isinstance(testinclude, dict):
            self.testinclude = {}
            for test in testinclude:
                self.testinclude[test] = self.campaign.tests[
                    test
                ].observationwells
        else:
            self.testinclude = testinclude

        for test in self.testinclude:
            if not isinstance(self.campaign.tests[test], testslib.PumpingTest):
                raise ValueError(test + " is not a pumping test.")
            if not self.campaign.tests[test].constant_rate:
                raise ValueError(test + " is not a constant rate test.")
            if (
                not self.campaign.tests[test].state(
                    wells=self.testinclude[test]
                )
                == "transient"
            ):
                raise ValueError(test + ": selection is not transient.")

        rwell_list = []
        rinf_list = []
        for test in self.testinclude:
            pwell = self.campaign.tests[test].pumpingwell
            rwell_list.append(self.campaign.wells[pwell].radius)
            rinf_list.append(self.campaign.tests[test].radius)
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
            processlib.normpumptest(
                self.campaign.tests[test], pumpingrate=prate
            )
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
        typ : :class:`str` or :class:`float`, optional
            Typ of the time selection. You can select from:

                * ``"exp"``: for exponential behavior
                * ``"log"``: for logarithmic behavior
                * ``"geo"``: for geometric behavior
                * ``"lin"``: for linear behavior
                * ``"quad"``: for quadratic behavior
                * ``"cub"``: for cubic behavior
                * :class:`float`: here you can specifi any exponent
                  ("quad" would be equivalent to 2)

            Default: "quad"

        steps : :class:`int`, optional
            Number of generated time steps. Default: 10
        """
        if time is None:
            for test in self.testinclude:
                for obs in self.testinclude[test]:
                    _, temptime = self.campaign.tests[test].observations[obs]()
                    tmin = max(tmin, temptime.min())
                    tmax = min(tmax, temptime.max())
                    tmin = tmax if tmin > tmax else tmin
            time = ana.specialrange(tmin, tmax, steps, typ)

        for test in self.testinclude:
            for obs in self.testinclude[test]:
                processlib.filterdrawdown(
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

        radnames = []

        for test in self.testinclude:
            pwell = self.campaign.wells[self.campaign.tests[test].pumpingwell]
            for obs in self.testinclude[test]:
                temphead, _ = self.campaign.tests[test].observations[obs]()
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

                tempname = (self.campaign.tests[test].pumpingwell, obs)
                radnames.append(tempname)

        # sort everything by the radii
        idx = rad.argsort()
        radnames = np.array(radnames)
        self.rad = rad[idx]
        self.data = data[:, idx]
        self.radnames = radnames[idx]

    def gen_setup(
        self, prate_kw="rate", rad_kw="rad", time_kw="time", dummy=False
    ):
        """Generate the Spotpy Setup.

        Parameters
        ----------
        prate_kw : :class:`str`, optional
            Keyword name for the pumping rate in the used type curve.
            Default: "rate"
        rad_kw : :class:`str`, optional
            Keyword name for the radius in the used type curve.
            Default: "rad"
        time_kw : :class:`str`, optional
            Keyword name for the time in the used type curve.
            Default: "time"
        dummy : :class:`bool`, optional
            Add a dummy parameter to the model. This could be used to equalize
            sensitivity analysis.
            Default: False
        """
        self.extra_kw_names = {"Qw": prate_kw, "rad": rad_kw, "time": time_kw}
        self.setup_kw["val_fix"].setdefault(prate_kw, self.prate)
        self.setup_kw["val_fix"].setdefault(rad_kw, self.rad)
        self.setup_kw["val_fix"].setdefault(time_kw, self.time)
        self.setup_kw.setdefault("data", self.data)
        self.setup_kw["dummy"] = dummy
        self.setup = spotpylib.TypeCurve(**self.setup_kw)

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
            If ``None``, it will be the current time +
            ``"_db"``.
            Default: ``None``
        traceplotname : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy estimation.
            If ``None``, it will be the current time +
            ``"_paratrace.pdf"``.
            Default: ``None``
        fittingplotname : :class:`str`, optional
            File-name of the fitting plot of the estimation.
            If ``None``, it will be the current time +
            ``"_fit.pdf"``.
            Default: ``None``
        interactplotname : :class:`str`, optional
            File-name of the parameter interaction plot
            of the spotpy estimation.
            If ``None``, it will be the current time +
            ``"_parainteract.pdf"``.
            Default: ``None``
        estname : :class:`str`, optional
            File-name of the results of the spotpy estimation.
            If ``None``, it will be the current time +
            ``"_estimate"``.
            Default: ``None``
        """
        if self.setup.dummy:
            raise ValueError(
                "Estimate: for parameter estimation"
                + " you can't use a dummy paramter."
            )
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
        paranames = dcopy(self.setup.para_names)
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
                para = []
                header = []
                for name in void_names:
                    para.append(para_opt[0][name])
                    header.append(name[3:])
                    self.estimated_para[header[-1]] = para[-1]
                np.savetxt(paraname, para, header=" ".join(header))

        if rank == 0:
            # plot the estimation-results
            plotter.plotparatrace(
                self.result,
                parameternames=paranames,
                parameterlabels=paralabels,
                stdvalues=self.estimated_para,
                plotname=traceplotname,
            )
            plotter.plotfit_transient(
                setup=self.setup,
                data=self.data,
                para=self.estimated_para,
                rad=self.rad,
                time=self.time,
                radnames=self.radnames,
                extra=self.extra_kw_names,
                plotname=fittingplotname,
            )
            plotter.plotparainteract(self.result, paralabels, interactplotname)

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
            Default: estimated
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
            If ``None``, it will be the current time +
            ``"_sensitivity_db"``.
            Default: ``None``
        plotname : :class:`str`, optional
            File-name of the result plot of the sensitivity analysis.
            If ``None``, it will be the current time +
            ``"_sensitivity.pdf"``.
            Default: ``None``
        traceplotname : :class:`str`, optional
            File-name of the parameter trace plot of the spotpy sensitivity
            analysis.
            If ``None``, it will be the current time +
            ``"_senstrace.pdf"``.
            Default: ``None``
        sensname : :class:`str`, optional
            File-name of the results of the FAST estimation.
            If ``None``, it will be the current time +
            ``"_estimate"``.
            Default: ``None``
        """
        if len(self.setup.para_names) == 1 and not self.setup.dummy:
            raise ValueError(
                "Sensitivity: for estimation with only one parameter"
                + " you have to use a dummy paramter."
            )
        if rep is None:
            rep = spotpylib.fast_rep(
                len(self.setup.para_names) + int(self.setup.dummy)
            )

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

        sens_base, sens_ext = os.path.splitext(sensname)
        sensname1 = sens_base + "_S1" + sens_ext

        # generate the parameter-names for plotting
        paranames = dcopy(self.setup.para_names)
        paralabels = [self.setup.val_plot_names[name] for name in paranames]

        if self.setup.dummy:
            paranames.append("dummy")
            paralabels.append("dummy")

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
            sens_est = sampler.analyze(
                bounds, np.nan_to_num(data["like1"]), len(paranames), paranames
            )
            self.sens = {}
            for sen_typ in sens_est:
                self.sens[sen_typ] = {
                    par: sen for par, sen in zip(paranames, sens_est[sen_typ])
                }
            header = " ".join(paranames)
            np.savetxt(sensname, sens_est["ST"], header=header)
            np.savetxt(sensname1, sens_est["S1"], header=header)
            plotter.plotsensitivity(paralabels, sens_est, plotname)
            plotter.plotparatrace(
                data,
                parameternames=paranames,
                parameterlabels=paralabels,
                stdvalues=None,
                plotname=traceplotname,
            )
