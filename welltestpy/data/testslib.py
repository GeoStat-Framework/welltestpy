# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing flow datastructures for tests on a fieldsite.

.. currentmodule:: welltestpy.data.testslib

The following classes and functions are provided

.. autosummary::
   Test
   PumpingTest
   load_test
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import csv
import shutil
import zipfile
from io import TextIOWrapper as TxtIO

import numpy as np

from welltestpy.tools._extimport import BytIO
from welltestpy.data.varlib import (
    Variable,
    Observation,
    StdyHeadObs,
    DrawdownObs,
    load_var,
    load_obs,
    _nextr,
    _formstr,
    _formname,
)

__all__ = ["Test", "PumpingTest", "load_test"]


class Test(object):
    """General class for a well based test.

    This is a class for a well based test on a field site.
    It has a name, a descrition and a timeframe string.

    Parameters
    ----------
    name : :class:`str`
        Name of the test.
    description : :class:`str`, optional
        Description of the test.
        Default: ``"no description"``
    timeframe : :class:`str`, optional
        Timeframe of the test.
        Default: ``None``
    """

    def __init__(self, name, description="no description", timeframe=None):
        self.name = _formstr(name)
        self.description = str(description)
        self.timeframe = str(timeframe)
        self._testtype = "Test"

    def __repr__(self):
        return (
            self.testtype + " '" + self.name + "', Info: " + self.description
        )

    @property
    def testtype(self):
        """:class:`str`: String containing the test type"""
        return self._testtype


class PumpingTest(Test):
    """Class for a pumping test.

    This is a class for a pumping test on a field site.
    It has a name, a descrition, a timeframe and a pumpingwell string.

    Parameters
    ----------
    name : :class:`str`
        Name of the test.
    pumpingwell : :class:`str`
        Pumping well of the test.
    pumpingrate : :class:`float` or :class:`Variable`
        Pumping rate of at the pumping well. If a `float` is given,
        it is assumed to be given in ``m^3/s``.
    observations : :class:`dict`, optional
        Observations made within the pumping test. The dict-keys are the
        well names of the observation wells or the pumpingwell. Values
        need to be an instance of :class:`Observation`
        Default: ``None``
    aquiferdepth : :class:`float` or :class:`Variable`, optional
        Aquifer depth at the field site. If a `float` is given,
        it is assumed to be given in ``m``.
        Default: ``1.0``
    aquiferradius : :class:`float` or :class:`Variable`, optional
        Aquifer radius ot the field site. If a `float` is given,
        it is assumed to be given in ``m``.
        Default: ``inf``
    description : :class:`str`, optional
        Description of the test.
        Default: ``"Pumpingtest"``
    timeframe : :class:`str`, optional
        Timeframe of the test.
        Default: ``None``
    """

    def __init__(
        self,
        name,
        pumpingwell,
        pumpingrate,
        observations=None,
        aquiferdepth=1.0,
        aquiferradius=np.inf,
        description="Pumpingtest",
        timeframe=None,
    ):
        super(PumpingTest, self).__init__(name, description, timeframe)

        self._testtype = "PumpingTest"

        self.pumpingwell = str(pumpingwell)

        if isinstance(pumpingrate, Variable):
            self._pumpingrate = dcopy(pumpingrate)
        else:
            self._pumpingrate = Variable(
                "pumpingrate",
                pumpingrate,
                "Q",
                "m^3/s",
                "Pumpingrate at test '" + self.name + "'",
            )
        if not self._pumpingrate.scalar:
            raise ValueError("PumpingTest: 'pumpingrate' needs to be scalar")

        if isinstance(aquiferdepth, Variable):
            self._aquiferdepth = dcopy(aquiferdepth)
        else:
            self._aquiferdepth = Variable(
                "aquiferdepth",
                aquiferdepth,
                "L_a",
                "m",
                "mean aquiferdepth for test '" + str(name) + "'",
            )
        if not self._aquiferdepth.scalar:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be scalar")
        if self.aquiferdepth <= 0.0:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be positiv")

        if isinstance(aquiferradius, Variable):
            self._aquiferradius = dcopy(aquiferradius)
        else:
            self._aquiferradius = Variable(
                "aquiferradius",
                aquiferradius,
                "R",
                "m",
                "mean aquiferradius for test '" + str(name) + "'",
            )
        if not self._aquiferradius.scalar:
            raise ValueError("PumpingTest: 'aquiferradius' needs to be scalar")
        if self.aquiferradius <= 0.0:
            raise ValueError(
                "PumpingTest: 'aquiferradius' " + "needs to be positiv"
            )

        if observations is None:
            self.__observations = {}
        else:
            self.observations = observations

    @property
    def wells(self):
        """:class:`tuple` of :class:`str`: all well names"""
        tmp = list(self.__observations.keys())
        tmp.append(self.pumpingwell)
        return tuple(set(tmp))

    @property
    def pumpingrate(self):
        """:class:`float`: pumping rate at the pumping well"""
        return self._pumpingrate.value

    @pumpingrate.setter
    def pumpingrate(self, pumpingrate):
        tmp = dcopy(self._pumpingrate)
        if isinstance(pumpingrate, Variable):
            self._pumpingrate = dcopy(pumpingrate)
        else:
            self._pumpingrate(pumpingrate)
        if not self._pumpingrate.scalar:
            self._pumpingrate = dcopy(tmp)
            raise ValueError("PumpingTest: 'pumpingrate' needs to be scalar")

    @property
    def aquiferdepth(self):
        """:class:`float`: aquifer depth at the field site"""
        return self._aquiferdepth.value

    @aquiferdepth.setter
    def aquiferdepth(self, aquiferdepth):
        tmp = dcopy(self._aquiferdepth)
        if isinstance(aquiferdepth, Variable):
            self._aquiferdepth = dcopy(aquiferdepth)
        else:
            self._aquiferdepth(aquiferdepth)
        if not self._aquiferdepth.scalar:
            self._aquiferdepth = dcopy(tmp)
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be scalar")
        if self.aquiferdepth <= 0.0:
            self._aquiferdepth = dcopy(tmp)
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be positiv")

    @property
    def aquiferradius(self):
        """:class:`float`: aquifer radius at the field site"""
        return self._aquiferradius.value

    @aquiferradius.setter
    def aquiferradius(self, aquiferradius):
        tmp = dcopy(self._aquiferradius)
        if isinstance(aquiferradius, Variable):
            self._aquiferradius = dcopy(aquiferradius)
        else:
            self._aquiferradius(aquiferradius)
        if not self._aquiferradius.scalar:
            self._aquiferradius = dcopy(tmp)
            raise ValueError("PumpingTest: 'aquiferradius' needs to be scalar")
        if self.aquiferradius <= 0.0:
            self._aquiferradius = dcopy(tmp)
            raise ValueError(
                "PumpingTest: 'aquiferradius' " + "needs to be positiv"
            )

    @property
    def observations(self):
        """:class:`dict`: observations made at the field site"""
        return self.__observations

    @observations.setter
    def observations(self, obs):
        if obs is not None:
            if isinstance(obs, dict):
                for k in obs.keys():
                    if not isinstance(obs[k], Observation):
                        raise ValueError(
                            "PumpingTest: some 'observations' "
                            + "are not of type Observation"
                        )
                self.__observations = dcopy(obs)
            else:
                raise ValueError(
                    "PumpingTest: 'observations' should"
                    + " be given as dictonary"
                )
        else:
            self.__observations = {}

    def add_steady_obs(
        self,
        well,
        observation,
        description="Steady State Drawdown observation",
    ):
        """
        Add steady drawdown observations.

        Parameters
        ----------
        well : :class:`str`
            well where the observation is made.
        observation : :class:`Variable`
            Observation.
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Steady observation"``
        """
        obs = StdyHeadObs(well, observation, description)
        self.addobservations(obs)

    def add_transient_obs(
        self,
        well,
        time,
        observation,
        description="Transient Drawdown observation",
    ):
        """
        Add transient drawdown observations.

        Parameters
        ----------
        well : :class:`str`
            well where the observation is made.
        time : :class:`Variable`
            Time points of observation.
        observation : :class:`Variable`
            Observation.
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Drawdown observation"``
        """
        obs = DrawdownObs(well, time, observation, description)
        self.addobservations(obs)

    def addobservations(self, obs):
        """Add some specified observations.

        This will add observations to the pumping test.

        Parameters
        ----------
        obs : :class:`dict`
            Observations to be added.
        """
        if isinstance(obs, dict):
            for k in obs:
                if not isinstance(obs[k], Observation):
                    raise ValueError(
                        "PumpingTest_addobservations: some "
                        + "'observations' are not "
                        + "of type Observation"
                    )
                if k in self.observations:
                    raise ValueError(
                        "PumpingTest_addobservations: some "
                        + "'observations' are already present"
                    )
            for k in obs:
                self.__observations[k] = dcopy(obs[k])
        elif isinstance(obs, Observation):
            if obs in self.observations:
                raise ValueError(
                    "PumpingTest_addobservations: "
                    + "'observation' are already present"
                )
            self.__observations[obs.name] = dcopy(obs)
        else:
            raise ValueError(
                "PumpingTest_addobservations: 'observations' "
                + "should be given as dictonary with well as key"
            )

    def delobservations(self, obs):
        """Delete some specified observations.

        This will delete observations from the pumping test. You can give a
        list of observations or a single observation by name.

        Parameters
        ----------
        obs : :class:`list` of :class:`str` or :class:`str`
            Observations to be deleted.
        """
        if isinstance(obs, (list, tuple)):
            for k in obs:
                if k in self.observations:
                    del self.__observations[k]
        else:
            if obs in self.observations:
                del self.__observations[obs]

    def _addplot(self, plt_ax, wells, exclude=None):
        """Generate a plot of the pumping test.

        This will plot the pumping test on the given figure axes.

        Parameters
        ----------
        ax : :class:`Axes`
            Axes where the plot should be done.
        wells : :class:`dict`
            Dictonary containing the well classes sorted by name.
        exclude: :class:`list`, optional
            List of wells that should be excluded from the plot.
            Default: ``None``

        Notes
        -----
        This will be used by the Campaign class.
        """
        if exclude is None:
            exclude = []
        for k in self.observations:
            if k in exclude:
                continue
            if k != self.pumpingwell:
                dist = wells[k] - wells[self.pumpingwell]
            else:
                dist = wells[self.pumpingwell].radius
            if self.pumpingrate > 0:
                displace = np.maximum(self.observations[k].value[1], 0.0)
            else:
                displace = np.minimum(self.observations[k].value[1], 0.0)

            plt_ax.plot(
                self.observations[k].value[0],
                displace,
                linewidth=2,
                label=(self.observations[k].name + " r={:1.2f}".format(dist)),
            )
            plt_ax.set_xlabel(self.observations[k].labels[0])
            plt_ax.set_ylabel(self.observations[k].labels[1])

        plt_ax.set_title(repr(self))
        plt_ax.legend(loc="center right", fancybox=True, framealpha=0.75)

    #        plt_ax.legend(loc='best', fancybox=True, framealpha=0.75)

    def save(self, path="./", name=None):
        """Save a pumping test to file.

        This writes the variable to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``"./"``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Test_"+name``. Default: ``None``

        Notes
        -----
        The file will get the suffix ``".tst"``.
        """
        # ensure that 'path' is a string [ needed ?! ]
        # path = _formstr(path)
        # ensure that 'path' ends with a '/' if it's not empty
        if path != "" and path[-1] != "/":
            path += "/"
        if path == "":
            path = "./"
        # create the path if not existing
        if not os.path.exists(path):
            os.makedirs(path)
        # create a standard name if None is given
        if name is None:
            name = "Test_" + self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".tst":
            name += ".tst"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmptest/"
        patht = path + tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht + "info.csv", "w") as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Testtype", "PumpingTest"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            writer.writerow(["timeframe", self.timeframe])
            writer.writerow(["pumpingwell", self.pumpingwell])
            # define names for the variable-files
            pumprname = name[:-4] + "_PprVar.var"
            aquidname = name[:-4] + "_AqdVar.var"
            aquirname = name[:-4] + "_AqrVar.var"
            # save variable-files
            writer.writerow(["pumpingrate", pumprname])
            self._pumpingrate.save(patht, pumprname)
            writer.writerow(["aquiferdepth", aquidname])
            self._aquiferdepth.save(patht, aquidname)
            writer.writerow(["aquiferradius", aquirname])
            self._aquiferradius.save(patht, aquirname)
            okeys = tuple(self.observations.keys())
            writer.writerow(["Observations", len(okeys)])
            obsname = {}
            for k in okeys:
                obsname[k] = name[:-4] + "_" + k + "_Obs.obs"
                writer.writerow([k, obsname[k]])
                self.observations[k].save(patht, obsname[k])
        # compress everything to one zip-file
        with zipfile.ZipFile(path + name, "w") as zfile:
            zfile.write(patht + "info.csv", "info.csv")
            zfile.write(patht + pumprname, pumprname)
            zfile.write(patht + aquidname, aquidname)
            zfile.write(patht + aquirname, aquirname)
            for k in okeys:
                zfile.write(patht + obsname[k], obsname[k])
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


def load_test(tstfile):
    """Load a test from file.

    This reads a test from a csv file.

    Parameters
    ----------
    tstfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(tstfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            row = _nextr(data)
            if row[0] != "Testtype":
                raise Exception
            if row[1] == "PumpingTest":
                routine = _load_pumping_test
            else:
                raise Exception
    except Exception:
        raise Exception("loadTest: loading the test " + "was not possible")

    return routine(tstfile)


def _load_pumping_test(tstfile):
    """Load a pumping test from file.

    This reads a pumping test from a csv file.

    Parameters
    ----------
    tstfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(tstfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            if next(data)[1] != "PumpingTest":
                raise Exception
            name = next(data)[1]
            description = next(data)[1]
            timeframe = next(data)[1]
            pumpingwell = next(data)[1]
            pumpingrate = load_var(TxtIO(zfile.open(next(data)[1])))
            aquiferdepth = load_var(TxtIO(zfile.open(next(data)[1])))
            aquiferradius = load_var(TxtIO(zfile.open(next(data)[1])))
            obscnt = np.int(next(data)[1])
            observations = {}
            for __ in range(obscnt):
                row = _nextr(data)
                observations[row[0]] = load_obs(BytIO(zfile.read(row[1])))

        pumpingtest = PumpingTest(
            name,
            pumpingwell,
            pumpingrate,
            observations,
            aquiferdepth,
            aquiferradius,
            description,
            timeframe,
        )
    except Exception:
        raise Exception(
            "loadPumpingTest: loading the pumpingtest " + "was not possible"
        )
    return pumpingtest
