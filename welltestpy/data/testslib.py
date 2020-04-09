# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing flow datastructures for tests on a fieldsite.

.. currentmodule:: welltestpy.data.testslib

The following classes and functions are provided

.. autosummary::
   Test
   PumpingTest
"""
from copy import deepcopy as dcopy

import numpy as np

from ..tools import plotter
from . import varlib, data_io
from ..process import processlib

__all__ = ["Test", "PumpingTest"]


class Test:
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
        self.name = data_io._formstr(name)
        self.description = str(description)
        self.timeframe = str(timeframe)
        self._testtype = "Test"

    def __repr__(self):
        """Representation."""
        return (
            self.testtype + " '" + self.name + "', Info: " + self.description
        )

    @property
    def testtype(self):
        """:class:`str`: String containing the test type."""
        return self._testtype

    def plot(self, wells, exclude=None, fig=None, ax=None, **kwargs):
        """Generate a plot of the pumping test.

        This will plot the test on the given figure axes.

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
        # update ax (or create it if None) and return it
        return ax


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
        super().__init__(name, description, timeframe)

        self._pumpingrate = None
        self._aquiferdepth = None
        self._aquiferradius = None
        self.__observations = {}
        self._testtype = "PumpingTest"

        self.pumpingwell = str(pumpingwell)
        self.pumpingrate = pumpingrate
        self.aquiferdepth = aquiferdepth
        self.aquiferradius = aquiferradius
        self.observations = observations

    def make_steady(self, time="latest"):
        """
        Convert the pumping test to a steady state test.

        Parameters
        ----------
        time : :class:`str` or :class:`float`, optional
            Selected time point for steady state.
            If "latest", the latest common time point is used.
            If None, it takes the last observation per well.
            If float, it will be interpolated.
            Default: "latest"
        """
        if time == "latest":
            tout = np.inf
            for obs in self.observations:
                if self.observations[obs].state == "transient":
                    tout = min(tout, np.max(self.observations[obs].time))
        elif time is None:
            tout = 0.0
            for obs in self.observations:
                if self.observations[obs].state == "transient":
                    tout = max(tout, np.max(self.observations[obs].time))
        else:
            tout = float(time)
        for obs in self.observations:
            if self.observations[obs].state == "transient":
                processlib.filterdrawdown(self.observations[obs], tout=tout)
                del self.observations[obs].time
        if (
            isinstance(self._pumpingrate, varlib.Observation)
            and self._pumpingrate.state == "transient"
        ):
            processlib.filterdrawdown(self._pumpingrate, tout=tout)
            del self._pumpingrate.time

    def state(self, wells=None):
        """
        Get the state of observation.

        Either None, "steady", "transient" or "mixed".

        Parameters
        ----------
        wells : :class:`list`, optional
            List of wells, to check the observation state at. Default: all
        """
        wells = self.observationwells if wells is None else list(wells)
        states = set()
        for obs in wells:
            if obs not in self.observations:
                raise ValueError(obs + " is an unknown well.")
            states.add(self.observations[obs].state)
        if len(states) == 1:
            return states.pop()
        if len(states) > 1:
            return "mixed"
        return None

    @property
    def wells(self):
        """:class:`tuple` of :class:`str`: all well names."""
        tmp = list(self.__observations.keys())
        tmp.append(self.pumpingwell)
        wells = list(set(tmp))
        wells.sort()
        return wells

    @property
    def observationwells(self):
        """:class:`tuple` of :class:`str`: all well names."""
        tmp = list(self.__observations.keys())
        wells = list(set(tmp))
        wells.sort()
        return wells

    @property
    def constant_rate(self):
        """:class:`bool`: state if this is a constant rate test."""
        return np.isscalar(self.rate)

    @property
    def rate(self):
        """:class:`float`: pumping rate at the pumping well."""
        return self._pumpingrate.value

    @property
    def pumpingrate(self):
        """:class:`float`: pumping rate variable at the pumping well."""
        return self._pumpingrate

    @pumpingrate.setter
    def pumpingrate(self, pumpingrate):
        if isinstance(pumpingrate, (varlib.Variable, varlib.Observation)):
            self._pumpingrate = dcopy(pumpingrate)
        elif self._pumpingrate is None:
            self._pumpingrate = varlib.Variable(
                "pumpingrate",
                pumpingrate,
                "Q",
                "m^3/s",
                "Pumpingrate at test '" + self.name + "'",
            )
        else:
            self._pumpingrate(pumpingrate)
        if (
            isinstance(self._pumpingrate, varlib.Variable)
            and not self.constant_rate
        ):
            raise ValueError("PumpingTest: 'pumpingrate' not scalar")
        if (
            isinstance(self._pumpingrate, varlib.Observation)
            and self._pumpingrate.state == "steady"
            and not self.constant_rate
        ):
            raise ValueError("PumpingTest: 'pumpingrate' not scalar")

    @property
    def depth(self):
        """:class:`float`: aquifer depth at the field site."""
        return self._aquiferdepth.value

    @property
    def aquiferdepth(self):
        """:class:`float`: aquifer depth at the field site."""
        return self._aquiferdepth

    @aquiferdepth.setter
    def aquiferdepth(self, aquiferdepth):
        if isinstance(aquiferdepth, varlib.Variable):
            self._aquiferdepth = dcopy(aquiferdepth)
        elif self._aquiferdepth is None:
            self._aquiferdepth = varlib.Variable(
                "aquiferdepth",
                aquiferdepth,
                "L_a",
                "m",
                "mean aquiferdepth for test '" + str(self.name) + "'",
            )
        else:
            self._aquiferdepth(aquiferdepth)
        if not self._aquiferdepth.scalar:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be scalar")
        if self.depth <= 0.0:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be positiv")

    @property
    def radius(self):
        """:class:`float`: aquifer radius at the field site."""
        return self._aquiferradius.value

    @property
    def aquiferradius(self):
        """:class:`float`: aquifer radius at the field site."""
        return self._aquiferradius

    @aquiferradius.setter
    def aquiferradius(self, aquiferradius):
        if isinstance(aquiferradius, varlib.Variable):
            self._aquiferradius = dcopy(aquiferradius)
        elif self._aquiferradius is None:
            self._aquiferradius = varlib.Variable(
                "aquiferradius",
                aquiferradius,
                "R",
                "m",
                "mean aquiferradius for test '" + str(self.name) + "'",
            )
        else:
            self._aquiferradius(aquiferradius)
        if not self._aquiferradius.scalar:
            raise ValueError("PumpingTest: 'aquiferradius' needs to be scalar")
        if self.radius <= 0.0:
            raise ValueError(
                "PumpingTest: 'aquiferradius' " + "needs to be positiv"
            )

    @property
    def observations(self):
        """:class:`dict`: observations made at the field site."""
        return self.__observations

    @observations.setter
    def observations(self, obs):
        self.__observations = {}
        if obs is not None:
            self.add_observations(obs)

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
        obs = varlib.StdyHeadObs(well, observation, description)
        self.add_observations(obs)

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
        obs = varlib.DrawdownObs(well, observation, time, description)
        self.add_observations(obs)

    def add_observations(self, obs):
        """Add some specified observations.

        Parameters
        ----------
        obs : :class:`dict`, :class:`list`, :class:`Observation`
            Observations to be added.
        """
        if isinstance(obs, dict):
            for k in obs:
                if not isinstance(obs[k], varlib.Observation):
                    raise ValueError(
                        "PumpingTest_add_observations: some "
                        + "'observations' are not "
                        + "of type Observation"
                    )
                if k in self.observations:
                    raise ValueError(
                        "PumpingTest_add_observations: some "
                        + "'observations' are already present"
                    )
            for k in obs:
                self.__observations[k] = dcopy(obs[k])
        elif isinstance(obs, varlib.Observation):
            if obs in self.observations:
                raise ValueError(
                    "PumpingTest_add_observations: "
                    + "'observation' are already present"
                )
            self.__observations[obs.name] = dcopy(obs)
        else:
            try:
                iter(obs)
            except TypeError:
                raise ValueError(
                    "PumpingTest_add_observations: 'obs' can't be read."
                )
            else:
                for ob in obs:
                    if not isinstance(ob, varlib.Observation):
                        raise ValueError(
                            "PumpingTest_add_observations: some "
                            + "'observations' are not "
                            + "of type Observation"
                        )
                    self.__observations[ob.name] = dcopy(ob)

    def del_observations(self, obs):
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

    def plot(self, wells, exclude=None, fig=None, ax=None, **kwargs):
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
        return plotter.plot_pump_test(
            pump_test=self,
            wells=wells,
            exclude=exclude,
            fig=fig,
            ax=ax,
            **kwargs
        )

    def save(self, path="", name=None):
        """Save a pumping test to file.

        This writes the variable to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``""``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Test_"+name``. Default: ``None``

        Notes
        -----
        The file will get the suffix ``".tst"``.
        """
        return data_io.save_pumping_test(self, path, name)
