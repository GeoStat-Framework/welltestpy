# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:40:35 2016

@author: Sebastian Mueller
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import csv
import shutil
import zipfile
from io import TextIOWrapper as TxtIO

import numpy as np

# from ..tools._extimport import *

from welltestpy.tools._extimport import BytIO
from welltestpy.data.varlib import (Variable, Observation, loadVar, loadObs)
from .varlib import (_nextr, _formstr, _formname)


class Test(object):
    def __init__(self, name, description="no description", timeframe=None):
        self.name = _formstr(name)
        self.description = str(description)
        self.timeframe = str(timeframe)
        self._testtype = "Test"

    def __repr__(self):
        return self.testtype+" '"+self.name+"', Info: "+self.description

    @property
    def testtype(self):
        return self._testtype


class PumpingTest(Test):
    def __init__(self, name, pumpingwell, pumpingrate, observations=None,
                 aquiferdepth=1.0, aquiferradius=np.inf,
                 description="Pumpingtest", timeframe=None):
        super(PumpingTest, self).__init__(name, description, timeframe)

        self._testtype = "PumpingTest"

        self.pumpingwell = str(pumpingwell)

        if isinstance(pumpingrate, Variable):
            self._pumpingrate = dcopy(pumpingrate)
        else:
            self._pumpingrate = Variable("pumpingrate", pumpingrate, "Q",
                                         "m^3/s",
                                         "Pumpingrate at test '"+self.name+"'")
        if not self._pumpingrate.scalar:
            raise ValueError("PumpingTest: 'pumpingrate' needs to be scalar")

        if isinstance(aquiferdepth, Variable):
            self._aquiferdepth = dcopy(aquiferdepth)
        else:
            self._aquiferdepth = Variable("aquiferdepth", aquiferdepth,
                                          "L_a", "m",
                                          "mean aquiferdepth for test '" +
                                          str(name)+"'")
        if not self._aquiferdepth.scalar:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be scalar")
        if self.aquiferdepth <= 0.0:
            raise ValueError("PumpingTest: 'aquiferdepth' needs to be positiv")

        if isinstance(aquiferradius, Variable):
            self._aquiferradius = dcopy(aquiferradius)
        else:
            self._aquiferradius = Variable("aquiferradius", aquiferradius,
                                           "R", "m",
                                           "mean aquiferradius for test '" +
                                           str(name)+"'")
        if not self._aquiferradius.scalar:
            raise ValueError("PumpingTest: 'aquiferradius' needs to be scalar")
        if self.aquiferradius <= 0.0:
            raise ValueError("PumpingTest: 'aquiferradius' " +
                             "needs to be positiv")

        if observations is None:
            self.observations = {}
        else:
            self.observations = observations

    @property
    def wells(self):
        tmp = list(self.__observations.keys())
        tmp.append(self.pumpingwell)
        return tuple(set(tmp))

    @property
    def pumpingrate(self):
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
            raise ValueError("PumpingTest: 'aquiferradius' " +
                             "needs to be positiv")

    @property
    def observations(self):
        return self.__observations

    @observations.setter
    def observations(self, obs):
        if obs is not None:
            if isinstance(obs, dict):
                for k in obs.keys():
                    if not isinstance(obs[k], Observation):
                        raise ValueError("PumpingTest: some 'observations' " +
                                         "are not of type Observation")
                self.__observations = dcopy(obs)
            else:
                raise ValueError("PumpingTest: 'observations' should" +
                                 " be given as dictonary")
        else:
            self.__observations = {}

    def addobservations(self, obs):
        if isinstance(obs, dict):
            for k in obs:
                if not isinstance(obs[k], Observation):
                    raise ValueError("PumpingTest_addobservations: some " +
                                     "'observations' are not " +
                                     "of type Observation")
                if k in self.observations:
                    raise ValueError("PumpingTest_addobservations: some " +
                                     "'observations' are already present")
            for k in obs:
                self.__observations[k] = dcopy(obs[k])
        else:
            raise ValueError("PumpingTest_addobservations: 'observations' " +
                             "should be given as dictonary with well as key")

    def delobservations(self, obs):
        if isinstance(obs, (list, tuple)):
            for k in obs:
                if k in self.observations:
                    del self.__observations[k]
        else:
            if obs in self.observations:
                del self.__observations[obs]

    def _addplot(self, ax, wells, exclude=None):
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

            ax.plot(self.observations[k].value[0],
                    displace,
                    linewidth=2,
                    label=self.observations[k].name+" r={:1.2f}".format(dist))
            ax.set_xlabel(self.observations[k].labels[0])
            ax.set_ylabel(self.observations[k].labels[1])

        ax.set_title(repr(self))
        ax.legend(loc='center right', fancybox=True, framealpha=0.75)
#        ax.legend(loc='best', fancybox=True, framealpha=0.75)

    def save(self, path="./", name=None):
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
            name = "Test_"+self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".tst":
            name += ".tst"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmptest/"
        patht = path+tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht+"info.csv", 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Testtype", "PumpingTest"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            writer.writerow(["timeframe", self.timeframe])
            writer.writerow(["pumpingwell", self.pumpingwell])
            # define names for the variable-files
            pumprname = name[:-4]+"_PprVar.var"
            aquidname = name[:-4]+"_AqdVar.var"
            aquirname = name[:-4]+"_AqrVar.var"
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
                obsname[k] = name[:-4]+"_"+k+"_Obs.obs"
                writer.writerow([k, obsname[k]])
                self.observations[k].save(patht, obsname[k])
        # compress everything to one zip-file
        with zipfile.ZipFile(path+name, "w") as zfile:
            zfile.write(patht+"info.csv", "info.csv")
            zfile.write(patht+pumprname, pumprname)
            zfile.write(patht+aquidname, aquidname)
            zfile.write(patht+aquirname, aquirname)
            for k in okeys:
                zfile.write(patht+obsname[k], obsname[k])
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


def loadTest(tstfile):
    try:
        with zipfile.ZipFile(tstfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            row = _nextr(data)
            if row[0] != "Testtype":
                raise Exception
            if row[1] == "PumpingTest":
                routine = _loadPumpingTest
            else:
                raise Exception
    except:
        raise Exception("loadTest: loading the test " +
                        "was not possible")

    return routine(tstfile)


def _loadPumpingTest(tstfile):
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
            pumpingrate = loadVar(TxtIO(zfile.open(next(data)[1])))
            aquiferdepth = loadVar(TxtIO(zfile.open(next(data)[1])))
            aquiferradius = loadVar(TxtIO(zfile.open(next(data)[1])))
            obscnt = np.int(next(data)[1])
            observations = {}
            for __ in range(obscnt):
                row = _nextr(data)
                observations[row[0]] = loadObs(BytIO(zfile.read(row[1])))

        pumpingtest = PumpingTest(name, pumpingwell, pumpingrate, observations,
                                  aquiferdepth, aquiferradius,
                                  description, timeframe)
    except:
        raise Exception("loadPumpingTest: loading the pumpingtest " +
                        "was not possible")
    return pumpingtest
