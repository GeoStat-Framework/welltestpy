"""
welltestpy subpackage providing flow datastructures for variables.

.. currentmodule:: welltestpy.data.varlib

The following classes and functions are provided

.. autosummary::
   Variable
   TimeVar
   HeadVar
   TemporalVar
   CoordinatesVar
   Observation
   StdyObs
   DrawdownObs
   StdyHeadObs
   Well
   load_var
   load_obs
   load_well
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import csv
import shutil
import zipfile
from io import TextIOWrapper as TxtIO

import numpy as np

from welltestpy.tools.plotter import Editor

__all__ = [
    "Variable",
    "TimeVar",
    "HeadVar",
    "TemporalVar",
    "CoordinatesVar",
    "Observation",
    "StdyObs",
    "DrawdownObs",
    "StdyHeadObs",
    "Well",
    "load_var",
    "load_obs",
    "load_well",
]


class Variable(object):
    """Class for a variable.

    This is a class for a physical variable which is either a scalar or an
    array.

    It has a name, a value, a symbol, a unit and a descrition string.

    Attributes
    ----------
    name : :class:`str`
        Name of the Variable.
    symbole : :class:`str`
        Name of the Variable.
    units : :class:`str`
        Units of the Variable.
    description : :class:`str`
        Description of the Variable.
    """
    def __init__(self, name, value,
                 symbol="x", units="-", description="no description"):
        """Variable initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Variable.
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Value of the Variable.
        symbole : :class:`str`, optional
            Name of the Variable. Default: ``"x"``
        units : :class:`str`, optional
            Units of the Variable. Default: ``"-"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"no description"``
        """
        self.name = _formstr(name)
        self.__value = None
        self.value = value
        self.symbol = str(symbol)
        self.units = str(units)
        self.description = str(description)

    def __call__(self, value=None):
        """Call a variable.

        Here you can set a new value or you can get the value of the variable.

        Parameters
        ----------
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`,\
        optional
            Value of the Variable. Default: ``None``

        Returns
        -------
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Value of the Variable.
        """
        if value is not None:
            self.value = value
        return self.value

    @property
    def info(self):
        """:class:`str`: Info about the Variable."""
        info = ""
        info += " Variable-name: "+str(self.name)+"\n"
        info += " -Value:        "+str(self.value)+"\n"
        info += " -Symbol:       "+str(self.symbol)+"\n"
        info += " -Units:        "+str(self.units)+"\n"
        info += " -Description:  "+str(self.description)+"\n"
#        print(" Variable-name: "+str(self.name))
#        print(" -Value:        "+str(self.value))
#        print(" -Symbol:       "+str(self.symbol))
#        print(" -Units:        "+str(self.units))
#        print(" -Description:  "+str(self.description))
        return info

    @property
    def scalar(self):
        """:class:`bool`: State if the variable is of scalar type."""
        return np.isscalar(self.__value)

    @property
    def label(self):
        """:class:`str`: String containing: ``symbol in units``."""
        return self.symbol+" in "+self.units

    @property
    def value(self):
        """:class:`int` or :class:`float` or :class:`numpy.ndarray`:
        Value of the Variable."""
        return self.__value

    @value.setter
    def value(self, value):
        if np.asanyarray(value).dtype == np.float:
            if np.ndim(np.squeeze(value)) == 0:
                self.__value = np.float(np.squeeze(value))
            else:
                self.__value = np.squeeze(np.asanyarray(value, dtype=np.float))
        elif np.asanyarray(value).dtype == np.int:
            if np.ndim(np.squeeze(value)) == 0:
                self.__value = np.int(np.squeeze(value))
            else:
                self.__value = np.squeeze(np.asanyarray(value, dtype=np.int))
        else:
            raise ValueError("Variable: 'value' is neither integer nor float")

    def __repr__(self):
        return str(self.name)+" "+self.symbol+": " + \
               str(self.value)+" "+self.units

    def __str__(self):
        return str(self.name)+" "+self.label

    def save(self, path="./", name=None):
        """Save a variable to file.

        This writes the variable to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``"./"``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Var_"+name``. Default: ``None``

        Note
        ----
        The file will get the suffix ``".var"``.
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
            name = "Var_"+self.name
        # ensure the name ends with '.var'
        if name[-4:] != ".var":
            name += ".var"
        name = _formname(name)
        # write the csv-file
        with open(path+name, 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Variable"])
            writer.writerow(["name", self.name])
            writer.writerow(["symbol", self.symbol])
            writer.writerow(["units", self.units])
            writer.writerow(["description", self.description])
            if np.asanyarray(self.__value).dtype == np.int:
                writer.writerow(["integer"])
            else:
                writer.writerow(["float"])
            if self.scalar:
                writer.writerow(["scalar"])
                writer.writerow(["value", self.value])
            else:
                writer.writerow(["shape"]+list(np.shape(self.value)))
                tmpvalue = np.reshape(self.value, -1)
                writer.writerow(["values", len(tmpvalue)])
                for val in tmpvalue:
                    writer.writerow([val])


class TimeVar(Variable):
    """Variable class special for time series.

    Note
    ----
    Here the variable should be at most 1 dimensional and the name is fix set
    to ``"time"``.
    """
    def __init__(self, value,
                 symbol="t", units="s", description="time given in seconds"):
        """Time variable initialisation.

        Parameters
        ----------
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Value of the Variable.
        symbole : :class:`str`, optional
            Name of the Variable. Default: ``"t"``
        units : :class:`str`, optional
            Units of the Variable. Default: ``"s"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"time given in seconds"``
        """
        super(TimeVar, self).__init__("time", value,
                                      symbol, units, description)
        if np.ndim(self.value) > 1:
            raise ValueError("TimeVar: 'time' should have " +
                             "at most one dimension")


class HeadVar(Variable):
    """Variable class special for groundwater head.

    Note
    ----
    Here the variable name is fix set to ``"head"``.
    """
    def __init__(self, value,
                 symbol="h", units="m", description="head given in meters"):
        """Head variable initialisation.

        Parameters
        ----------
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Value of the Variable.
        symbole : :class:`str`, optional
            Name of the Variable. Default: ``"h"``
        units : :class:`str`, optional
            Units of the Variable. Default: ``"m"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"head given in meters"``
        """
        super(HeadVar, self).__init__("head", value,
                                      symbol, units, description)


class TemporalVar(Variable):
    """Variable class for a temporal variable.
    """
    def __init__(self, value=0.0):
        """Temporal variable initialisation.

        Parameters
        ----------
        value : :class:`int` or :class:`float` or :class:`numpy.ndarray`,
        optional
            Value of the Variable. Default: ``0.0``
        """
        super(TemporalVar, self).__init__("temporal", value,
                                          description="temporal variable")


class CoordinatesVar(Variable):
    """Variable class special for coordinates.

    Note
    ----
    Here the variable name is fix set to ``"coordinates"``.

    ``lat`` and ``lon`` should have the same shape.
    """
    def __init__(self, lat, lon,
                 symbol="[Lat,Lon]", units="[deg,deg]",
                 description="Coordinates given in " +
                 "degree-North and degree-East"):
        """Coordinate variable initialisation.

        Parameters
        ----------
        lat : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Lateral values of the coordinates.
        lon : :class:`int` or :class:`float` or :class:`numpy.ndarray`
            Longitutional values of the coordinates.
        symbole : :class:`str`, optional
            Name of the Variable. Default: ``"[Lat,Lon]"``
        units : :class:`str`, optional
            Units of the Variable. Default: ``"[deg,deg]"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Coordinates given in
            degree-North and degree-East"``
        """
        ilat = np.array(np.squeeze(lat), ndmin=1)
        ilon = np.array(np.squeeze(lon), ndmin=1)

        if (len(ilat.shape) != 1 or
                len(ilon.shape) != 1 or
                ilat.shape != ilon.shape):
            raise ValueError("CoordinatesVar: 'lat' and 'lon' should have" +
                             "same quantity and should be given as lists")

        value = np.array([ilat, ilon]).T

        super(CoordinatesVar, self).__init__("coordinates", value,
                                             symbol, units, description)


class Observation(object):
    """Class for a observation.

    This is a class for time-dependent observations.

    It has a name and a descrition.

    Attributes
    ----------
    name : :class:`str`
        Name of the Observation.
    description : :class:`str`
        Description of the Variable.
    """
    def __init__(self, name, time, observation,
                 description="Observation"):
        """Observation initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Variable.
        time : :class:`Variable`
            Value of the Variable.
        observation : :class:`Variable`
            Name of the Variable. Default: ``"x"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Observation"``
        """
        self.__it = None
        self.__itfinished = None
        self.name = _formstr(name)
        self.description = str(description)

        if isinstance(observation, Variable):
            self._observation = dcopy(observation)
        else:
            raise ValueError("Observation: " +
                             "'observation' must be instance of 'variable'")

        if time is not None:
            if isinstance(time, Variable):
                self._time = dcopy(time)
            else:
                self._time = TimeVar(time)

            self.__state = "transient"
        else:
            self.__state = "steady"
        self._checkshape()

    def __call__(self, in1=None, in2=None, time=None, observation=None):
        """Call a variable.

        Here you can set a new value or you can get the value of the variable.

        Parameters
        ----------
        in1 : :class:`int` or :class:`float` or :class:`numpy.ndarray` or \
        :class:`Variable`, optional
            New Value for time (if transient) / observation (if steady).
            Default: ``"None"``
        in2 : :class:`int` or :class:`float` or :class:`numpy.ndarray` or \
        :class:`Variable`, optional
            New Value for observation (if transient).
            Default: ``"None"``
        time : :class:`int` or :class:`float` or :class:`numpy.ndarray` or \
        :class:`Variable`, optional
            New Value for time.
            Default: ``"None"``
        observation : :class:`int` or :class:`float` or :class:`numpy.ndarray`\
        or :class:`Variable`, optional
            New Value for observation.
            Default: ``"None"``

        Returns
        -------
        [:class:`tuple` of] :class:`int` or :class:`float`\
        or :class:`numpy.ndarray`
            ``(time, observation)`` or ``observation``.
        """
        # in1 and in2 are for non-keyword call
        if self.state == "transient":
            if time is None:
                time = in1
            if observation is None:
                observation = in2
            tmp1 = dcopy(self._time)
            tmp2 = dcopy(self._observation)
            self._settime(time)
            self._setobservation(observation)
            if not self._checkshape():
                self._settime(tmp1)
                self._setobservation(tmp2)
                raise ValueError("Observation: " +
                                 "'observation' and 'time' have a " +
                                 "shape-missmatch")
            return self.time, self.observation
        else:
            if observation is None:
                observation = in1
            self._setobservation(observation)
            return self.observation

    def __repr__(self):
        return "Observation '"+str(self.name)+"' "+str(self.label)

    def __str__(self):
        return self.__repr__()

    @property
    def labels(self):
        """[:class:`tuple` of] :class:`str`:
        String containing: ``symbol in units``."""
        if self.state == "transient":
            return self._time.label, self._observation.label
        return self._observation.label

    @property
    def label(self):
        """[:class:`tuple` of] :class:`str`:
        String containing: ``symbol in units``."""
        return self.labels

    @property
    def info(self):
        """Get informations about the observation.

        Here you can display informations about the observation.
        """
        info = ""
        info += "Observation-name: "+str(self.name)+"\n"
        info += " -Description:    "+str(self.description)+"\n"
        info += " -Kind:           "+str(self.kind)+"\n"
        info += " -State:          "+str(self.state)+"\n"
        if self.state == "transient":
            info += " --- "+"\n"
            info += self._time.info+"\n"
        info += " --- "+"\n"
        info += self._observation.info+"\n"
#        print("Observation-name: "+str(self.name))
#        print(" -Description:    "+str(self.description))
#        print(" -Kind:           "+str(self.kind))
#        print(" -State:          "+str(self.state))
#        if self.state == "transient":
#            print(" --- ")
#            self._time.info
#        print(" --- ")
#        self._observation.info
        return info

    @property
    def value(self):
        """[:class:`tuple` of] :class:`int` or :class:`float`
        or :class:`numpy.ndarray`:
        Value of the Observation."""
        if self.state == "transient":
            return self.time, self.observation
        return self.observation

    @property
    def state(self):
        """:class:`str`: String containing state of the observation.
        Either ``"steady"`` or ``"transient"``."""
        return self.__state

    @property
    def kind(self):
        """:class:`str`: name of the observation variable"""
        return self._observation.name

    @property
    def time(self):
        ''':class:`int` or :class:`float` or :class:`numpy.ndarray`:
        time values of the observation'''
        if self.state == "transient":
            return self._time.value
        return None

    @property
    def observation(self):
        ''':class:`int` or :class:`float` or :class:`numpy.ndarray`:
        observed values of the observation'''
        return self._observation.value

    @property
    def units(self):
        """[:class:`tuple` of] :class:`str`:
        String containing the units of the observation"""
        if self.state == "steady":
            return self._observation.units
        return self._time.units+","+self._observation.units

    @time.setter
    def time(self, time):
        if self.state == "steady":
            self.__state = "transient"
            self._settime(time)
            if not self._checkshape():
                del self.time
                raise ValueError("Observation: " +
                                 "'time' has a " +
                                 "shape-missmatch with 'observation'")
        else:
            tmp = dcopy(self._time)
            self._settime(time)
            if not self._checkshape():
                self._settime(tmp)
                raise ValueError("Observation: " +
                                 "'time' has a " +
                                 "shape-missmatch with 'observation'")

    @time.deleter
    def time(self):
        self.__state = "steady"
        del self._time

    @observation.setter
    def observation(self, observation):
        tmp = dcopy(self._observation)
        self._setobservation(observation)
        if not self._checkshape():
            self._setobservation(tmp)
            raise ValueError("Observation: " +
                             "'observation' has a " +
                             "shape-missmatch with 'time'")

    def reshape(self):
        '''Reshape obeservations to flat array.'''
        if self.state == "transient":
            tmp = len(np.shape(self.time))
            self._settime(np.reshape(self.time, -1))
            shp = np.shape(self.time)+np.shape(self.observation)[tmp:]
            self._setobservation(np.reshape(self.observation, shp))

    def _settime(self, time):
        if isinstance(time, Variable):
            self._time = dcopy(time)
        else:
            self._time(time)

    def _setobservation(self, observation):
        if isinstance(observation, Variable):
            self._observation = dcopy(observation)
        else:
            self._observation(observation)

    def _checkshape(self):
        if self.state == "transient":
            if np.shape(self.time) != \
               np.shape(self.observation)[:len(np.shape(self._time()))]:
                return False
        return True

    def __iter__(self):
        if self.state == "transient":
            self.__it = np.nditer(self.time, flags=['multi_index'])
        else:
            self.__itfinished = False
        return self

    def next(self):
        '''Iterate through observations'''
        if self.state == "transient":
            if self.__it.finished:
                raise StopIteration
            ret = (np.asscalar(self.__it[0]),
                   self.observation[self.__it.multi_index])
            self.__it.iternext()
        else:
            if self.__itfinished:
                raise StopIteration
            ret = self.observation
            self.__itfinished = True
        return ret

    # for python 2&3 compatibility overwrite "__next__" with "next"
    __next__ = next

    def edit(self):
        '''Edit the observed time-series with a graphical interface.'''
        if self.state == "transient" and len(np.shape(self.time)) == 1:
            Editor(self)

    def save(self, path="./", name=None):
        """Save an observation to file.

        This writes the observation to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``"./"``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Obs_"+name``. Default: ``None``

        Note
        ----
        The file will get the suffix ``".obs"``.
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
            name = "Obs_"+self.name
        # ensure the name ends with '.obs'
        if name[-4:] != ".obs":
            name += ".obs"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmpobserv/"
        patht = path+tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht+"info.csv", 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Observation"])
            writer.writerow(["name", self.name])
            writer.writerow(["state", self.state])
            writer.writerow(["description", self.description])
            if self.state == "steady":
                obsname = name[:-4]+"_ObsVar.var"
                writer.writerow(["observation", obsname])
                self._observation.save(patht, obsname)
            else:
                timname = name[:-4]+"_TimVar.var"
                obsname = name[:-4]+"_ObsVar.var"
                writer.writerow(["time", timname])
                writer.writerow(["observation", obsname])
                self._time.save(patht, timname)
                self._observation.save(patht, obsname)
        # compress everything to one zip-file
        with zipfile.ZipFile(path+name, "w") as zfile:
            # zfile.write(patht+name[:-4]+".csv", name[:-4]+".csv")
            zfile.write(patht+"info.csv", "info.csv")
            if self.state == "transient":
                zfile.write(patht+timname, timname)
            zfile.write(patht+obsname, obsname)
        shutil.rmtree(patht, ignore_errors=True)


class StdyObs(Observation):
    """Observation class special for steady observations.
    """
    def __init__(self, name, observation,
                 description="Steady observation"):
        """Steady observation initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Variable.
        observation : :class:`Variable`
            Name of the Variable. Default: ``"x"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Steady observation"``
        """
        super(StdyObs, self).__init__(name, None, observation, description)

    def _settime(self, time):
        '''For steady observations, this raises a ``ValueError``'''
        raise ValueError("Observation: " +
                         "'time' not allowed in steady-state")


class DrawdownObs(Observation):
    """Observation class special for drawdown observations.
    """
    def __init__(self, name, time, observation,
                 description="Drawdown observation"):
        """Steady observation initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Variable.
        observation : :class:`Variable`
            Name of the Variable. Default: ``"x"``
        observation : :class:`Variable`
            Name of the Variable. Default: ``"x"``
        description : :class:`str`, optional
            Description of the Variable. Default: ``"Steady observation"``
        """
        if not isinstance(time, Variable):
            time = TimeVar(time)
        if not isinstance(observation, Variable):
            observation = HeadVar(observation)
        super(DrawdownObs, self).__init__(name, time, observation, description)


class StdyHeadObs(Observation):
    """Observation class special for steady drawdown observations.
    """
    def __init__(self, name, observation,
                 description="Steady State Drawdown observation"):
        if not isinstance(observation, Variable):
            observation = HeadVar(observation)
        super(StdyHeadObs, self).__init__(name, None, observation, description)

    def _settime(self, time):
        '''For steady observations, this raises a ``ValueError``'''
        raise ValueError("Observation: " +
                         "'time' not allowed in steady-state")


class Well(object):
    """Class for a pumping-/observation-well.

    This is a class for a well within a aquifer-testing campaign.

    It has a name, a radius, coordinates and a depth.

    Attributes
    ----------
    name : :class:`str`
        Name of the well.

    Note
    ----
    You can calculate the distance between two wells ``w1`` and ``w2`` by
    simply calculating the difference ``w1 - w2``.
    """
    def __init__(self, name, radius, coordinates, welldepth=1.0,
                 aquiferdepth=None):
        """Well initialisation.

        Parameters
        ----------
        name : :class:`str`
            Name of the Variable.
        radius : :class:`Variable` or :class:`float`
            Value of the Variable.
        coordinates : :class:`Variable` or :class:`numpy.ndarray`
            Value of the Variable.
        welldepth : :class:`Variable` or :class:`float`, optional
            Depth of the well. Default: 1.0
        aquiferdepth : :class:`Variable` or :class:`float`, optional
            Depth of the aquifer at the well. Default: ``"None"``
        """
        self.name = _formstr(name)

        if isinstance(radius, Variable):
            self._radius = dcopy(radius)
        else:
            self._radius = Variable("radius", radius, "r", "m",
                                    "Inner radius of well '"+str(name)+"'")
        if not self._radius.scalar:
            raise ValueError("Well: 'radius' needs to be scalar")
        if self.radius <= 0.0:
            raise ValueError("Well: 'radius' needs to be positiv")

        if isinstance(coordinates, Variable):
            self._coordinates = dcopy(coordinates)
        else:
            self._coordinates = Variable("coordinates", coordinates, "XY", "m",
                                         "coordinates of well '"+str(name)+"'")
        if np.shape(self.coordinates) != (2,) and\
           not np.isscalar(self.coordinates):
            raise ValueError("Well: 'coordinates' should be given as " +
                             "[x,y] values or one single distance value")

        if isinstance(welldepth, Variable):
            self._welldepth = dcopy(welldepth)
        else:
            self._welldepth = Variable("welldepth", welldepth, "L_w", "m",
                                       "depth of well '"+str(name)+"'")
        if not self._welldepth.scalar:
            raise ValueError("Well: 'welldepth' needs to be scalar")
        if self.welldepth <= 0.0:
            raise ValueError("Well: 'welldepth' needs to be positiv")

        if isinstance(aquiferdepth, Variable):
            self._aquiferdepth = dcopy(aquiferdepth)
        else:
            if aquiferdepth is None:
                self._aquiferdepth = Variable("aquiferdepth", welldepth,
                                              "L_a", "m",
                                              "aquiferdepth at well '" +
                                              str(name)+"'")
            else:
                self._aquiferdepth = Variable("aquiferdepth", aquiferdepth,
                                              "L_a", "m",
                                              "aquiferdepth at well '" +
                                              str(name)+"'")
        if not self._aquiferdepth.scalar:
            raise ValueError("Well: 'aquiferdepth' needs to be scalar")
        if self.aquiferdepth <= 0.0:
            raise ValueError("Well: 'aquiferdepth' needs to be positiv")

    @property
    def info(self):
        """Get informations about the variable.

        Here you can display informations about the variable.
        """
        info = ""
        info += "----"+"\n"
        info += "Well-name: "+str(self.name)+"\n"
        info += "--"+"\n"
        info += self._radius.info+"\n"
        info += self._coordinates.info+"\n"
        info += self._welldepth.info+"\n"
        info += self._aquiferdepth.info+"\n"
        info += "----"+"\n"
#        print("----")
#        print("Well-name: "+str(self.name))
#        print("--")
#        self._radius.info
#        self._coordinates.info
#        self._welldepth.info
#        self._aquiferdepth.info
#        print("----")
        return info

    @property
    def radius(self):
        """:class:`float`: Radius of the well"""
        return self._radius.value

    @radius.setter
    def radius(self, radius):
        tmp = dcopy(self._radius)
        if isinstance(radius, Variable):
            self._radius = dcopy(radius)
        else:
            self._radius(radius)
        if not self._radius.scalar:
            self._radius = dcopy(tmp)
            raise ValueError("Well: 'radius' needs to be scalar")
        if self.radius <= 0.0:
            self._radius = dcopy(tmp)
            raise ValueError("Well: 'radius' needs to be positiv")

    @property
    def coordinates(self):
        """:class:`numpy.ndarray`: Coordinates of the well"""
        return self._coordinates.value

    @coordinates.setter
    def coordinates(self, coordinates):
        tmp = dcopy(self._coordinates)
        if isinstance(coordinates, Variable):
            self._coordinates = dcopy(coordinates)
        else:
            self._coordinates(coordinates)
        if np.shape(self.coordinates) != (2,) and\
           not np.isscalar(self.coordinates):
            self._coordinates = dcopy(tmp)
            raise ValueError("Well: 'coordinates' should be given as " +
                             "[x,y] values or one single distance value")

    @property
    def welldepth(self):
        """:class:`float`: Depth of the well"""
        return self._welldepth.value

    @welldepth.setter
    def welldepth(self, welldepth):
        tmp = dcopy(self._welldepth)
        if isinstance(welldepth, Variable):
            self._welldepth = dcopy(welldepth)
        else:
            self._welldepth(welldepth)
        if not self._welldepth.scalar:
            self._welldepth = dcopy(tmp)
            raise ValueError("Well: 'welldepth' needs to be scalar")
        if self.welldepth <= 0.0:
            self._welldepth = dcopy(tmp)
            raise ValueError("Well: 'welldepth' needs to be positiv")

    @property
    def aquiferdepth(self):
        """:class:`float`: Aquifer depth at the well"""
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
            raise ValueError("Well: 'aquiferdepth' needs to be scalar")
        if self.aquiferdepth <= 0.0:
            self._aquiferdepth = dcopy(tmp)
            raise ValueError("Well: 'aquiferdepth' needs to be positiv")

    def distance(self, well):
        """Calculate distance to the well.

        Parameters
        ----------
        well : :class:`Well` or :class:`tuple` of :class:`float`
            Coordinates to calculate the distance to or another well.
        """
        if isinstance(well, Well):
            return np.linalg.norm(self.coordinates - well.coordinates)
        else:
            try:
                return np.linalg.norm(self.coordinates - well)
            except ValueError:
                raise ValueError("Well: the distant-well needs to be an " +
                                 "instance of Well-class " +
                                 "or a tupel of x-y coordinates " +
                                 "or a single distance value " +
                                 "and of same coordinates-type.")

    def __repr__(self):
        return (str(self.name)+" r="+str(self.radius) +
                " at "+repr(self._coordinates))

    def __sub__(self, well):
        return self.distance(well)

    def __add__(self, well):
        return self.distance(well)

    def __and__(self, well):
        return self.distance(well)

    def __rsub__(self, well):
        return self.distance(well)

    def __radd__(self, well):
        return self.distance(well)

    def __rand__(self, well):
        return self.distance(well)

    def __abs__(self):
        return np.linalg.norm(self.coordinates)

    def save(self, path="./", name=None):
        """Save a well to file.

        This writes the variable to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``"./"``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Well_"+name``. Default: ``None``

        Note
        ----
        The file will get the suffix ``".wel"``.
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
            name = "Well_"+self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".wel":
            name += ".wel"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmpwell/"
        patht = path+tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht+"info.csv", 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Well"])
            writer.writerow(["name", self.name])
            # define names for the variable-files
            radiuname = name[:-4]+"_RadVar.var"
            coordname = name[:-4]+"_CooVar.var"
            welldname = name[:-4]+"_WedVar.var"
            aquifname = name[:-4]+"_AqdVar.var"
            # save variable-files
            writer.writerow(["radius", radiuname])
            self._radius.save(patht, radiuname)
            writer.writerow(["coordinates", coordname])
            self._coordinates.save(patht, coordname)
            writer.writerow(["welldepth", welldname])
            self._welldepth.save(patht, welldname)
            writer.writerow(["aquiferdepth", aquifname])
            self._aquiferdepth.save(patht, aquifname)
        # compress everything to one zip-file
        with zipfile.ZipFile(path+name, "w") as zfile:
            # zfile.write(patht+name[:-4]+".csv", name[:-4]+".csv")
            zfile.write(patht+"info.csv", "info.csv")
            zfile.write(patht+radiuname, radiuname)
            zfile.write(patht+coordname, coordname)
            zfile.write(patht+welldname, welldname)
            zfile.write(patht+aquifname, aquifname)
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


# Loading routines ###

def load_var(varfile):
    """Load a variable from file.

    This reads a variable from a csv file.

    Parameters
    ----------
    varfile : :class:`str`
        Path to the file
    """
    try:
        with open(varfile, "r") as vfile:
            data = csv.reader(vfile)
            if next(data)[0] != "Variable":
                raise Exception
            name = next(data)[1]
            symbol = next(data)[1]
            units = next(data)[1]
            description = next(data)[1]
            integer = (next(data)[0] == "integer")
            shapenfo = _nextr(data)
            if shapenfo[0] == "scalar":
                if integer:
                    value = np.int(next(data)[1])
                else:
                    value = np.float(next(data)[1])
            else:
                shape = tuple(np.array(shapenfo[1:], dtype=np.int))
                vcnt = np.int(next(data)[1])
                vlist = []
                for __ in range(vcnt):
                    vlist.append(next(data)[0])
                if integer:
                    value = np.array(vlist, dtype=np.int).reshape(shape)
                else:
                    value = np.array(vlist, dtype=np.float).reshape(shape)

        var = Variable(name, value, symbol, units, description)
    except Exception:
        try:
            data = csv.reader(varfile)
            if next(data)[0] != "Variable":
                raise Exception
            name = next(data)[1]
            symbol = next(data)[1]
            units = next(data)[1]
            description = next(data)[1]
            integer = (next(data)[0] == "integer")
            shapenfo = _nextr(data)
            if shapenfo[0] == "scalar":
                if integer:
                    value = np.int(next(data)[1])
                else:
                    value = np.float(next(data)[1])
            else:
                shape = tuple(np.array(shapenfo[1:], dtype=np.int))
                vcnt = np.int(next(data)[1])
                vlist = []
                for __ in range(vcnt):
                    vlist.append(next(data)[0])
                if integer:
                    value = np.array(vlist, dtype=np.int).reshape(shape)
                else:
                    value = np.array(vlist, dtype=np.float).reshape(shape)

            var = Variable(name, value, symbol, units, description)
        except Exception:
            raise Exception("loadVar: loading the variable was not possible")
    return var


def load_obs(obsfile):
    """Load an observation from file.

    This reads a observation from a csv file.

    Parameters
    ----------
    obsfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(obsfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            if next(data)[0] != "Observation":
                raise Exception
            name = next(data)[1]
            steady = (next(data)[1] == "steady")
            description = next(data)[1]
            if not steady:
                timef = next(data)[1]
            obsf = next(data)[1]

            if not steady:
                time = load_var(TxtIO(zfile.open(timef)))
            else:
                time = None

            obs = load_var(TxtIO(zfile.open(obsf)))

        observation = Observation(name, time, obs, description)
    except Exception:
        raise Exception("loadObs: loading the observation was not possible")
    return observation


def load_well(welfile):
    """Load a well from file.

    This reads a well from a csv file.

    Parameters
    ----------
    welfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(welfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            if next(data)[0] != "Well":
                raise Exception
            name = next(data)[1]
            radf = next(data)[1]
            coordf = next(data)[1]
            welldf = next(data)[1]
            aquidf = next(data)[1]

            rad = load_var(TxtIO(zfile.open(radf)))
            coord = load_var(TxtIO(zfile.open(coordf)))
            welld = load_var(TxtIO(zfile.open(welldf)))
            aquid = load_var(TxtIO(zfile.open(aquidf)))

        well = Well(name, rad, coord, welld, aquid)
    except Exception:
        raise Exception("loadWell: loading the well was not possible")
    return well


# TOOLS ###

def _formstr(string):
    # remove spaces, tabs, linebreaks and other separators
    return ''.join(str(string).split())


def _formname(string):
    # remove slashes
    string = ''.join(str(string).split("/"))
    # remove spaces, tabs, linebreaks and other separators
    return _formstr(string)


def _nextr(data):
    return tuple(filter(None, next(data)))
