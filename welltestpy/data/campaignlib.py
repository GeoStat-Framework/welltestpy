# -*- coding: utf-8 -*-
"""Welltestpy subpackage providing flow datastructures for field-campaigns.

.. currentmodule:: welltestpy.data.campaignlib

The following classes and functions are provided

.. autosummary::
   FieldSite
   Campaign
   load_fieldsite
   load_campaign
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
from welltestpy.tools.plotter import CampaignPlot, WellPlot
from welltestpy.data.varlib import (
    Variable,
    CoordinatesVar,
    load_var,
    Well,
    load_well,
    _nextr,
    _formstr,
    _formname,
)
from welltestpy.data.testslib import Test, load_test

__all__ = ["FieldSite", "Campaign", "load_fieldsite", "load_campaign"]


class FieldSite(object):
    """Class for a field site.

    This is a class for a field site.
    It has a name and a descrition.

    Parameters
    ----------
    name : :class:`str`
        Name of the field site.
    description : :class:`str`, optional
        Description of the field site.
        Default: ``"no description"``
    coordinates : :class:`Variable`, optional
        Coordinates of the field site (lat, lon).
        Default: ``None``
    """

    def __init__(self, name, description="Field site", coordinates=None):
        self.name = _formstr(name)
        self.description = str(description)
        self._coordinates = None
        self.coordinates = coordinates

    @property
    def info(self):
        """:class:`str`: Info about the field site."""
        info = ""
        info += "----" + "\n"
        info += "Field-site:   " + str(self.name) + "\n"
        info += "Description:  " + str(self.description) + "\n"
        info += "--" + "\n"
        if self._coordinates is not None:
            info += self._coordinates.info + "\n"
        info += "----" + "\n"
        #        print("----")
        #        print("Field-site:   "+str(self.name))
        #        print("Description:  "+str(self.description))
        #        print("--")
        #        if hasattr(self, '_coordinates'):
        #            self._coordinates.info
        #        print("----")
        return info

    @property
    def coordinates(self):
        """:class:`numpy.ndarray`: Coordinates of the field site."""
        if self._coordinates is not None:
            return self._coordinates.value
        return None

    @coordinates.setter
    def coordinates(self, coordinates):
        if coordinates is not None:
            if isinstance(coordinates, Variable):
                self._coordinates = dcopy(coordinates)
            else:
                self._coordinates = CoordinatesVar(
                    coordinates[0], coordinates[1]
                )
        else:
            self._coordinates = None

    def __repr__(self):
        """Representation."""
        return self.name

    def save(self, path="", name=None):
        """Save a field site to file.

        This writes the field site to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``""``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Field_"+name``. Default: ``None``

        Notes
        -----
        The file will get the suffix ``".fds"``.
        """
        path = os.path.normpath(path)
        # create the path if not existing
        if not os.path.exists(path):
            os.makedirs(path)
        # create a standard name if None is given
        if name is None:
            name = "Field_" + self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".fds":
            name += ".fds"
        name = _formname(name)
        # create temporal directory for the included files
        patht = os.path.join(path, ".tmpfield")
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(os.path.join(patht, "info.csv"), "w") as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Fieldsite"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            # define names for the variable-files
            if self._coordinates is not None:
                coordname = name[:-4] + "_CooVar.var"
                # save variable-files
                writer.writerow(["coordinates", coordname])
                self._coordinates.save(patht, coordname)
            else:
                writer.writerow(["coordinates", "None"])
        # compress everything to one zip-file
        with zipfile.ZipFile(os.path.join(path, name), "w") as zfile:
            zfile.write(os.path.join(patht, "info.csv"), "info.csv")
            if self._coordinates is not None:
                zfile.write(os.path.join(patht, coordname), coordname)
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


class Campaign(object):
    """Class for a well based campaign.

    This is a class for a well based test campaign on a field site.
    It has a name, a descrition and a timeframe.

    Parameters
    ----------
    name : :class:`str`
        Name of the campaign.
    fieldsite : :class:`str` or :class:`Variable`, optional
        The field site.
        Default: ``"Fieldsite"``
    wells : :class:`dict`, optional
        The wells within the fild site. Keys are the well names and values
        are an instance of :class:`Well`.
        Default: ``None``
    wells : :class:`dict`, optional
        The tests within the campaign. Keys are the test names and values
        are an instance of :class:`Test`.
        Default: ``None``
    timeframe : :class:`str`, optional
        Timeframe of the campaign.
        Default: ``None``
    description : :class:`str`, optional
        Description of the field site.
        Default: ``"Welltest campaign"``
    """

    def __init__(
        self,
        name,
        fieldsite="Fieldsite",
        wells=None,
        tests=None,
        timeframe=None,
        description="Welltest campaign",
    ):
        self.name = _formstr(name)
        self.description = str(description)
        self._fieldsite = None
        self.fieldsite = fieldsite
        self.__wells = {}
        self.wells = wells
        self.__tests = {}
        self.tests = tests
        self.timeframe = str(timeframe)

    @property
    def fieldsite(self):
        """:class:`FieldSite`: Field site where the campaign was realised."""
        return self._fieldsite

    @fieldsite.setter
    def fieldsite(self, fieldsite):
        if fieldsite is not None:
            if isinstance(fieldsite, FieldSite):
                self._fieldsite = dcopy(fieldsite)
            else:
                self._fieldsite = FieldSite(str(fieldsite))
        else:
            self._fieldsite = None

    @property
    def wells(self):
        """:class:`dict`: Wells within the campaign."""
        return self.__wells

    @wells.setter
    def wells(self, wells):
        if wells is not None:
            if isinstance(wells, dict):
                for k in wells.keys():
                    if not isinstance(wells[k], Well):
                        raise ValueError(
                            "Campaign: some 'wells' are not of " + "type Well"
                        )
                    if not k == wells[k].name:
                        raise ValueError(
                            "Campaign: 'well'-keys should be "
                            + "the Well name"
                        )
                self.__wells = dcopy(wells)
            elif isinstance(wells, (list, tuple)):
                for wel in wells:
                    if not isinstance(wel, Well):
                        raise ValueError(
                            "Campaign: some 'wells' " + "are not of type Well"
                        )
                self.__wells = {}
                for wel in wells:
                    self.__wells[wel.name] = dcopy(wel)
            else:
                raise ValueError(
                    "Campaign: 'wells' should be given "
                    + "as dictonary or list"
                )
        else:
            self.__wells = {}
        self.__updatewells()

    def add_well(
        self, name, radius, coordinates, welldepth=1.0, aquiferdepth=None
    ):
        """Add a single well to the campaign.

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
        well = Well(name, radius, coordinates, welldepth, aquiferdepth)
        self.addwells(well)

    def addwells(self, wells):
        """Add some specified wells.

        This will add wells to the campaign.

        Parameters
        ----------
        wells : :class:`dict`
            Wells to be added.
        """
        if isinstance(wells, dict):
            for k in wells.keys():
                if not isinstance(wells[k], Well):
                    raise ValueError(
                        "Campaign_addwells: some 'wells' "
                        + "are not of type Well"
                    )
                if k in tuple(self.__wells.keys()):
                    raise ValueError(
                        "Campaign_addwells: some 'wells' "
                        + "are already present"
                    )
                if not k == wells[k].name:
                    raise ValueError(
                        "Campaign_addwells: 'well'-keys "
                        + "should be the Well name"
                    )
            for k in wells.keys():
                self.__wells[k] = dcopy(wells[k])
        elif isinstance(wells, (list, tuple)):
            for wel in wells:
                if not isinstance(wel, Well):
                    raise ValueError(
                        "Campaign_addwells: some 'wells' "
                        + "are not of type Well"
                    )
                if wel.name in tuple(self.__wells.keys()):
                    raise ValueError(
                        "Campaign_addwells: some 'wells' "
                        + "are already present"
                    )
            for wel in wells:
                self.__wells[wel.name] = dcopy(wel)
        elif isinstance(wells, Well):
            self.__wells[wells.name] = dcopy(wells)
        else:
            raise ValueError(
                "Campaign_addwells: 'wells' should be "
                + "given as dictonary, list or single 'Well'"
            )

    def delwells(self, wells):
        """Delete some specified wells.

        This will delete wells from the campaign. You can give a
        list of wells or a single well by name.

        Parameters
        ----------
        wells : :class:`list` of :class:`str` or :class:`str`
            Wells to be deleted.
        """
        if isinstance(wells, (list, tuple)):
            for wel in wells:
                if wel in tuple(self.__wells.keys()):
                    del self.__wells[wel]
        else:
            if wells in tuple(self.__wells.keys()):
                del self.__wells[wells]

    @property
    def tests(self):
        """:class:`dict`: Tests within the campaign."""
        return self.__tests

    @tests.setter
    def tests(self, tests):
        if tests is not None:
            if isinstance(tests, dict):
                for k in tests.keys():
                    if not isinstance(tests[k], Test):
                        raise ValueError(
                            "Campaign: 'tests' are not of " + "type Test"
                        )
                    if not k == tests[k].name:
                        raise ValueError(
                            "Campaign: 'tests'-keys "
                            + "should be the Test name"
                        )
                self.__tests = dcopy(tests)
            elif isinstance(tests, (list, tuple)):
                for tes in tests:
                    if not isinstance(tes, Test):
                        raise ValueError(
                            "Campaign: some 'tests' are not of " + "type Test"
                        )
                self.__tests = {}
                for tes in tests:
                    self.__tests[tes.name] = dcopy(tes)
            elif isinstance(tests, Test):
                self.__tests[tests.name] = dcopy(tests)
            else:
                raise ValueError(
                    "Campaign: 'tests' should be given "
                    + "as dictonary, list or 'Test'"
                )
        else:
            self.__tests = {}

    def addtests(self, tests):
        """Add some specified tests.

        This will add tests to the campaign.

        Parameters
        ----------
        tests : :class:`dict`
            Tests to be added.
        """
        if isinstance(tests, dict):
            for k in tests.keys():
                if not isinstance(tests[k], Test):
                    raise ValueError(
                        "Campaign_addtests: some 'tests' "
                        + "are not of type Test"
                    )
                if k in tuple(self.__tests.keys()):
                    raise ValueError(
                        "Campaign_addtests: some 'tests' "
                        + "are already present"
                    )
                if not k == tests[k].name:
                    raise ValueError(
                        "Campaign_addtests: 'tests'-keys "
                        + "should be the Test name"
                    )
            for k in tests.keys():
                self.__tests[k] = dcopy(tests[k])
        elif isinstance(tests, (list, tuple)):
            for tes in tests:
                if not isinstance(tes, Test):
                    raise ValueError(
                        "Campaign_addtests: some 'tests' "
                        + "are not of type Test"
                    )
                if tes.name in tuple(self.__tests.keys()):
                    raise ValueError(
                        "Campaign_addtests: some 'tests' "
                        + "are already present"
                    )
            for tes in tests:
                self.__tests[tes.name] = dcopy(tes)
        elif isinstance(tests, Test):
            if tests.name in tuple(self.__tests.keys()):
                raise ValueError("Campaign.addtests: 'test' already present")
            self.__tests[tests.name] = dcopy(tests)
        else:
            raise ValueError(
                "Campaign_addtests: 'tests' should be "
                + "given as dictonary, list or single 'Test'"
            )

    def deltests(self, tests):
        """Delete some specified tests.

        This will delete tests from the campaign. You can give a
        list of tests or a single test by name.

        Parameters
        ----------
        tests : :class:`list` of :class:`str` or :class:`str`
            Tests to be deleted.
        """
        if isinstance(tests, (list, tuple)):
            for tes in tests:
                if tes in tuple(self.__tests.keys()):
                    del self.__tests[tes]
        else:
            if tests in tuple(self.__tests.keys()):
                del self.__tests[tests]

    def __updatewells(self):
        pass

    def plot(self, select_tests=None, **kwargs):
        """Generate a plot of the tests within the campaign.

        This will plot an overview of the tests within the campaign.

        Parameters
        ----------
        select_tests : :class:`list`, optional
            Tests that should be plotted. If None, all will be displayed.
            Default: ``None``
        **kwargs
            Keyword-arguments forwarded to :any:`CampaignPlot`
        """
        CampaignPlot(self, select_tests, **kwargs)

    def plot_wells(self, **kwargs):
        """Generate a plot of the wells within the campaign.

        This will plot an overview of the wells within the campaign.

        Parameters
        ----------
        **kwargs
            Keyword-arguments forwarded to :any:`WellPlot`.
        """
        WellPlot(self, **kwargs)

    def save(self, path="", name=None):
        """Save the campaign to file.

        This writes the campaign to a csv file.

        Parameters
        ----------
        path : :class:`str`, optional
            Path where the variable should be saved. Default: ``""``
        name : :class:`str`, optional
            Name of the file. If ``None``, the name will be generated by
            ``"Cmp_"+name``. Default: ``None``

        Notes
        -----
        The file will get the suffix ``".cmp"``.
        """
        path = os.path.normpath(path)
        # create the path if not existing
        if not os.path.exists(path):
            os.makedirs(path)
        # create a standard name if None is given
        if name is None:
            name = "Cmp_" + self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".cmp":
            name += ".cmp"
        name = _formname(name)
        # create temporal directory for the included files
        patht = os.path.join(path, ".tmpcmp")
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(os.path.join(patht, "info.csv"), "w") as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Campaign"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            writer.writerow(["timeframe", self.timeframe])
            # define names for the variable-files
            if self._fieldsite is not None:
                fieldsname = name[:-4] + "_Fieldsite.fds"
                # save variable-files
                writer.writerow(["fieldsite", fieldsname])
                self._fieldsite.save(patht, fieldsname)
            else:
                writer.writerow(["fieldsite", "None"])

            wkeys = tuple(self.wells.keys())
            writer.writerow(["Wells", len(wkeys)])
            wellsname = {}
            for k in wkeys:
                wellsname[k] = name[:-4] + "_" + k + "_Well.wel"
                writer.writerow([k, wellsname[k]])
                self.wells[k].save(patht, wellsname[k])

            tkeys = tuple(self.tests.keys())
            writer.writerow(["Tests", len(tkeys)])
            testsname = {}
            for k in tkeys:
                testsname[k] = name[:-4] + "_" + k + "_Test.tst"
                writer.writerow([k, testsname[k]])
                self.tests[k].save(patht, testsname[k])

        # compress everything to one zip-file
        with zipfile.ZipFile(os.path.join(path, name), "w") as zfile:
            zfile.write(os.path.join(patht, "info.csv"), "info.csv")
            if self._fieldsite is not None:
                zfile.write(os.path.join(patht, fieldsname), fieldsname)
            for k in wkeys:
                zfile.write(os.path.join(patht, wellsname[k]), wellsname[k])
            for k in tkeys:
                zfile.write(os.path.join(patht, testsname[k]), testsname[k])
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


def load_fieldsite(fdsfile):
    """Load a field site from file.

    This reads a field site from a csv file.

    Parameters
    ----------
    fdsfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(fdsfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            if next(data)[0] != "Fieldsite":
                raise Exception
            name = next(data)[1]
            description = next(data)[1]
            coordinfo = next(data)[1]
            if coordinfo == "None":
                coordinates = None
            else:
                coordinates = load_var(TxtIO(zfile.open(coordinfo)))
        fieldsite = FieldSite(name, description, coordinates)
    except Exception:
        raise Exception(
            "loadFieldSite: loading the fieldsite " + "was not possible"
        )
    return fieldsite


def load_campaign(cmpfile):
    """Load a campaign from file.

    This reads a campaign from a csv file.

    Parameters
    ----------
    cmpfile : :class:`str`
        Path to the file
    """
    try:
        with zipfile.ZipFile(cmpfile, "r") as zfile:
            info = TxtIO(zfile.open("info.csv"))
            data = csv.reader(info)
            if next(data)[0] != "Campaign":
                raise Exception
            name = next(data)[1]
            description = next(data)[1]
            timeframe = next(data)[1]
            row = _nextr(data)
            if row[1] == "None":
                fieldsite = None
            else:
                fieldsite = load_fieldsite(BytIO(zfile.read(row[1])))
            wcnt = np.int(next(data)[1])
            wells = {}
            for __ in range(wcnt):
                row = _nextr(data)
                wells[row[0]] = load_well(BytIO(zfile.read(row[1])))

            tcnt = np.int(next(data)[1])
            tests = {}
            for __ in range(tcnt):
                row = _nextr(data)
                tests[row[0]] = load_test(BytIO(zfile.read(row[1])))

        campaign = Campaign(
            name, fieldsite, wells, tests, timeframe, description
        )
    except Exception:
        raise Exception(
            "loadPumpingTest: loading the pumpingtest " + "was not possible"
        )
    return campaign
