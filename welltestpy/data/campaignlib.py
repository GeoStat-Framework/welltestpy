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
from welltestpy.tools.plotter import CampaignPlot, WellPlot
from welltestpy.data.varlib import (Variable, CoordinatesVar, loadVar,
                                    Well, loadWell,
                                    _nextr, _formstr, _formname)
from welltestpy.data.testslib import (Test, loadTest)


class FieldSite(object):
    def __init__(self, name, description="Field site", coordinates=None):
        # , picture=None):
        self.name = _formstr(name)
        self.description = str(description)
        self.coordinates = coordinates
#        self.picture = picture

    @property
    def info(self):
        print("----")
        print("Field-site:   "+str(self.name))
        print("Description:  "+str(self.description))
#        if hasattr(self, '_picture'):
#            print("Picture-file: "+self.picture)
        print("--")
        if hasattr(self, '_coordinates'):
            self._coordinates.info
        print("----")

    @property
    def coordinates(self):
        if hasattr(self, '_coordinates'):
            return self._coordinates.value

    @coordinates.setter
    def coordinates(self, coordinates):
        if coordinates is not None:
            if isinstance(coordinates, Variable):
                # if isinstance(coordinates, CoordinatesVar):
                self._coordinates = dcopy(coordinates)
#                else:
#                    raise ValueError("Fieldsite: 'coordinates' are not of "+\
#                                     "type CoordinatesVar")
            else:
                self._coordinates = CoordinatesVar(coordinates[0],
                                                   coordinates[1])
        else:
            if hasattr(self, '_coordinates'):
                del self._coordinates

    def __repr__(self):
        return self.name

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
            name = "Field_"+self.name
        # ensure the name ends with '.csv'
        if name[-4:] != ".fds":
            name += ".fds"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmpfield/"
        patht = path+tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht+"info.csv", 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Fieldsite"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            # define names for the variable-files
            if hasattr(self, '_coordinates'):
                coordname = name[:-4]+"_CooVar.var"
                # save variable-files
                writer.writerow(["coordinates", coordname])
                self._coordinates.save(patht, coordname)
            else:
                writer.writerow(["coordinates", "None"])
        # compress everything to one zip-file
        with zipfile.ZipFile(path+name, "w") as zfile:
            zfile.write(patht+"info.csv", "info.csv")
            if hasattr(self, '_coordinates'):
                zfile.write(patht+coordname, coordname)
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)

#    @property
#    def picture(self):
#        if hasattr(self, '_picture'):
#            return self._picture
#
#    @picture.setter
#    def picture(self,picture):
#        if picture != None:
#            picture = str(picture)
#            # if os.access(picture, os.W_OK):
#            if im.what(picture):
#                self._picture = picture
#            else:
#                raise ValueError("Fieldsite: 'picture' is no valid pic-file")
#        else:
#            if hasattr(self, '_picture'):
#                del self._picture


class Campaign(object):
    def __init__(self, name, fieldsite="Fieldsite", wells=None, tests=None,
                 timeframe=None, description="Welltest campaign"):
        self.name = _formstr(name)
        self.description = str(description)
        self.fieldsite = fieldsite
        self.wells = wells
        self.tests = tests
        self.timeframe = str(timeframe)

    @property
    def fieldsite(self):
        if hasattr(self, '_fieldsite'):
            return self._fieldsite

    @fieldsite.setter
    def fieldsite(self, fieldsite):
        if fieldsite is not None:
            if isinstance(fieldsite, FieldSite):
                self._fieldsite = dcopy(fieldsite)
            else:
                self._fieldsite = FieldSite(str(fieldsite))
        else:
            if hasattr(self, '_fieldsite'):
                del self._fieldsite

    @property
    def wells(self):
        return self.__wells

    @wells.setter
    def wells(self, wells):
        if wells is not None:
            if isinstance(wells, dict):
                for k in wells.keys():
                    if not isinstance(wells[k], Well):
                        raise ValueError("Campaign: some 'wells' are not of " +
                                         "type Well")
                    if not k == wells[k].name:
                        raise ValueError("Campaign: 'well'-keys should be " +
                                         "the Well name")
                self.__wells = dcopy(wells)
            elif isinstance(wells, (list, tuple)):
                for wel in wells:
                    if not isinstance(wel, Well):
                        raise ValueError("Campaign: some 'wells' " +
                                         "are not of type Well")
                self.__wells = {}
                for wel in wells:
                    self.__wells[wel.name] = dcopy(wel)
            else:
                raise ValueError("Campaign: 'wells' should be given " +
                                 "as dictonary or list")
        else:
            self.__wells = {}
        self.__updatewells()

    def addwells(self, wells):
        if isinstance(wells, dict):
            for k in wells.keys():
                if not isinstance(wells[k], Well):
                    raise ValueError("Campaign_addwells: some 'wells' " +
                                     "are not of type Well")
                if k in tuple(self.__wells.keys()):
                    raise ValueError("Campaign_addwells: some 'wells' " +
                                     "are already present")
                if not k == wells[k].name:
                    raise ValueError("Campaign_addwells: 'well'-keys " +
                                     "should be the Well name")
            for k in wells.keys():
                self.__wells[k] = dcopy(wells[k])
        elif isinstance(wells, (list, tuple)):
            for wel in wells:
                if not isinstance(wel, Well):
                    raise ValueError("Campaign_addwells: some 'wells' " +
                                     "are not of type Well")
                if wel.name in tuple(self.__wells.keys()):
                    raise ValueError("Campaign_addwells: some 'wells' " +
                                     "are already present")
            for wel in wells:
                self.__wells[wel.name] = dcopy(wel)
        elif isinstance(wells, Well):
            self.__wells[wells.name] = dcopy(wells)
        else:
            raise ValueError("Campaign_addwells: 'wells' should be " +
                             "given as dictonary, list or single 'Well'")

    def delwells(self, wells):
        if isinstance(wells, (list, tuple)):
            for wel in wells:
                if wel in tuple(self.__wells.keys()):
                    del self.__wells[wel]
        else:
            if wells in tuple(self.__wells.keys()):
                del self.__wells[wells]

    @property
    def tests(self):
        return self.__tests

    @tests.setter
    def tests(self, tests):
        if tests is not None:
            if isinstance(tests, dict):
                for k in tests.keys():
                    if not isinstance(tests[k], Test):
                        raise ValueError("Campaign: 'tests' are not of " +
                                         "type Test")
                    if not k == tests[k].name:
                        raise ValueError("Campaign: 'tests'-keys " +
                                         "should be the Test name")
                self.__tests = dcopy(tests)
            elif isinstance(tests, (list, tuple)):
                for tes in tests:
                    if not isinstance(tes, Test):
                        raise ValueError("Campaign: some 'tests' are not of " +
                                         "type Test")
                self.__tests = {}
                for tes in tests:
                    self.__tests[tes.name] = dcopy(tes)
            elif isinstance(tests, Test):
                self.__tests[tests.name] = dcopy(tests)
            else:
                raise ValueError("Campaign: 'tests' should be given " +
                                 "as dictonary, list or 'Test'")
        else:
            self.__tests = {}

    def addtests(self, tests):
        if isinstance(tests, dict):
            for k in tests.keys():
                if not isinstance(tests[k], Test):
                    raise ValueError("Campaign_addtests: some 'tests' " +
                                     "are not of type Test")
                if k in tuple(self.__tests.keys()):
                    raise ValueError("Campaign_addtests: some 'tests' " +
                                     "are already present")
                if not k == tests[k].name:
                    raise ValueError("Campaign_addtests: 'tests'-keys " +
                                     "should be the Test name")
            for k in tests.keys():
                self.__tests[k] = dcopy(tests[k])
        elif isinstance(tests, (list, tuple)):
            for tes in tests:
                if not isinstance(tes, Test):
                    raise ValueError("Campaign_addtests: some 'tests' " +
                                     "are not of type Test")
                if tes.name in tuple(self.__tests.keys()):
                    raise ValueError("Campaign_addtests: some 'tests' " +
                                     "are already present")
            for tes in tests:
                self.__tests[tes.name] = dcopy(tes)
        elif isinstance(tests, Test):
            self.__tests[tests.name] = dcopy(tests)
        else:
            raise ValueError("Campaign_addtests: 'tests' should be " +
                             "given as dictonary, list or single 'Test'")

    def deltests(self, tests):
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
        CampaignPlot(self, select_tests, **kwargs)

    def plot_wells(self, **kwargs):
        WellPlot(self, **kwargs)

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
        if name[-4:] != ".cmp":
            name += ".cmp"
        name = _formname(name)
        # create temporal directory for the included files
        tmp = ".tmpcmp/"
        patht = path+tmp
        if os.path.exists(patht):
            shutil.rmtree(patht, ignore_errors=True)
        os.makedirs(patht)
        # write the csv-file
        # with open(patht+name[:-4]+".csv", 'w') as csvf:
        with open(patht+"info.csv", 'w') as csvf:
            writer = csv.writer(csvf, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["Campaign"])
            writer.writerow(["name", self.name])
            writer.writerow(["description", self.description])
            writer.writerow(["timeframe", self.timeframe])
            # define names for the variable-files
            if hasattr(self, '_fieldsite'):
                fieldsname = name[:-4]+"_Fieldsite.fds"
                # save variable-files
                writer.writerow(["fieldsite", fieldsname])
                self._fieldsite.save(patht, fieldsname)
            else:
                writer.writerow(["fieldsite", "None"])

            wkeys = tuple(self.wells.keys())
            writer.writerow(["Wells", len(wkeys)])
            wellsname = {}
            for k in wkeys:
                wellsname[k] = name[:-4]+"_"+k+"_Well.wel"
                writer.writerow([k, wellsname[k]])
                self.wells[k].save(patht, wellsname[k])

            tkeys = tuple(self.tests.keys())
            writer.writerow(["Tests", len(tkeys)])
            testsname = {}
            for k in tkeys:
                testsname[k] = name[:-4]+"_"+k+"_Test.tst"
                writer.writerow([k, testsname[k]])
                self.tests[k].save(patht, testsname[k])

        # compress everything to one zip-file
        with zipfile.ZipFile(path+name, "w") as zfile:
            zfile.write(patht+"info.csv", "info.csv")
            if hasattr(self, '_fieldsite'):
                zfile.write(patht+fieldsname, fieldsname)
            for k in wkeys:
                zfile.write(patht+wellsname[k], wellsname[k])
            for k in tkeys:
                zfile.write(patht+testsname[k], testsname[k])
        # delete the temporary directory
        shutil.rmtree(patht, ignore_errors=True)


def loadFieldSite(fdsfile):
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
                coordinates = loadVar(TxtIO(zfile.open(coordinfo)))
        fieldsite = FieldSite(name, description, coordinates)
    except:
        raise Exception("loadFieldSite: loading the fieldsite " +
                        "was not possible")
    return fieldsite


def loadCampaign(cmpfile):
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
                fieldsite = loadFieldSite(BytIO(zfile.read(row[1])))
            wcnt = np.int(next(data)[1])
            wells = {}
            for __ in range(wcnt):
                row = _nextr(data)
                wells[row[0]] = loadWell(BytIO(zfile.read(row[1])))

            tcnt = np.int(next(data)[1])
            tests = {}
            for __ in range(tcnt):
                row = _nextr(data)
                tests[row[0]] = loadTest(BytIO(zfile.read(row[1])))

        campaign = Campaign(name, fieldsite, wells, tests,
                            timeframe, description)
    except:
        raise Exception("loadPumpingTest: loading the pumpingtest " +
                        "was not possible")
    return campaign
