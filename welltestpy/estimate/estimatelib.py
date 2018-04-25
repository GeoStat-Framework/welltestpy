# -*- coding: utf-8 -*-
"""Estimation library from welltestpy

This module contanins methods for parameter estimation from well-test data.
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy as dcopy
import os
import time as timemodule

import numpy as np
import spotpy

from welltestpy.data.campaignlib import Campaign
from welltestpy.process.processlib import (normpumptest,
                                           filterdrawdown)
from welltestpy.estimate.spotpy_classes import (Theissetup,
                                                Stat2Dsetup)
from welltestpy.tools.plotter import (plotfitting3D,
                                      plotfitting3Dtheis,
                                      plotparainteract,
                                      plotparatrace,
                                      plotsensitivity)


class Estimate(Campaign):
    def __init__(self, name, campaign=None,
                 setup=None, description="Estimation",
                 results=None):
        if campaign is None:
            super(Estimate, self).__init__(name, description=description)
        else:
            super(Estimate, self).__init__(name,
                                           campaign.fieldsite,
                                           campaign.wells,
                                           campaign.tests,
                                           campaign.timeframe,
                                           description)

        self.campaign = dcopy(campaign)
        self.setup = setup
        self.results = results

    def resetdata(self):
        self.__init__(self.name, self.campaign, self.setup, self.description)


class Stat2Dest(object):
    def __init__(self, name, campaign, testinclude=None):
        self.name = name
        self.campaign_raw = dcopy(campaign)
        self.campaign = dcopy(campaign)

        self.prate = None

        self.time = None
        self.rad = None
        self.data = None
        self.radnames = None

        self.para = None
        self.result = None
        self.sens = None

        if testinclude is None:
            wells = self.campaign.tests.keys()
            pumpdict = {}
            for wel in wells:
                pumpdict[wel] = self.campaign.tests[wel].observations.keys()
            self.testinclude = pumpdict
        else:
            self.testinclude = testinclude

    def setpumprate(self, prate=-1.):
        for test in self.testinclude:
            normpumptest(self.campaign.tests[test], pumpingrate=prate)
        self.prate = prate

    def settime(self, time=None, tmin=-np.inf, tmax=np.inf):
        if time is None:
            for test in self.testinclude:
                for obs in self.testinclude[test]:
                    temptime, _ = self.campaign.tests[test].observations[obs]()
                    tmin = max(tmin, temptime.min())
                    tmax = min(tmax, temptime.max())

            # set the first timepoint to at least 10s
            tmin = max(tmin, 10)

#            time = np.expm1(np.linspace(np.log1p(tmin),
#                                        np.log1p(tmax), 10))
            time = np.power(np.linspace(np.sqrt(tmin+1.),
                                        np.sqrt(tmax+1.), 10)+1., 2) - 1.
        for test in self.testinclude:
            for obs in self.testinclude[test]:
                filterdrawdown(self.campaign.tests[test].observations[obs],
                               tout=time)

        self.time = time

    def genrtdata(self):
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

                tempname = self.campaign.tests[test].pumpingwell+"-"+obs
                radnames = np.hstack((radnames, tempname))

        # sort everything by the radii
        idx = rad.argsort()

        self.rad = rad[idx]
        self.data = data[:, idx]
        self.radnames = radnames[idx]

    def run(self, rep=5000, parallel="seq", run=True, folder=None,
            dbname=None, plotname1=None, plotname2=None, plotname3=None,
            mu=None, sig2=None, corr=None, lnS=None,  # to fix some parameters
            murange=None, sig2range=None, corrrange=None, lnSrange=None):
            # parameter ranges

        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd()+"/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        if not os.path.exists(dire):
            os.makedirs(dire)

        if dbname is None:
            dbname = folder+act_time+"_stat2D_db"
        else:
            dbname = folder+dbname
        if plotname1 is None:
            plotname1 = folder+act_time+"_stat2D_plot_paratrace.pdf"
        else:
            plotname1 = folder+plotname1
        if plotname2 is None:
            plotname2 = folder+act_time+"_stat2D_plot_fit.pdf"
        else:
            plotname2 = folder+plotname2
        if plotname3 is None:
            plotname3 = folder+act_time+"_stat2D_plot_parainteract.pdf"
        else:
            plotname3 = folder+plotname3
        paraname = folder+act_time+"_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r'$\mu$')
            paranames.append('mu')
        if sig2 is None:
            paralabels.append(r'$\sigma^{2}$')
            paranames.append('sig2')
        if corr is None:
            paralabels.append(r'$\ell$')
            paranames.append('corr')
        if lnS is None:
            paralabels.append(r"$\ln(S)$")
            paranames.append('lnS')

        if run:
            # generate the spotpy-setup
            setup = Stat2Dsetup(self.rad, self.time, self.data, Qw=self.prate,
                                mu=mu, sig2=sig2, corr=corr, lnS=lnS,
                                # to fix some parameters
                                murange=murange, sig2range=sig2range,
                                corrrange=corrrange, lnSrange=lnSrange)
            # initialize the sampler
            sampler = spotpy.algorithms.sceua(setup,
                                              dbname=dbname,
                                              dbformat='csv',
                                              parallel=parallel,
                                              save_sim=False)
            # start the estimation with the sce-ua algorithm
            sampler.sample(rep, ngs=20, kstop=100, pcento=1e-4, peps=1e-3)
            # save best parameter-set
            self.para = sampler.status.params
            np.savetxt(paraname, self.para)
            # save the results
            self.result = sampler.getdata()

        # plot the estimation-results
        plotparatrace(self.result,
                      parameternames=paranames,
                      parameterlabels=paralabels,
                      stdvalues=self.para,
                      filename=plotname1)
        plotfitting3D(self.data, self.para, self.rad, self.time,
                      self.radnames, self.prate, plotname2)
        plotparainteract(self.result, paralabels, plotname3)

    def sensitivity(self, rep=5000, parallel="seq",
                    folder=None, dbname=None, plotname=None, plotname1=None,
                    mu=None, sig2=None, corr=None, lnS=None,
                    # to fix some parameters
                    murange=None, sig2range=None,
                    corrrange=None, lnSrange=None):
                    # parameter ranges

        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd()+"/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        if not os.path.exists(dire):
            os.makedirs(dire)

        if dbname is None:
            dbname = folder+act_time+"_sensitivity_db"
        else:
            dbname = folder+dbname
        if plotname is None:
            plotname = folder+act_time+"_stat2D_plot_sensitivity.pdf"
        else:
            plotname = folder+plotname
        if plotname1 is None:
            plotname1 = folder+act_time+"_stat2D_plot_senstrace.pdf"
        else:
            plotname1 = folder+plotname1
        sensname = folder+act_time+"_FAST_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r'$T^G$')
            paranames.append('mu')
        if sig2 is None:
            paralabels.append(r'$\sigma^{2}$')
            paranames.append('sig2')
        if corr is None:
            paralabels.append(r'$\ell$')
            paranames.append('corr')
        if lnS is None:
            paralabels.append(r"$S$")
            paranames.append('lnS')

        bestvalues = {}

        for par_i, par_e in enumerate(paranames):
            bestvalues[par_e] = self.para[par_i]

        # generate the spotpy-setup
        setup = Stat2Dsetup(self.rad, self.time, self.data, Qw=self.prate,
                            bestvalues=bestvalues,
                            mu=mu, sig2=sig2, corr=corr, lnS=lnS,
                            # to fix some parameters
                            murange=murange, sig2range=sig2range,
                            corrrange=corrrange, lnSrange=lnSrange)

        # initialize the sampler
        sampler = spotpy.algorithms.fast(setup,
                                         dbname=dbname,
                                         dbformat='csv',
                                         parallel=parallel,
                                         save_sim=True)

        sampler.sample(rep)

        data = sampler.getdata()

        parmin = sampler.parameter()['minbound']
        parmax = sampler.parameter()['maxbound']

#        bounds = []
#        for i in range(len(parmin)):
#            bounds.append([parmin[i], parmax[i]])

        bounds = zip(parmin, parmax)

        self.sens = sampler.analyze(bounds,
                                    data['like1'],
                                    len(bounds),
                                    paranames)

        np.savetxt(sensname, self.sens["ST"])

        plotsensitivity(paralabels, self.sens, plotname)
        plotparatrace(data,
                      parameternames=paranames,
                      parameterlabels=paralabels,
                      stdvalues=self.para,
                      filename=plotname1)


# theis

class Theisest(object):
    def __init__(self, name, campaign, testinclude=None):
        self.name = name
        self.campaign_raw = dcopy(campaign)
        self.campaign = dcopy(campaign)

        self.prate = None

        self.time = None
        self.rad = None
        self.data = None
        self.radnames = None

        self.para = None
        self.result = None
        self.sens = None

        if testinclude is None:
            wells = self.campaign.tests.keys()
            pumpdict = {}
            for wel in wells:
                pumpdict[wel] = self.campaign.tests[wel].observations.keys()
            self.testinclude = pumpdict
        else:
            self.testinclude = testinclude

    def setpumprate(self, prate=-1.):
        for test in self.testinclude:
            normpumptest(self.campaign.tests[test], pumpingrate=prate)
        self.prate = prate

    def settime(self, time=None, tmin=-np.inf, tmax=np.inf):
        if time is None:
            for test in self.testinclude:
                for obs in self.testinclude[test]:
                    temptime, _ = self.campaign.tests[test].observations[obs]()
                    tmin = max(tmin, temptime.min())
                    tmax = min(tmax, temptime.max())

            # set the first timepoint to at least 10s
            tmin = max(tmin, 10)

#            time = np.expm1(np.linspace(np.log1p(tmin),
#                                        np.log1p(tmax), 10))
            time = np.power(np.linspace(np.sqrt(tmin+1.),
                                        np.sqrt(tmax+1.), 10)+1., 2) - 1.
        for test in self.testinclude:
            for obs in self.testinclude[test]:
                filterdrawdown(self.campaign.tests[test].observations[obs],
                               tout=time)

        self.time = time

    def genrtdata(self):
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

                tempname = self.campaign.tests[test].pumpingwell+"-"+obs
                radnames = np.hstack((radnames, tempname))

        # sort everything by the radii
        idx = rad.argsort()

        self.rad = rad[idx]
        self.data = data[:, idx]
        self.radnames = radnames[idx]

    def run(self, rep=5000, parallel="seq", run=True, folder=None,
            dbname=None, plotname1=None, plotname2=None, plotname3=None,
            mu=None, lnS=None,
            # to fix some parameters
            murange=None, lnSrange=None):
            # parameter ranges

        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd()+"/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        if not os.path.exists(dire):
            os.makedirs(dire)

        if dbname is None:
            dbname = folder+act_time+"_stat2D_db"
        else:
            dbname = folder+dbname
        if plotname1 is None:
            plotname1 = folder+act_time+"_stat2D_plot_paratrace.pdf"
        else:
            plotname1 = folder+plotname1
        if plotname2 is None:
            plotname2 = folder+act_time+"_stat2D_plot_fit.pdf"
        else:
            plotname2 = folder+plotname2
        if plotname3 is None:
            plotname3 = folder+act_time+"_stat2D_plot_parainteract.pdf"
        else:
            plotname3 = folder+plotname3
        paraname = folder+act_time+"_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r'$\mu$')
            paranames.append('mu')
        if lnS is None:
            paralabels.append(r"$\ln(S)$")
            paranames.append('lnS')

        if run:
            # generate the spotpy-setup
            setup = Theissetup(self.rad, self.time, self.data, Qw=self.prate,
                               mu=mu, lnS=lnS,  # to fix some parameters
                               murange=murange, lnSrange=lnSrange)
            # initialize the sampler
            sampler = spotpy.algorithms.sceua(setup,
                                              dbname=dbname,
                                              dbformat='csv',
                                              parallel=parallel,
                                              save_sim=False)
            # start the estimation with the sce-ua algorithm
            sampler.sample(rep, ngs=20, kstop=100, pcento=1e-4, peps=1e-3)
            # save best parameter-set
            self.para = sampler.status.params
            np.savetxt(paraname, self.para)
            # save the results
            self.result = sampler.getdata()

        # plot the estimation-results
        plotparatrace(self.result,
                      parameternames=paranames,
                      parameterlabels=paralabels,
                      stdvalues=self.para,
                      filename=plotname1)
        plotfitting3Dtheis(self.data, self.para, self.rad, self.time,
                           self.radnames, self.prate, plotname2)
        plotparainteract(self.result, paralabels, plotname3)

    def sensitivity(self, rep=5000, parallel="seq",
                    folder=None, dbname=None, plotname=None,
                    mu=None, lnS=None,  # to fix some parameters
                    murange=None, lnSrange=None):  # parameter ranges

        act_time = timemodule.strftime("%Y-%m-%d_%H-%M-%S")

        # generate the filenames
        if folder is None:
            folder = os.getcwd()+"/"
        elif folder[-1] != "/":
            folder += "/"
        dire = os.path.dirname(folder)
        if not os.path.exists(dire):
            os.makedirs(dire)

        if dbname is None:
            dbname = folder+act_time+"_sensitivity_db"
        else:
            dbname = folder+dbname
        if plotname is None:
            plotname = folder+act_time+"_stat2D_plot_sensitivity.pdf"
        else:
            plotname = folder+plotname
        sensname = folder+act_time+"_FAST_estimate.txt"

        # generate the parameter-names for plotting
        paralabels = []
        paranames = []
        if mu is None:
            paralabels.append(r'$T^G$')
            paranames.append('mu')
        if lnS is None:
            paralabels.append(r"$S$")
            paranames.append('lnS')

        bestvalues = {}

        for par_i, par_e in enumerate(paranames):
            bestvalues[par_e] = self.para[par_i]

        # generate the spotpy-setup
        setup = Theissetup(self.rad, self.time, self.data, Qw=self.prate,
                           bestvalues=bestvalues,
                           mu=mu, lnS=lnS,  # to fix some parameters
                           murange=murange, lnSrange=lnSrange)

        # initialize the sampler
        sampler = spotpy.algorithms.fast(setup,
                                         dbname=dbname,
                                         dbformat='csv',
                                         parallel=parallel,
                                         save_sim=True)

        sampler.sample(rep)

        data = sampler.getdata()

        parmin = sampler.parameter()['minbound']
        parmax = sampler.parameter()['maxbound']

#        bounds = []
#        for i in range(len(parmin)):
#            bounds.append([parmin[i], parmax[i]])

        bounds = zip(parmin, parmax)

        self.sens = sampler.analyze(bounds,
                                    data['like1'],
                                    len(bounds),
                                    paranames)

        np.savetxt(sensname, self.sens["ST"])

        plotsensitivity(paralabels, self.sens, plotname)
