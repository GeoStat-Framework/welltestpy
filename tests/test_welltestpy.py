# -*- coding: utf-8 -*-
"""
This is the unittest of AnaFlow.
"""

import unittest
import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import welltestpy as wtp
from welltestpy.tools import triangulate, sym, plot_well_pos

import anaflow as ana


class TestWTP(unittest.TestCase):
    def setUp(self):
        self.rate = -1e-4
        self.time = np.geomspace(10, 7200, 10)
        self.transmissivity = 1e-4
        self.storage = 1e-4
        self.s_types = ["ST", "S1"]

    def test_create(self):
        # create the field-site and the campaign
        field = wtp.FieldSite(name="UFZ", coordinates=[51.3538, 12.4313])
        campaign = wtp.Campaign(name="UFZ-campaign", fieldsite=field)

        # add 4 wells to the campaign
        campaign.add_well(name="well_0", radius=0.1, coordinates=(0.0, 0.0))
        campaign.add_well(name="well_1", radius=0.1, coordinates=(1.0, -1.0))
        campaign.add_well(name="well_2", radius=0.1, coordinates=(2.0, 2.0))
        campaign.add_well(name="well_3", radius=0.1, coordinates=(-2.0, -1.0))

        # generate artificial drawdown data with the Theis solution
        self.rad = [
            campaign.wells["well_0"].radius,  # well radius of well_0
            campaign.wells["well_0"] - campaign.wells["well_1"],  # dist. 0-1
            campaign.wells["well_0"] - campaign.wells["well_2"],  # dist. 0-2
            campaign.wells["well_0"] - campaign.wells["well_3"],  # dist. 0-3
        ]
        drawdown = ana.theis(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            transmissivity=self.transmissivity,
            rate=self.rate,
        )

        # create a pumping test at well_0
        pumptest = wtp.PumpingTest(
            name="well_0",
            pumpingwell="well_0",
            pumpingrate=self.rate,
            description="Artificial pump test with Theis",
        )

        # add the drawdown observation at the 4 wells
        pumptest.add_transient_obs("well_0", self.time, drawdown[:, 0])
        pumptest.add_transient_obs("well_1", self.time, drawdown[:, 1])
        pumptest.add_transient_obs("well_2", self.time, drawdown[:, 2])
        pumptest.add_transient_obs("well_3", self.time, drawdown[:, 3])

        # add the pumping test to the campaign
        campaign.addtests(pumptest)
        # plot the well constellation and a test overview
        campaign.plot_wells()
        campaign.plot()
        # save the whole campaign
        campaign.save()
        # test making steady
        campaign.tests["well_0"].make_steady()

    def test_est_theis(self):
        campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
        estimation = wtp.estimate.Theis("est_theis", campaign, generate=True)
        estimation.run()
        res = estimation.estimated_para
        estimation.sensitivity()
        self.assertAlmostEqual(np.exp(res["mu"]), self.transmissivity, 2)
        self.assertAlmostEqual(np.exp(res["lnS"]), self.storage, 2)
        sens = estimation.sens
        for s_typ in self.s_types:
            self.assertTrue(sens[s_typ]["mu"] > sens[s_typ]["lnS"])

    def test_est_thiem(self):
        campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
        estimation = wtp.estimate.Thiem("est_thiem", campaign, generate=True)
        estimation.run()
        res = estimation.estimated_para
        # since we only have one parameter,
        # we need a dummy parameter to estimate sensitivity
        estimation.gen_setup(dummy=True)
        estimation.sensitivity()
        self.assertAlmostEqual(np.exp(res["mu"]), self.transmissivity, 2)
        sens = estimation.sens
        for s_typ in self.s_types:
            self.assertTrue(sens[s_typ]["mu"] > sens[s_typ]["dummy"])

    def test_est_ext_thiem2D(self):
        campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
        estimation = wtp.estimate.ExtThiem2D(
            "est_ext_thiem2D", campaign, generate=True
        )
        estimation.run()
        res = estimation.estimated_para
        estimation.sensitivity()
        self.assertAlmostEqual(np.exp(res["mu"]), self.transmissivity, 2)
        self.assertAlmostEqual(res["var"], 0.0, 0)
        sens = estimation.sens
        for s_typ in self.s_types:
            self.assertTrue(sens[s_typ]["mu"] > sens[s_typ]["var"])
            self.assertTrue(sens[s_typ]["var"] > sens[s_typ]["len_scale"])

    # def test_est_ext_thiem3D(self):
    #     campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
    #     estimation = wtp.estimate.ExtThiem3D(
    #         "est_ext_thiem3D", campaign, generate=True
    #     )
    #     estimation.run()
    #     res = estimation.estimated_para
    #     estimation.sensitivity()
    #     self.assertAlmostEqual(np.exp(res["mu"]), self.transmissivity, 2)
    #     self.assertAlmostEqual(res["var"], 0.0, 0)

    def test_triangulate(self):
        dist_mat = np.zeros((4, 4), dtype=float)
        dist_mat[0, 1] = 3  # distance between well 0 and 1
        dist_mat[0, 2] = 4  # distance between well 0 and 2
        dist_mat[1, 2] = 2  # distance between well 1 and 2
        dist_mat[0, 3] = 1  # distance between well 0 and 3
        dist_mat[1, 3] = 3  # distance between well 1 and 3
        dist_mat[2, 3] = -1  # unknown distance between well 2 and 3
        dist_mat = sym(dist_mat)  # make the distance matrix symmetric
        well_const = triangulate(dist_mat, prec=0.1)
        self.assertEqual(len(well_const), 4)
        # plot all possible well constellations
        plot_well_pos(well_const)


if __name__ == "__main__":
    unittest.main()
