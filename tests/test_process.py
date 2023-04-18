# -*- coding: utf-8 -*-
"""
This is the unittest of welltestpy.process.
"""
import copy
import unittest

import numpy as np

import welltestpy as wtp


class TestProcess(unittest.TestCase):
    def setUp(self):
        # generate artificial data
        self.trns_obs = wtp.data.DrawdownObs("trans", [1, 2, 3], [4, 5, 6])
        self.stdy_obs = wtp.data.StdyHeadObs("steady", [1, 2, 3])

    def test_cooper_jacob(self):
        # create copies to not alter data of setUp
        trns_copy = copy.deepcopy(self.trns_obs)
        stdy_copy = copy.deepcopy(self.stdy_obs)
        # apply correction
        wtp.process.cooper_jacob_correction(trns_copy, sat_thickness=4)
        wtp.process.cooper_jacob_correction(stdy_copy, sat_thickness=4)
        # reference values
        ref = [0.875, 1.5, 1.875]
        # check if correct
        self.assertTrue(np.all(np.isclose(ref, trns_copy.observation)))
        self.assertTrue(np.all(np.isclose(ref, stdy_copy.observation)))


if __name__ == "__main__":
    unittest.main()
