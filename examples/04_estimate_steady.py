# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Thiem("Estimate_thiem", campaign, generate=True)
estimation.run()
# since we only have one parameter,
# we need a dummy parameter to estimate sensitivity
estimation.gen_setup(dummy=True)
estimation.sensitivity()
