# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.data.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Thiem("Estimate_thiem", campaign, generate=True)
estimation.run()
estimation.gen_setup(dummy=True)
estimation.sensitivity()
