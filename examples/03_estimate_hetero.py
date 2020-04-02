# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.ExtTheis2D("Estimate_het2D", campaign, generate=True)
estimation.run()
estimation.sensitivity()
