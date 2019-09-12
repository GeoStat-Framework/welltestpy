# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.data.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.ExtThiem2D("Est_steady_het", campaign, generate=True)
estimation.run()
estimation.sensitivity()
