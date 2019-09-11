# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.data.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Theis("Estimate_theis", campaign, generate=True)
estimation.run()
