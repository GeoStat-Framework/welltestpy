"""
Estimate heterogeneous parameters
---------------------------------

Here we demonstrate how to estimate parameters of heterogeneity, namely
mean, variance and correlation length of log-transmissivity, as well as the
storage with the aid the the extended Theis solution in 2D.
"""

import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.ExtTheis2D("Estimate_het2D", campaign, generate=True)
estimation.run()
estimation.sensitivity()
