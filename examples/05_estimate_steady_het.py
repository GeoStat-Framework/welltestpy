"""
Estimate steady heterogeneous parameters
----------------------------------------

Here we demonstrate how to estimate parameters of heterogeneity, namely
mean, variance and correlation length of log-transmissivity,
with the aid the the extended Thiem solution in 2D.
"""

import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.ExtThiem2D("Est_steady_het", campaign, generate=True)
estimation.run()
estimation.sensitivity()
