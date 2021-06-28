"""
Diagnostic plot
-------------------

A diagnostic plot is a simultaneous plot of the drawdown and the
logarithmic derivative of the drawdown in a log-log plot.
Often, this plot is used to identify the right approach for the aquifer estimations.

"""


import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
campaign.diagnostic_plot("well_0", "well_1")
