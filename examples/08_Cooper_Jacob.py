"""
Correcting drawdown : The Cooper-Jacob method
---------------------------------------------
Here we demonstrate the correction established by Cooper and Jacob in 1946.
This method corrects drawdown data for the reduction in saturated thickness
resulting from groundwater withdrawal by a pumping well and thereby enables
pumping tests in an unconfined aquifer to be interpreted by methods for
confined aquifers.
"""

import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
campaign.tests["well_0"].correct_observations()
campaign.plot()
