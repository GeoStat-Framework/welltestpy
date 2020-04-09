"""
Estimate homogeneous parameters
-------------------------------

Here we estimate transmissivity and storage from a pumping test campaign
with the classical theis solution.
"""

import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Theis("Estimate_theis", campaign, generate=True)
estimation.run()

###############################################################################
# In addition, we run a sensitivity analysis, to get an impression
# of the impact of each parameter

estimation.sensitivity()
