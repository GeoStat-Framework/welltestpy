"""
Creating a pumping test campaign
--------------------------------

In the following we are going to create an artificial pumping test campaign
on a field site.
"""

import numpy as np
import welltestpy as wtp
import anaflow as ana


###############################################################################
# Create the field-site and the campaign

field = wtp.FieldSite(name="UFZ", coordinates=[51.353839, 12.431385])
campaign = wtp.Campaign(name="UFZ-campaign", fieldsite=field)

###############################################################################
# Add 4 wells to the campaign

campaign.add_well(name="well_0", radius=0.1, coordinates=(0.0, 0.0))
campaign.add_well(name="well_1", radius=0.1, coordinates=(1.0, -1.0))
campaign.add_well(name="well_2", radius=0.1, coordinates=(2.0, 2.0))
campaign.add_well(name="well_3", radius=0.1, coordinates=(-2.0, -1.0))

###############################################################################
# Generate artificial drawdown data with the Theis solution

rate = -1e-4
time = np.geomspace(10, 7200, 10)
transmissivity = 1e-4
storage = 1e-4
rad = [
    campaign.wells["well_0"].radius,  # well radius of well_0
    campaign.wells["well_0"] - campaign.wells["well_1"],  # distance 0-1
    campaign.wells["well_0"] - campaign.wells["well_2"],  # distance 0-2
    campaign.wells["well_0"] - campaign.wells["well_3"],  # distance 0-3
]
drawdown = ana.theis(
    time=time,
    rad=rad,
    storage=storage,
    transmissivity=transmissivity,
    rate=rate,
)

###############################################################################
# Create a pumping test at well_0

pumptest = wtp.PumpingTest(
    name="well_0",
    pumpingwell="well_0",
    pumpingrate=rate,
    description="Artificial pump test with Theis",
)

###############################################################################
# Add the drawdown observation at the 4 wells

pumptest.add_transient_obs("well_0", time, drawdown[:, 0])
pumptest.add_transient_obs("well_1", time, drawdown[:, 1])
pumptest.add_transient_obs("well_2", time, drawdown[:, 2])
pumptest.add_transient_obs("well_3", time, drawdown[:, 3])

###############################################################################
# Add the pumping test to the campaign

campaign.addtests(pumptest)
# optionally make the test (quasi)steady
# campaign.tests["well_0"].make_steady()

###############################################################################
# Plot the well constellation and a test overview
campaign.plot_wells()
campaign.plot()

###############################################################################
# Save the whole campaign to a file

campaign.save()
