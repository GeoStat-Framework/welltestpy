# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.data.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Theisest("Estimate_theis", campaign)
estimation.setpumprate()
estimation.settime()
estimation.genrtdata()
estimation.run(
    dbname="database",
    plotname1="paratrace.pdf",
    plotname2="fit_plot.pdf",
    plotname3="parainteract.pdf",
    estname="estimation.txt",
)
