# -*- coding: utf-8 -*-
import welltestpy as wtp

campaign = wtp.data.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Stat2Dest("Estimate_theis", campaign)
estimation.setpumprate()
estimation.settime()
estimation.genrtdata()
estimation.run(
    dbname="database_het",
    plotname1="paratrace_het.pdf",
    plotname2="fit_plot_het.pdf",
    plotname3="parainteract_het.pdf",
    estname="estimation_het.txt",
)
