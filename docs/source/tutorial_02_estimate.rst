Tutorial 2: Estimate transmissivity and storativity
===================================================

The pumping test from example 1 can now be loaded and used to estimate the values for
transmissivity and storativity.


.. code-block:: python

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

This will give the following plots:

.. image:: pics/02_fit.png
   :width: 400px
   :align: center

.. image:: pics/02_paratrace.png
   :width: 400px
   :align: center

.. image:: pics/02_parainter.png
   :width: 400px
   :align: center

The results are:

* `ln(T) = -9.22` which is equivalent to `T = 0.99 * 10^-4 m^2/s`
* `ln(S) = -9.10` which is equivalent to `S = 1.11 * 10^-4`
