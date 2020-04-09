# Welcome to welltestpy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1229051.svg)](https://doi.org/10.5281/zenodo.1229051)
[![PyPI version](https://badge.fury.io/py/welltestpy.svg)](https://badge.fury.io/py/welltestpy)
[![Build Status](https://travis-ci.com/GeoStat-Framework/welltestpy.svg?branch=master)](https://travis-ci.com/GeoStat-Framework/welltestpy)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/welltestpy/badge.svg?branch=master)](https://coveralls.io/github/GeoStat-Framework/welltestpy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/welltestpy/badge/?version=stable)](https://geostat-framework.readthedocs.io/projects/welltestpy/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/WTP.png" alt="welltestpy-LOGO" width="251px"/>
</p>

## Purpose

welltestpy provides a framework to handle, process, plot and analyse data from well based field campaigns.


## Installation

You can install the latest version with the following command:

    pip install welltestpy


## Documentation for welltestpy

You can find the documentation under [geostat-framework.readthedocs.io][doc_link].


### Example 1: A campaign containing a pumping test

In the following, we will take a look at an artificial pumping test campaign,
that is stored in a file called `Cmp_UFZ-campaign.cmp`.

```python
import welltestpy as wtp

# load the campaign
campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")

# plot the well constellation and a test overview
campaign.plot_wells()
campaign.plot()
```

#### This will give the following plots:

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/01_wells.png" alt="Wells" width="600px"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/01_pumptest.png" alt="Pumptest" width="600px"/>
</p>


### Example 2: Estimate transmissivity and storativity

The pumping test from example 1 can now be loaded and used to estimate the values for
transmissivity and storativity.

```python
import welltestpy as wtp

campaign = wtp.load_campaign("Cmp_UFZ-campaign.cmp")
estimation = wtp.estimate.Theis("Estimate_theis", campaign, generate=True)
estimation.run()
```

#### This will give the following plots:

Type-Curve fitting:

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/02_fit.png" alt="Fit" width="600px"/>
</p>

Evolution of parameter estimation with SCE:

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/02_paratrace.png" alt="Trace" width="600px"/>
</p>

Scatterplot of paramter distribution during estimation:

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/master/docs/source/pics/02_parainter.png" alt="Interaction" width="600px"/>
</p>

The results are:

* `ln(T) = -9.22` which is equivalent to `T = 0.99 * 10^-4 m^2/s`
* `ln(S) = -9.10` which is equivalent to `S = 1.11 * 10^-4`


### Provided Subpackages

```python
welltestpy.data      # Subpackage to handle data from field campaigns
welltestpy.estimate  # Subpackage to estimate field parameters
welltestpy.process   # Subpackage to pre- and post-process data
welltestpy.tools     # Subpackage with tools for plotting and triagulation
```


## Requirements

- [NumPy >= 1.14.5](https://www.numpy.org)
- [SciPy >= 1.1.0](https://www.scipy.org)
- [Pandas >= 0.23.2](https://pandas.pydata.org)
- [AnaFlow >= 1.0.0](https://github.com/GeoStat-Framework/AnaFlow)
- [SpotPy >= 1.5.0](https://github.com/thouska/spotpy)
- [Matplotlib >= 3.0.0](https://matplotlib.org)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[MIT][license_link] © 2018-2020

[license_link]: https://github.com/GeoStat-Framework/welltestpy/blob/master/LICENSE
[doc_link]: https://welltestpy.readthedocs.io
