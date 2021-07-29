# Welcome to welltestpy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1229051.svg)](https://doi.org/10.5281/zenodo.1229051)
[![PyPI version](https://badge.fury.io/py/welltestpy.svg)](https://badge.fury.io/py/welltestpy)
[![Build Status](https://github.com/GeoStat-Framework/welltestpy/workflows/Continuous%20Integration/badge.svg?branch=main)](https://github.com/GeoStat-Framework/welltestpy/actions)
[![Coverage Status](https://coveralls.io/repos/github/GeoStat-Framework/welltestpy/badge.svg?branch=main)](https://coveralls.io/github/GeoStat-Framework/welltestpy?branch=main)
[![Documentation Status](https://readthedocs.org/projects/welltestpy/badge/?version=latest)](https://geostat-framework.readthedocs.io/projects/welltestpy/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/welltestpy/main/docs/source/pics/WTP.png" alt="welltestpy-LOGO" width="251px"/>
</p>

## Purpose

welltestpy provides a framework to handle, process, plot and analyse data from well based field campaigns.


## Installation

You can install the latest version with the following command:

    pip install welltestpy

Or from conda

    conda install -c conda-forge welltestpy


## Documentation for welltestpy

You can find the documentation including tutorials and examples under
https://welltestpy.readthedocs.io.


## Citing welltestpy

If you are using this package you can cite our
[Groundwater publication](https://doi.org/10.1111/gwat.13121) by:

> Müller, S., Leven, C., Dietrich, P., Attinger, S. and Zech, A. (2021):
> How to Find Aquifer Statistics Utilizing Pumping Tests? Two Field Studies Using welltestpy.
> Groundwater, https://doi.org/10.1111/gwat.13121

To cite the code, please visit the [Zenodo page](https://doi.org/10.5281/zenodo.1229051).


## Provided Subpackages

```python
welltestpy.data      # Subpackage to handle data from field campaigns
welltestpy.estimate  # Subpackage to estimate field parameters
welltestpy.process   # Subpackage to pre- and post-process data
welltestpy.tools     # Subpackage with tools for plotting and triagulation
```


## Requirements

- [NumPy >= 1.14.5](https://www.numpy.org)
- [SciPy >= 1.1.0](https://www.scipy.org)
- [AnaFlow >= 1.0.0](https://github.com/GeoStat-Framework/AnaFlow)
- [SpotPy >= 1.5.0](https://github.com/thouska/spotpy)
- [Matplotlib >= 3.0.0](https://matplotlib.org)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[MIT](https://github.com/GeoStat-Framework/welltestpy/blob/main/LICENSE)
