=====================
WellTestPy Quickstart
=====================

.. image:: pics/WTP.png
   :width: 150px
   :align: center

WellTestPy provides a framework to handle and plot data from well based field campaigns as well as a data interpretation module.


Installation
============

The package can be installed via `pip <https://pypi.org/project/welltestpy/>`_.
On Windows you can install `WinPython <https://winpython.github.io/>`_ to get
Python and pip running.

.. code-block:: none

    pip install welltestpy


Provided Subpackages
====================

The following functions are provided directly

.. code-block:: python

    welltestpy.data      # Subpackage to handle data from field campaigns
    welltestpy.estimate  # Subpackage to estimate field parameters
    welltestpy.process   # Subpackage to pre- and post-process data
    welltestpy.tools     # Subpackage with miscellaneous tools


Requirements
============

- `NumPy >= 1.13.0 <https://www.numpy.org>`_
- `SciPy >= 0.19.1 <https://www.scipy.org>`_
- `AnaFlow <https://github.com/GeoStat-Framework/AnaFlow>`_
- `Matplotlib <https://matplotlib.org>`_
- `Pandas <https://pandas.pydata.org>`_
- `SpotPy <https://github.com/thouska/spotpy>`_



License
=======

`GPL <https://github.com/GeoStat-Framework/welltestpy/blob/master/LICENSE>`_ © 2019
