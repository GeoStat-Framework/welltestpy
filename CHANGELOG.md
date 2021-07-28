# Changelog

All notable changes to **welltestpy** will be documented in this file.

## [Unreleased]

### Enhancements
- added `cooper_jacob_correction` to `process` (thanks to Jarno Herrmann)
- added `diagnostic_plots` module (thanks to Jarno Herrmann)
- added `screensize`, `screen`, `aquifer` and `is_piezometer` attribute to `Well` class
- added version information to output files
- added `__repr__` to `Campaign`

### Changes
- modernized packaging workflow using `pyproject.toml`
- removed `setup.py` (use `pip>21.1` for editable installs)
- removed `dev` as extra install dependencies
- better exceptions in loading routines

### Bugfixes
- loading steady pumping tests was not possible due to a bug


## [1.0.3] - 2021-02

### Enhancements
- Estimations: run method now provides `plot_style` keyword to control plotting

### Changes
- Fit plot style for transient pumping tests was updated

### Bugfixes
- Estimations: run method was throwing an Error when setting `run=False`
- Plotter: all plotting routines now respect setted font-type from matplotlib


## [1.0.2] - 2020-09-03

### Bugfixes
- `StdyHeadObs` and `StdyObs` weren't usable due to an unnecessary `time` check


## [1.0.1] - 2020-04-09

### Bugfixes
- Wrong URL in setup


## [1.0.0] - 2020-04-09

### Enhancements
- new estimators
  - ExtTheis3D
  - ExtTheis2D
  - Neuman2004
  - Theis
  - ExtThiem3D
  - ExtThiem2D
  - Neuman2004Steady
  - Thiem
- better plotting
- unit-tests run with py35-py38 on Linux/Win/Mac
- coverage calculation
- sphinx gallery for examples
- allow style setting in plotting routines

### Bugfixes
- estimation results stored as dict (order could alter before)

### Changes
- py2 support dropped
- `Fieldsite.coordinates` now returns a `Variable`; `Fieldsite.pos` as shortcut
- `Fieldsite.pumpingrate` now returns a `Variable`; `Fieldsite.rate` as shortcut
- `Fieldsite.auqiferradius` now returns a `Variable`; `Fieldsite.radius` as shortcut
- `Fieldsite.auqiferdepth` now returns a `Variable`; `Fieldsite.depth` as shortcut
- `Well.coordinates` now returns a `Variable`; `Well.pos` as shortcut
- `Well.welldepth` now returns a `Variable`; `Well.depth` as shortcut
- `Well.wellradius` added and returns the radius `Variable`
- `Well.aquiferdepth` now returns a `Variable`
- `Fieldsite.addobservations` renamed to `Fieldsite.add_observations`
- `Fieldsite.delobservations` renamed to `Fieldsite.del_observations`
- `Observation` has changed order of inputs/outputs. Now: `observation`, `time`


## [0.3.2] - 2019-03-08

### Bugfixes
- adopt AnaFlow API


## [0.3.1] - 2019-03-08

### Bugfixes
- update travis workflow


## [0.3.0] - 2019-02-28

### Enhancements
- added documentation


## [0.2.0] - 2018-04-25

### Enhancements
- added license


## [0.1.0] - 2018-04-25

First alpha release of welltespy.

[Unreleased]: https://github.com/GeoStat-Framework/welltestpy/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/GeoStat-Framework/welltestpy/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/GeoStat-Framework/welltestpy/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/GeoStat-Framework/welltestpy/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GeoStat-Framework/welltestpy/compare/v0.3.2...v1.0.0
[0.3.2]: https://github.com/GeoStat-Framework/welltestpy/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/GeoStat-Framework/welltestpy/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/GeoStat-Framework/welltestpy/compare/v0.2...v0.3.0
[0.2.0]: https://github.com/GeoStat-Framework/welltestpy/compare/v0.1...v0.2
[0.1.0]: https://github.com/GeoStat-Framework/welltestpy/releases/tag/v0.1
