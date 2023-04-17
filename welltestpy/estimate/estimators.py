# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing classes for parameter estimation.

.. currentmodule:: welltestpy.estimate.estimators

The following classes are provided

.. autosummary::
   ExtTheis3D
   ExtTheis2D
   Neuman2004
   Theis
   ExtThiem3D
   ExtThiem2D
   Neuman2004Steady
   Thiem
"""
import anaflow as ana

from . import steady_lib, transient_lib


__all__ = [
    "ExtTheis3D",
    "ExtTheis2D",
    "Neuman2004",
    "Theis",
    "ExtThiem3D",
    "ExtThiem2D",
    "Neuman2004Steady",
    "Thiem",
]


# ext_theis_3D


class ExtTheis3D(transient_lib.TransientPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended theis solution in 3D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function and an anisotropy ratio 0 < e <= 1.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "cond_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
            "storage": (2e-6, 4e-1),
            "anis": (0, 1),
        }
        val_ranges = val_ranges or {}
        val_fix = val_fix or {"lat_ext": 1.0}
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("cond_gmean", "log")
        val_fit_type.setdefault("storage", "log")
        val_plot_names = {
            "cond_gmean": "$K_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
            "storage": "$S$",
            "anis": "$e$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.ext_theis_3d,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# ext_theis_2D


class ExtTheis2D(transient_lib.TransientPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended theis solution in 2D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        paramters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "trans_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
            "storage": (2e-6, 4e-1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("trans_gmean", "log")
        val_fit_type.setdefault("storage", "log")
        val_plot_names = {
            "trans_gmean": "$T_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
            "storage": "$S$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.ext_theis_2d,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# neuman 2004


class Neuman2004(transient_lib.TransientPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the apparent Transmissivity from Neuman 2004
    which assumes a log-normal distributed transmissivity field
    with an exponential correlation function.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "trans_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
            "storage": (2e-6, 4e-1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("trans_gmean", "log")
        val_fit_type.setdefault("storage", "log")
        val_plot_names = {
            "trans_gmean": "$T_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
            "storage": "$S$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.neuman2004,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# theis


class Theis(transient_lib.TransientPumping):
    """Class for an estimation of homogeneous subsurface parameters.

    With this class you can run an estimation of homogeneous subsurface
    parameters. It utilizes the theis solution.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {"transmissivity": (1e-7, 2e-1), "storage": (2e-6, 4e-1)}
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("transmissivity", "log")
        val_fit_type.setdefault("storage", "log")
        val_plot_names = {"transmissivity": "$T$", "storage": "$S$"}
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.theis,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# ext_thiem_3d


class ExtThiem3D(steady_lib.SteadyPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended thiem solution in 3D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function and an anisotropy ratio 0 < e <= 1.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    make_steady : :class:`bool`, optional
        State if the tests should be converted to steady observations.
        See: :any:`PumpingTest.make_steady`.
        Default: True
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        make_steady=True,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "cond_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
            "anis": (0, 1),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        val_fix = {"lat_ext": 1.0} if val_fix is None else val_fix
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("cond_gmean", "log")
        val_plot_names = {
            "cond_gmean": "$K_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
            "anis": "$e$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.ext_thiem_3d,
            val_ranges=val_ranges,
            make_steady=make_steady,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# ext_thiem_2D


class ExtThiem2D(steady_lib.SteadyPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters. It utilizes the extended thiem solution in 2D which assumes
    a log-normal distributed transmissivity field with a gaussian correlation
    function.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    make_steady : :class:`bool`, optional
        State if the tests should be converted to steady observations.
        See: :any:`PumpingTest.make_steady`.
        Default: True
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        make_steady=True,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "trans_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("trans_gmean", "log")
        val_plot_names = {
            "trans_gmean": "$T_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            make_steady=make_steady,
            type_curve=ana.ext_thiem_2d,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# neuman 2004 steady


class Neuman2004Steady(steady_lib.SteadyPumping):
    """Class for an estimation of stochastic subsurface parameters.

    With this class you can run an estimation of statistical subsurface
    parameters from steady drawdown.
    It utilizes the apparent Transmissivity from Neuman 2004
    which assumes a log-normal distributed transmissivity field
    with an exponential correlation function.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    make_steady : :class:`bool`, optional
        State if the tests should be converted to steady observations.
        See: :any:`PumpingTest.make_steady`.
        Default: True
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        make_steady=True,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {
            "trans_gmean": (1e-7, 2e-1),
            "var": (0, 10),
            "len_scale": (1, 50),
        }
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("trans_gmean", "log")
        val_plot_names = {
            "trans_gmean": "$T_G$",
            "var": r"$\sigma^2$",
            "len_scale": r"$\ell$",
        }
        super().__init__(
            name=name,
            campaign=campaign,
            make_steady=make_steady,
            type_curve=ana.neuman2004_steady,
            val_ranges=val_ranges,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )


# thiem


class Thiem(steady_lib.SteadyPumping):
    """Class for an estimation of homogeneous subsurface parameters.

    With this class you can run an estimation of homogeneous subsurface
    parameters. It utilizes the thiem solution.

    Parameters
    ----------
    name : :class:`str`
        Name of the Estimation.
    campaign : :class:`welltestpy.data.Campaign`
        The pumping test campaign which should be used to estimate the
        parameters
    make_steady : :class:`bool`, optional
        State if the tests should be converted to steady observations.
        See: :any:`PumpingTest.make_steady`.
        Default: True
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signature.
        Default: None
    val_fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting transformation type for each value.
        Names should be as in the type-curve signature.
        val_fit_type can be "lin", "log", "exp", "sqrt", "quad", "inv"
        or a tuple of two callable functions where the
        first is the transformation and the second is its inverse.
        "log" is for example equivalent to ``(np.log, np.exp)``.
        By default, values will be fitted linear.
        Default: None
    val_fit_name : :class:`dict` or :any:`None`
        Display name of the fitting transformation.
        Will be the val_fit_type string if it is a predefined one,
        or ``f`` if it is a given callable as default for each value.
        Default: None
    testinclude : :class:`dict`, optional
        Dictionary of which tests should be included. If ``None`` is given,
        all available tests are included.
        Default: ``None``
    generate : :class:`bool`, optional
        State if time stepping, processed observation data and estimation
        setup should be generated with default values.
        Default: ``False``
    """

    def __init__(
        self,
        name,
        campaign,
        make_steady=True,
        val_ranges=None,
        val_fix=None,
        val_fit_type=None,
        val_fit_name=None,
        testinclude=None,
        generate=False,
    ):
        def_ranges = {"transmissivity": (1e-7, 2e-1)}
        val_ranges = {} if val_ranges is None else val_ranges
        for def_name, def_val in def_ranges.items():
            val_ranges.setdefault(def_name, def_val)
        val_fit_type = val_fit_type or {}
        val_fit_type.setdefault("transmissivity", "log")
        val_plot_names = {"transmissivity": "$T$"}
        super().__init__(
            name=name,
            campaign=campaign,
            type_curve=ana.thiem,
            val_ranges=val_ranges,
            make_steady=make_steady,
            val_fix=val_fix,
            val_fit_type=val_fit_type,
            val_fit_name=val_fit_name,
            val_plot_names=val_plot_names,
            testinclude=testinclude,
            generate=generate,
        )
