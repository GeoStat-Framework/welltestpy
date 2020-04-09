# -*- coding: utf-8 -*-
"""
welltestpy subpackage providing Spotpy classes for the estimating.

.. currentmodule:: welltestpy.estimate.spotpylib

The following functions and classes are provided

.. autosummary::
   TypeCurve
   fast_rep
"""
import functools as ft

import numpy as np
import spotpy


__all__ = ["TypeCurve", "fast_rep"]


# functions for fitting
FIT = {
    "linear": lambda x: x,
    "lin": lambda x: x,
    "logarithmic": np.exp,
    "log": np.exp,
    "exponential": np.log,
    "exp": np.log,
    "squareroot": lambda x: np.power(x, 2),
    "sqrt": lambda x: np.power(x, 2),
    "quadratic": np.sqrt,
    "quad": np.sqrt,
    "inverse": lambda x: 1.0 / x,
    "inv": lambda x: 1.0 / x,
}


def fast_rep(para_no, infer_fac=4, freq_step=2):
    """Get number of iterations needed for the FAST algorithm.

    Parameters
    ----------
    para_no : :class:`int`
        Number of parameters in the model.
    infer_fac : :class:`int`, optional
        The inference fractor. Default: 4
    freq_step : :class:`int`, optional
        The frequency step. Default: 2
    """
    return 2 * int(
        para_no * (1 + 4 * infer_fac ** 2 * (1 + (para_no - 2) * freq_step))
    )


class TypeCurve:
    r"""Spotpy class for an estimation of subsurface parameters.

    This class fits a given Type Curve to given data.
    Values will be sampled uniformly in given ranges.

    Fitting values will be done linear, logarithmic or by user specified
    function.

    Parameters
    ----------
    type_curve : :any:`callable`
        The given type-curve. Output will be reshaped to flat array.
    data : :class:`numpy.ndarray`
        Observed data as array. Will be reshaped to flat array.
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Ranges should be a tuple containing min and max value.
    val_fix : :class:`dict` or :any:`None`
        Dictionary containing fixed values for the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        Default: None
    fit_type : :class:`dict` or :any:`None`
        Dictionary containing fitting type for each value in the type-curve.
        Names should be as in the type-curve signiture
        or replaced in val_kw_names.
        fit_type can be "lin", "log" (np.exp(val) will be used)
        or a callable function.
        By default, values will be fit linearly.
        Default: None
    val_kw_names : :class:`dict` or :any:`None`
        Dictionary containing keyword names in the type-curve for each value.

            {value-name: kwargs-name in type_curve}

        This is usefull if fitting is not done by linear values.
        By default, parameter names will be value names.
        Default: None
    val_plot_names : :class:`dict` or :any:`None`
        Dictionary containing plotable strings for the parameters.

            {value-name: plotting-string}

        Default: None
    dummy : :class:`bool`, optional
        Add a dummy parameter to the model. This could be used to equalize
        sensitivity analysis.
        Default: False
    """

    def __init__(
        self,
        type_curve,
        data,
        val_ranges,
        val_fix=None,
        fit_type=None,
        val_kw_names=None,
        val_plot_names=None,
        dummy=False,
    ):
        self.func = type_curve
        assert callable(self.func), "type_curve not callable"
        self.fit_type = {} if fit_type is None else fit_type
        self.val_kw_names = {} if val_kw_names is None else val_kw_names
        self.val_plot_names = {} if val_plot_names is None else val_plot_names
        self.val_ranges = val_ranges
        assert self.val_ranges, "No ranges given"
        self.val_fix = {} if val_fix is None else val_fix
        self.val_fix_kw = {}
        for fix in self.val_fix:
            name = self.val_kw_names.get(fix, fix)
            fit_fix = self.fit_type.get(fix, "lin")
            fit_fix = fit_fix if callable(fit_fix) else FIT[fit_fix]
            self.val_fix_kw[name] = fit_fix(self.val_fix[fix])
        # if values haven given ranges but should be fixed, remove ranges
        for inter in set(self.val_ranges) & set(self.val_fix):
            del self.val_ranges[inter]

        self.para_names = list(val_ranges)
        self.para_dist = []
        self.data = np.array(data, dtype=float).reshape(-1)
        self.sim_kwargs = {}
        self.fit_func = {}

        for val in self.para_names:
            self.para_dist.append(
                spotpy.parameter.Uniform(val, *self.val_ranges[val])
            )
            fit_t = self.fit_type.get(val, "lin")
            self.fit_func[val] = fit_t if callable(fit_t) else FIT[fit_t]
            self.val_kw_names.setdefault(val, val)
            self.val_plot_names.setdefault(val, val)

        self.dummy = dummy
        if self.dummy:
            self.para_dist.append(spotpy.parameter.Uniform("dummy", 0, 1))

        self.sim = ft.partial(self.func, **self.val_fix_kw)

    def get_sim_kwargs(self, vector):
        """Generate keyword-args for simulation."""
        # if there is a dummy parameter it will be skipped automatically
        for i, para in enumerate(self.para_names):
            self.sim_kwargs[self.val_kw_names[para]] = self.fit_func[para](
                vector[i]
            )
        return self.sim_kwargs

    def parameters(self):
        """Generate a set of parameters."""
        return spotpy.parameter.generate(self.para_dist)

    def simulation(self, vector):
        """Run a simulation with the given parameters."""
        self.get_sim_kwargs(vector)
        return self.sim(**self.sim_kwargs).reshape(-1)

    def evaluation(self):
        """Accesss the observation data."""
        return self.data

    def objectivefunction(self, simulation, evaluation):
        """Calculate RMSE between observation and simulation."""
        return spotpy.objectivefunctions.rmse(
            evaluation=evaluation, simulation=simulation
        )
