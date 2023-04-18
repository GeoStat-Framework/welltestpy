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


def _quad(x):
    return np.power(x, 2)


def _inv(x):
    return 1.0 / x


def _lin(x):
    return x


FIT = {
    "lin": (_lin, _lin),
    "log": (np.log, np.exp),
    "exp": (np.exp, np.log),
    "sqrt": (np.sqrt, _quad),
    "quad": (_quad, np.sqrt),
    "inv": (_inv, _inv),
}
"""dict: all predefined fitting transformations and their inverse."""


def _is_callable_tuple(input):
    result = False
    length = 0
    try:
        length = len(input)
    except TypeError:
        length = -1
    finally:
        if length == 2:
            result = all(map(callable, input))
    return result


def fast_rep(para_no, infer_fac=4, freq_step=2):
    """Get number of iterations needed for the FAST algorithm.

    Parameters
    ----------
    para_no : :class:`int`
        Number of parameters in the model.
    infer_fac : :class:`int`, optional
        The inference factor. Default: 4
    freq_step : :class:`int`, optional
        The frequency step. Default: 2
    """
    return 2 * int(
        para_no * (1 + 4 * infer_fac**2 * (1 + (para_no - 2) * freq_step))
    )


class TypeCurve:
    r"""Spotpy class for an estimation of subsurface parameters.

    This class fits a given Type Curve to given data.
    Values will be sampled uniformly in given ranges and with given transformation.

    Parameters
    ----------
    type_curve : :any:`callable`
        The given type-curve. Output will be reshaped to flat array.
    data : :class:`numpy.ndarray`
        Observed data as array. Will be reshaped to flat array.
    val_ranges : :class:`dict`
        Dictionary containing the fit-ranges for each value in the type-curve.
        Names should be as in the type-curve signature.
        All values to be estimated should be present here.
        Ranges should be a tuple containing min and max value: ``(min, max)``.
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
        val_fit_type=None,
        val_fit_name=None,
        val_plot_names=None,
        dummy=False,
    ):
        self.func = type_curve
        if not callable(self.func):
            raise ValueError("type_curve not callable")
        self.val_fit_type = val_fit_type or {}
        self.val_plot_names = val_plot_names or {}
        if not isinstance(val_ranges, dict) or not val_ranges:
            raise ValueError("No ranges given")
        self.val_ranges = val_ranges.copy()
        self.val_fix = val_fix or {}
        # if values haven given ranges but should be fixed, remove ranges
        for inter in set(self.val_ranges) & set(self.val_fix):
            del self.val_ranges[inter]

        self.para_names = list(self.val_ranges)
        self.para_dist = []
        self.data = np.array(data, dtype=float).reshape(-1)
        self.sim_kwargs = {}
        self.fit_func = {}
        self.val_fit_name = val_fit_name or {}
        for val in self.para_names:
            # linear fitting by default
            fit_t = self.val_fit_type.get(val, "lin")
            fit_n = fit_t if fit_t in FIT else "f"
            self.val_fit_name.setdefault(
                val, fit_n if fit_n != "lin" else None
            )
            self.fit_func[val] = (
                fit_t if _is_callable_tuple(fit_t) else FIT.get(fit_t, None)
            )
            if not self.fit_func[val]:
                raise ValueError(f"Fitting transformation for '{val}' missing")
            # apply fitting transformation to ranges
            self.para_dist.append(
                spotpy.parameter.Uniform(
                    val, *map(self.fit_func[val][0], self.val_ranges[val])
                )
            )
            self.val_plot_names.setdefault(val, val)

        self.dummy = dummy
        if self.dummy:
            self.para_dist.append(spotpy.parameter.Uniform("dummy", 0, 1))

        self.sim = ft.partial(self.func, **self.val_fix)

    def get_sim_kwargs(self, vector):
        """Generate keyword-args for simulation."""
        # if there is a dummy parameter it will be skipped automatically
        for i, para in enumerate(self.para_names):
            self.sim_kwargs[para] = self.fit_func[para][1](vector[i])
        return self.sim_kwargs

    def parameters(self):
        """Generate a set of parameters."""
        return spotpy.parameter.generate(self.para_dist)

    def simulation(self, vector):
        """Run a simulation with the given parameters."""
        self.get_sim_kwargs(vector)
        return self.sim(**self.sim_kwargs).reshape(-1)

    def evaluation(self):
        """Accesses the observation data."""
        return self.data

    def objectivefunction(self, simulation, evaluation):
        """Calculate RMSE between observation and simulation."""
        return spotpy.objectivefunctions.rmse(
            evaluation=evaluation, simulation=simulation
        )
