"""Helper utilities used by the Jumper notebook."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts


@dataclass(frozen=True)
class TermSpec:
    """Specification for one ocean term handled in the notebook pipeline.

    Attributes
    ----------
    key : str
        Short key used as the dictionary identifier for this term.
    term : str
        Ocean variable name consumed by loaders and forecasting classes.
    filename : str
        File name pattern used to select matching simulation files.
    mean_axes : tuple[int, ...]
        Axes used in notebook analysis to compute mean prediction/reference
        profiles.
    err_axes : tuple[int, ...]
        Axes used in notebook analysis to reduce absolute-error arrays into
        summary statistics.
    """

    key: str
    term: str
    filename: str
    mean_axes: tuple[int, ...]
    err_axes: tuple[int, ...]


def build_simulations(
    specs: Sequence[TermSpec],
    *,
    data_path: str,
    start: int,
    end: int,
    comp: Any,
    ye: bool,
    dr_method: Any,
    stand: bool = True,
) -> dict[str, Simulation]:
    """Initialize and prepare simulations for all configured terms.

    Parameters
    ----------
    specs : Sequence[TermSpec]
        Specifications defining each term to load and prepare.
    data_path : str
        Root path containing simulation files.
    start : int
        Start index used when slicing simulation data.
    end : int
        End index used when slicing simulation data.
    comp : Any
        Dimensionality-reduction component configuration forwarded to
        :class:`~nemo_spinup_forecast.forecast.Simulation`.
    ye : bool
        Whether yearly processing is enabled for each simulation.
    dr_method : Any
        Dimensionality-reduction class or factory passed to
        :class:`~nemo_spinup_forecast.forecast.Simulation`.
    stand : bool, default=True
        Whether simulation data should be standardized during preparation.

    Returns
    -------
    dict[str, Simulation]
        Prepared simulation instances keyed by :attr:`TermSpec.key`.

    Notes
    -----
    This function calls :meth:`Simulation.get_simulation_data` for each term
    and prints one progress message per entry.
    """
    sims: dict[str, Simulation] = {}
    for spec in specs:
        s = Simulation(
            path=data_path,
            start=start,
            end=end,
            comp=comp,
            ye=ye,
            term=spec.term,
            filename=spec.filename,
            dimensionality_reduction=dr_method,
        )
        s.get_simulation_data(stand=stand)
        sims[spec.key] = s
        print(f"{spec.key} loaded & prepared (stand={stand})")
    return sims


def decompose_all(sims: Mapping[str, Simulation]) -> None:
    """Run dimensionality reduction for each simulation.

    Parameters
    ----------
    sims : Mapping[str, Simulation]
        Simulation objects keyed by term identifier.

    Returns
    -------
    None
        Simulations are updated in place.

    Notes
    -----
    Calls :meth:`Simulation.decompose` and prints one progress message per key.
    """
    for k, s in sims.items():
        s.decompose()
        print(f"Decomposition applied on {k}")


def compute_rmse_for_terms(
    specs: Sequence[TermSpec],
    sims: Mapping[str, Simulation],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute reconstruction error outputs for each configured term.

    Parameters
    ----------
    specs : Sequence[TermSpec]
        Term specifications defining the processing order and output keys.
    sims : Mapping[str, Simulation]
        Prepared and decomposed simulations keyed by :attr:`TermSpec.key`.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Tuple ``(recs, rmseVs, rmseMs)`` where each dictionary is keyed by
        :attr:`TermSpec.key`.

    Notes
    -----
    Uses all fitted components via ``len(s.pca.components_)`` before calling
    :meth:`Simulation.error`.
    """
    recs: dict[str, Any] = {}
    rmseVs: dict[str, Any] = {}
    rmseMs: dict[str, Any] = {}
    for spec in specs:
        s = sims[spec.key]
        n = len(s.pca.components_)
        rec, rmseV, rmseM = s.error(n)
        recs[spec.key] = rec
        rmseVs[spec.key] = rmseV
        rmseMs[spec.key] = rmseM
        print(f"RMSE computed for {spec.key}")
    return recs, rmseVs, rmseMs


def make_dicos(sims: Mapping[str, Simulation]) -> dict[str, dict[str, Any]]:
    """Create serialized simulation dictionaries for all terms.

    Parameters
    ----------
    sims : Mapping[str, Simulation]
        Simulation objects keyed by term identifier.

    Returns
    -------
    dict[str, dict[str, Any]]
        Serialized simulation payloads produced by :meth:`Simulation.make_dico`,
        keyed by term.

    Notes
    -----
    Prints one progress message per processed key.
    """
    d: dict[str, dict[str, Any]] = {}
    for k, s in sims.items():
        d[k] = s.make_dico()
        print(f"{k} to dictionary")
    return d


def load_ts_all(
    prepared_path: str,
    specs: Sequence[TermSpec],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Load all prepared time-series DataFrames and metadata dictionaries.

    Parameters
    ----------
    prepared_path : str
        Directory containing prepared ``.npz`` and PCA files.
    specs : Sequence[TermSpec]
        Term specifications defining which prepared terms to load.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], dict[str, dict[str, Any]]]
        Tuple ``(dfs, infos)`` keyed by :attr:`TermSpec.key`.

    Notes
    -----
    Each term is loaded via :func:`~nemo_spinup_forecast.forecast.load_ts` using
    ``(prepared_path, spec.term)``.
    """
    dfs: dict[str, pd.DataFrame] = {}
    infos: dict[str, dict[str, Any]] = {}
    for spec in specs:
        df, info = load_ts(prepared_path, spec.term)
        dfs[spec.key] = df
        infos[spec.key] = info
    return dfs, infos


def build_predictions(
    specs: Sequence[TermSpec],
    dfs: Mapping[str, pd.DataFrame],
    infos: Mapping[str, dict[str, Any]],
    forecast_method: Any,
    dr_method: Any,
) -> dict[str, Predictions]:
    """Construct prediction objects for each configured term.

    Parameters
    ----------
    specs : Sequence[TermSpec]
        Term specifications defining output keys and forecasted variables.
    dfs : Mapping[str, pandas.DataFrame]
        Time-series component DataFrames keyed by :attr:`TermSpec.key`.
    infos : Mapping[str, dict[str, Any]]
        Metadata dictionaries keyed by :attr:`TermSpec.key`.
    forecast_method : Any
        Forecasting method instance passed to :class:`Predictions`.
    dr_method : Any
        Dimensionality-reduction method passed to :class:`Predictions`.

    Returns
    -------
    dict[str, Predictions]
        Prediction objects keyed by :attr:`TermSpec.key`.
    """
    preds: dict[str, Predictions] = {}
    for spec in specs:
        preds[spec.key] = Predictions(
            spec.term, dfs[spec.key], infos[spec.key], forecast_method, dr_method
        )
    return preds


def parallel_forecast_all(
    specs: Sequence[TermSpec],
    preds: Mapping[str, Predictions],
    dfs: Mapping[str, pd.DataFrame],
    *,
    train_len: int,
    steps: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Run parallel forecasts for all terms and return outputs by key.

    Parameters
    ----------
    specs : Sequence[TermSpec]
        Term specifications defining processing order and output keys.
    preds : Mapping[str, Predictions]
        Prediction objects keyed by :attr:`TermSpec.key`.
    dfs : Mapping[str, pandas.DataFrame]
        Original component time-series DataFrames keyed by term.
    train_len : int
        Number of initial rows used as the training window.
    steps : int
        Forecast horizon in time steps.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], dict[str, Any], dict[str, Any]]
        Tuple ``(hats, hat_stds, metrics)`` keyed by :attr:`TermSpec.key`.

    Notes
    -----
    For each term, the function prepends ``dfs[key][:train_len]`` to the
    forecast output from :meth:`Predictions.parallel_forecast`.
    """
    hats: dict[str, pd.DataFrame] = {}
    hat_stds: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for spec in specs:
        # Forecast each time series component for each property
        hat, hat_std, m = preds[spec.key].parallel_forecast(train_len, steps)
        # Concatenate the forecasted time series period with the reference traning period
        hat = pd.concat([dfs[spec.key][:train_len], hat[:]])
        hats[spec.key] = hat
        hat_stds[spec.key] = hat_std
        metrics[spec.key] = m
    return hats, hat_stds, metrics


def abs_error_stats(
    err: np.ndarray,
    *,
    pred_steps: int,
    ref_cut: int,
    axes: tuple[int, ...],
) -> dict[str, Any]:
    """Compute absolute-error summary statistics for prediction and reference windows.

    Parameters
    ----------
    err : numpy.ndarray
        Absolute-error array, typically ``abs(reference - prediction)``.
    pred_steps : int
        Number of trailing time steps considered as the prediction window.
    ref_cut : int
        Number of trailing time steps excluded from the reference window.
        If set to ``0``, the full ``err`` array is used as the reference.
    axes : tuple[int, ...]
        Axes reduced with ``nanmean`` and ``nanstd``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``pred_mean``, ``pred_std``, ``ref_mean``,
        and ``ref_std``.

    Notes
    -----
    This function prints the prediction and reference window shapes.
    """
    pred = err[-pred_steps:]
    ref = err[:-ref_cut] if ref_cut else err
    print("pred shape:", pred.shape)
    print("ref shape:", ref.shape)
    return dict(
        pred_mean=np.nanmean(pred, axis=axes),
        pred_std=np.nanstd(pred, axis=axes),
        ref_mean=np.nanmean(ref, axis=axes),
        ref_std=np.nanstd(ref, axis=axes),
    )


def normalise_time_series(sim: Simulation) -> None:
    """Normalize a simulation time series in place.

    Parameters
    ----------
    sim : Simulation
        Simulation object whose ``simulation`` array will be normalized.

    Returns
    -------
    None
        The normalization is applied in place on ``sim.simulation``.

    Notes
    -----
    Uses ``sim.desc["mean"]`` and ``sim.desc["std"]`` for scaling.
    """
    sim.simulation = (sim.simulation - sim.desc["mean"]) / (2 * sim.desc["std"])
