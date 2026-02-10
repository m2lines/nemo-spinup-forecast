"""Helper utilities used by the Jumper notebook.

This notebook often runs the same workflow for multiple NEMO terms (e.g. SSH, Salinity,
Temperature). The small wrappers below keep the notebook readable by turning repeated
multi-term blocks into short, explicit loops.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts


@dataclass(frozen=True)
class TermSpec:
    key: str
    term: Any
    filename: str
    mean_axes: Tuple[int, ...]
    err_axes: Tuple[int, ...]


def build_simulations(
    specs: List[TermSpec],
    *,
    data_path: str,
    start: int,
    end: int,
    comp: Any,
    ye: bool,
    dr_method: Any,
    stand: bool = True,
) -> Dict[str, Simulation]:
    """Initialise + load `Simulation` objects for all terms."""
    sims: Dict[str, Simulation] = {}
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


def decompose_all(sims: Dict[str, Simulation]) -> None:
    """Run dimensionality reduction (e.g. PCA) for each `Simulation`."""
    for k, s in sims.items():
        s.decompose()
        print(f"Decomposition applied on {k}")


def compute_rmse_for_terms(specs: List[TermSpec], sims: Dict[str, Simulation]):
    """Compute an error metric (e.g., RMSE) between reconstructions and truth.

    Returns
    -------
    recs, rmseVs, rmseMs : dict
        Dictionaries keyed by `TermSpec.key`.
    """
    recs: Dict[str, Any] = {}
    rmseVs: Dict[str, Any] = {}
    rmseMs: Dict[str, Any] = {}
    for spec in specs:
        s = sims[spec.key]
        n = len(s.pca.components_)
        rec, rmseV, rmseM = s.error(n)
        recs[spec.key] = rec
        rmseVs[spec.key] = rmseV
        rmseMs[spec.key] = rmseM
        print(f"RMSE computed for {spec.key}")
    return recs, rmseVs, rmseMs


def make_dicos(sims: Dict[str, Simulation]) -> Dict[str, dict]:
    """Create a dictionary containing simulation data, statistics and information"""
    d: Dict[str, dict] = {}
    for k, s in sims.items():
        d[k] = s.make_dico()
        print(f"{k} to dictionary")
    return d


def load_ts_all(prepared_path: str, specs: List[TermSpec]):
    """Load all time-series DataFrames + info dicts from a prepared run directory."""
    dfs: Dict[str, pd.DataFrame] = {}
    infos: Dict[str, dict] = {}
    for spec in specs:
        df, info = load_ts(prepared_path, spec.term)
        dfs[spec.key] = df
        infos[spec.key] = info
    return dfs, infos


def build_predictions(
    specs: List[TermSpec],
    dfs: Dict[str, pd.DataFrame],
    infos: Dict[str, dict],
    forecast_method: Any,
    dr_method: Any,
) -> Dict[str, Predictions]:
    """Construct `Predictions` objects for each term."""
    preds: Dict[str, Predictions] = {}
    for spec in specs:
        preds[spec.key] = Predictions(
            spec.term, dfs[spec.key], infos[spec.key], forecast_method, dr_method
        )
    return preds


def parallel_forecast_all(
    specs: List[TermSpec],
    preds: Dict[str, Predictions],
    dfs: Dict[str, pd.DataFrame],
    *,
    train_len: int,
    steps: int,
):
    """Run `parallel_forecast` for all terms and return predictions + metrics.

    This also concatenates the forecasted time series period with the reference training period.
    """
    hats: Dict[str, pd.DataFrame] = {}
    hat_stds: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
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
    axes: Tuple[int, ...],
) -> Dict[str, np.ndarray]:
    """Compute mean/std for the prediction window and the reference window."""
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
    """Normalise a `Simulation`'s time-series data in-place"""
    sim.simulation = (sim.simulation - sim.desc["mean"]) / (2 * sim.desc["std"])
