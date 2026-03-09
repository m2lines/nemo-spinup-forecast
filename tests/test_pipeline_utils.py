import numpy as np
import pytest

from nemo_spinup_forecast.pipeline_utils import abs_error_stats


def test_3d_spatial_reduction_shape_and_values():
    """Reducing over spatial axes (y, x) leaves a time-indexed output."""
    err = np.zeros((8, 3, 4))  # (time, y, x)
    err[-3:] = 10.0  # pred window: last 3 time steps
    err[:-3] = 1.0  # ref window: first 5 time steps
    result = abs_error_stats(err, steps=3, axes=(1, 2))
    # pred = err[5:] → shape (3, 3, 4), reduce (y, x) → shape (3,)
    # ref  = err[:5] → shape (5, 3, 4), reduce (y, x) → shape (5,)
    assert result["pred_mean"].shape == (3,)
    assert result["ref_mean"].shape == (5,)
    np.testing.assert_allclose(result["pred_mean"], 10.0)
    np.testing.assert_allclose(result["ref_mean"], 1.0)


def test_steps_zero_gives_full_ref():
    """steps=0 → pred is empty (nan), ref is the full array (baseline case)."""
    err = np.ones((10, 4, 5))
    err[7:] = 3.0
    result = abs_error_stats(err, steps=0, axes=(0, 1, 2))
    assert np.isnan(result["pred_mean"])
    assert result["ref_mean"] == pytest.approx((7 * 1.0 + 3 * 3.0) / 10)


def test_steps_equals_full_length():
    """steps=len(err) → pred covers the entire time axis."""
    err = np.arange(6.0).reshape(6, 1, 1)
    result = abs_error_stats(err, steps=6, axes=(0, 1, 2))
    assert result["pred_mean"] == pytest.approx(2.5)  # mean of 0..5
