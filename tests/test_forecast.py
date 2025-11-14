import pytest


def test_import_forecast():
    """Test that the forecast module can be imported successfully."""
    try:
        import lib.forecast  # noqa: PLC0415 Ignores "imported but unused"

        assert True, "forecast imported successfully."
    except ImportError as e:
        pytest.fail(f"Failed to import forecast: {e}")
