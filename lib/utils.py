import yaml
from pathlib import Path


def get_ocean_term(property):
    """
    Retrieve an ocean-related term from the 'ocean_terms.yaml' file.

    Parameters
    ----------
    property : str
        The key of the term to retrieve from the YAML under the 'Terms' section.
        For example, 'temperature' to fetch the corresponding ocean temperature term.

    Returns
    -------
    str or None
        The term associated with the given property, or None if the file is missing
        or the property is not defined.

    Raises
    ------
    FileNotFoundError
        If the 'ocean_terms.yaml' file cannot be found.
    KeyError
        If the specified property is not found under 'Terms' in the YAML file.
    """
    try:
        with open("ocean_terms.yaml", "r") as f:
            terms = yaml.safe_load(f)

        # Attempt to retrieve the requested term
        return terms["Terms"][property]

    except FileNotFoundError:
        print(
            "\nCouldn’t find 'ocean_terms.yaml'. Please create the file with a 'Terms' section.\n"
        )
        return None

    except KeyError:
        print(
            f"\nThe term '{property}' was not found in the 'Terms' section of 'ocean_terms.yaml'.\n"
        )
        return None


def get_forecast_technique(nemo_directory, forecast_techniques):
    """Retrieve a forecasting technique from the 'techniques_config.yaml' file.

    Parameters
    ----------
    nemo_directory : Path
        The directory where the 'techniques_config.yaml' file is located.
    forecast_techniques : dict
        A dictionary of available forecasting techniques.

    Returns
    -------
    ForecastTechnique
        An instance of the specified forecasting technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the `forecast_techniques` dictionary.
    """
    with open(nemo_directory / "techniques_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config["Forecast_technique"]["name"] not in forecast_techniques:
        raise KeyError(
            f"Forecast_technique {config['Forecast_technique']['name']} not found. Have you specified a valid forecasting technique in the config file?"
        )
    else:
        return forecast_techniques[config["Forecast_technique"]["name"]]


def get_dr_technique(nemo_directory, dimensionality_reduction_techniques):
    """Retrieve a dimensionality reduction technique from the 'techniques_config.yaml' file.

    Parameters
    ----------
    nemo_directory : Path
        The directory where the 'techniques_config.yaml' file is located.
    dimensionality_reduction_techniques : dict
        A dictionary of available dimensionality reduction techniques.

    Returns
    -------
    DimensionalityReductionTechnique
        An instance of the specified dimensionality reduction technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the `dimensionality_reduction_techniques` dictionary.
    """
    with open(nemo_directory / "techniques_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config["DR_technique"]["name"] not in dimensionality_reduction_techniques:
        raise KeyError(
            f"DR_technique {config['DR_technique']['name']} not found. Have you specified a valid dimensionality reduction technique in the config file?"
        )
    else:
        return dimensionality_reduction_techniques[config["DR_technique"]["name"]]
