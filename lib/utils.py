import yaml


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
