#!/usr/bin/env bash

set -e
# Print an error message if any command fails
trap 'echo "Error: Command \"$BASH_COMMAND\" failed with exit code $? at line $LINENO." >&2' ERR

# Create and switch to the data directory
mkdir -p tests/data/nemo_data_e3 || { echo "Failed to create directory tests/data/nemo_data_e3" >&2; exit 1; }
cd tests/data/nemo_data_e3 || { echo "Failed to change directory to tests/data/nemo_data_e3" >&2; exit 1; }

# Download files (resume if partially downloaded)
wget -c https://github.com/isaacaka/test_releases/releases/download/v1.test.data/DINO_1m_To_1y_grid_T.nc || { echo "Failed to download DINO_1m_To_1y_grid_T.nc" >&2; exit 1; }
wget -c https://github.com/isaacaka/test_releases/releases/download/v1.test.data/DINO_1y_grid_T.nc      || { echo "Failed to download DINO_1y_grid_T.nc" >&2;      exit 1; }
