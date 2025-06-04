# Spin‑Up NEMO Forecasting Framework

**Overview**

This project provides a flexible framework for oceanographic time‑series forecasting. It separates dimensionality reduction (DR) and forecasting into interchangeable components, enabling you to swap in your own algorithms with minimal changes.

---

## 1. Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/m2lines/Spinup-Forecast
   cd <repo_dir>
   ```
2. **Set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Partial Project Structure

```text
├── lib/
│   ├── dimensionality_reduction.py   # Base + PCA/KPCA classes
│   ├── forecast_technique.py         # Base + GP/Recursive forecasters
│   └── restart.py                    # Restart‑file utilities
├── main_forecast.py                  # CLI for preparing & forecasting
├── main_restart.py                   # CLI for updating restart files
├── forecast.py                       # Orchestrates DR + forecasting
├── techniques_config.yaml            # Select DR & forecast technique
├── ocean_terms.yaml                  # Map variable names in NEMO grids
└── jumper.ipynb                      # Example notebook using PCA
```

---

## 3. Configuration

All user‑selectable techniques live in `techniques_config.yaml`:

```yaml
DR_technique:
  name: PCA                # Options: PCA, KernelPCA, or your custom class
Forecast_technique:
  name: GaussianProcessRecursiveForecaster  # Options: GaussianProcessForecaster, GaussianProcessRecursiveForecaster, or your custom class
```

Maps for NEMO grids live in `ocean_terms.yaml`:

```yaml
Terms:
  Salinity: soce
  Temperature: toce
  SSH: ssh
```

---

## 4. Quick Start — Prepare & Forecast
# TODO: Add explanation of what simulations files are needed

```bash
python main_forecast.py \
  --path /path/to/simulation/files \
  --ye True \
  --start 25 --end 65 \
  --comp 0.9 --steps 30
```

*Outputs*

* Prepared data in `simu_prepared/{term}/`
* Forecasted components in `simu_predicted/{term}.npy`

---

## 5. Extending the Framework

### 5.1 Adding a Custom Dimensionality Reduction

```python
from lib.dimensionality_reduction import DimensionalityReduction

class MyDR(DimensionalityReduction):
    def __init__(self, comp, **kwargs):
        self.comp = comp
        # initialise other parameters

    def set_from_simulation(self, sim):
        # copy metadata from Simulation
        ...

    def decompose(self, simulation, length):
        # return components, model‑instance, mask
        ...

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        # return mask, reconstructed‑array
        ...
```

Register the class in `forecast.py` and select it in `techniques_config.yaml` just as for the built‑in techniques.

### 5.2 Adding a Custom Forecasting Technique

```python
from lib.forecast_technique import BaseForecaster

class MyForecaster(BaseForecaster):
    def __init__(self, **params):
        # initialise your model
        ...

    def apply_forecast(self, y_train, x_train, x_pred):
        # fit your model on y_train (and x_train), predict x_pred
        # return (y_hat, y_hat_std)
        ...
```

Again, register your class and reference it in `techniques_config.yaml`.

---

## 6. Example Notebook

`Jumper.ipynb` demonstrates forecasting with `PCA`. Copy, modify, or extend it to test your own techniques.

### 6.1 Jumper.ipynb — Prepare and Forecast Simulations

The objective is to implement a Gaussian process forecast to forecast yearly simulations of the NEMO coupled climate model. For this we need simulation files of the sea surface height (zos or ssh), the salinity (so) and temperature (thetao).

We apply PCA on each simulation to transform those features to time series and observe the trend in the first component.

![img1](img/jumper1.png)

We forecast each component with a Gaussian process using the following kernel:

* **Long‑term trend**   : `0.1*DotProduct(sigma_0=0.0)`
* **Periodic patterns** : `10*ExpSineSquared(length_scale=5/45, periodicity=5/45)`
* **White noise**       : `2*WhiteKernel(noise_level=1)`

![img2](img/jumper3.png)

We then evaluate the RMSE:

![img2](img/jumper2.png)

---

## 7. Restart Workflow

### 7.1 Restart.ipynb — Update of Restart Files for NEMO

The objective is to update the last restart file to initialise the jump. For this we need the 340 restart files of the last simulated year, the predictions of sea surface height (zos or ssh), salinity (so) and temperature (thetao), and the Mask dataset of the corresponding simulation.

![img5](img/img3.png)

#### 1 – Predicted features

* **zos**       : Predicted sea surface height (ssh) – grid T – t,y,x
* **so**        : Predicted salinity – grid T – t,z,y,x
* **thetao**    : Predicted temperature – grid T – t,z,y,x

#### 2 – Maskdataset

*dimensions: t:1  y:331  x:360  z:75*

* `umask`   : continent mask for u grid (continent 0, sea 1)
* `vmask`   : continent mask for v grid (continent 0, sea 1)
* `e3t_0`   : initial cell thickness on z‑axis (grid T)
* `e2t`, `e1t`, `ff_f`, … 

#### 3 – Necessary features to update restart

* `e3t`    : `e3t_ini*(1+tmask4D*np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))`
* `deptht` : depth of the z‑axis – grid T
* `bathy`  : `np.ma.sum(e3t,axis=1)`

#### 4 – Restart file update

There are 340 restart files per year. Each file contains a slice of the x and y dimensions and 58 data variables, 15 of which are updated using the predictions.

> **NB**   `n` (now) and `b` (before) arrays represent the same state; in practice, NEMO saves two successive states in each file. In our code we set the two to the same state to use Euler forward for the restart.

* **ssh(n/b)** : sea‑surface height           → last prediction of zos

* **s(n/b)**   : sea salinity                 → last prediction of so

* **t(n/b)**   : sea temperature              → last prediction of thetao

* **v(n/b)**   : zonal velocity               → *The Planetary Ocean* (M. Fieux) p70

  $V(z)=\frac{g}{p\,f}\int_{z_0}^{Z}\frac{\partial\rho}{\partial x}\,dz + V_0$

* **u(n/b)**   : meridional velocity          → *The Planetary Ocean* p70

  $U(z)=\frac{g}{p\,f}\int_{z_0}^{Z}\frac{\partial\rho}{\partial y}\,dz + U_0$

* **sss\_m**    : sea‑surface salinity        → last prediction of so

* **sst\_m**    : sea‑surface temperature     → last prediction of thetao

* **ssu\_m**    : sea‑surface *u* velocity    → from new u

* **ssv\_m**    : sea‑surface *v* velocity    → from new v

* **rhop**     : potential density (kg m⁻³)  — see Equation of State of Sea‑Water (J. Le Sommer)

*Grid visuals*

![img6](img/grid0.png)
![img7](img/grid1.png)

---

## 8. Command‑Line Interface (`main`)

### *Prepare, forecast and initialise restart files in one command*

```bash
python main.py --ye True --start 25 --end 65 --comp 0.9 --steps 30 --path /scratchu/mtissot/SIMUp6Y
```

*Arguments*

* `ye`    : the simulation is expressed in years (`True`) or months (`False`)
* `start` : starting year (training data)
* `end`   : ending year (usually the last simulated year)
* `comp`  : number / ratio of components to accelerate
* `steps` : jump size (years if `ye=True`, months otherwise)
* `path`  : directory containing the simulation files

---

## 9. End‑to‑End Steps for Running Spin‑Up NEMO

1. **Run DINO for 50‑100 years.** A Slurm script is provided in the NEMO [notes](https://github.com/m2lines/Spinup-NEMO-notes/blob/main/nemo/buildandrun_NEMODINO.md). If you need more training data, concatenate monthly outputs `*grid_T.nc` with `ncrcat`.

2. **Create a virtual environment** (e.g. with conda) and install `requirements.txt`.

3. **Resample SSH**. Use the [Resample\_ssh.ipynb](https://github.com/m2lines/Spinup-NEMO/blob/resample_dino_data/Notebooks/Resample_ssh.ipynb) notebook to convert monthly SSH (`DINO_1m_grid_T.nc`) to annual (`DINO_1m_To_1y_grid_T.nc`). Temperature and salinity (3‑D) are already annual (`DINO_1y_grid_T.nc`).

4. **Create the projected state** with the updated `Jumper.ipynb`. Set `path` to the NEMO/DINO data directory:

   ![image](https://hackmd.io/_uploads/HkODLLHPyl.png)

5. **Prepare restart files**. Combine `mesh_mask_[0000].nc` and `DINO_[<time>]_restart_[<process>].nc` with **[REBUILD\_NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/4.2.0/tools/REBUILD_NEMO)**:

   ```bash
   ./rebuild_nemo -n ./nam_rebuild /path/to/DINO/restart/file/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild /path/to/DINO/mesh_mask 36
   ```

6. **Create an updated restart file** with `main_restart.py`:

   ```bash
   python main_restart.py \
     --restart_path /path/to/EXP00/ \
     --radical DINO_[<time>]_restart \
     --mask_file /full/path/to/EXP00/mesh_mask.nc \
     --prediction_path /full/path/to/directory/simus_predicted/
   ```

   * `--radical` is the prefix of the restart file;
   * `<time>` is the most recent restart time.

   The modified script (see the `run_with_DINO_data` branch) creates files named as the originals but with `NEW` prepended.

7. **Copy the experiment directory** inside the NEMO repository. Keep it as backup; the original will be overwritten.

8. **Copy the mesh mask and restart files** (`mesh_mask_<proc_id>.nc` and `DINO_[<time>]_restart_<proc_id>.nc`) back to the original experiment directory.

9. **Update `namelist_cfg`**

   Under `namrun` adjust:

   * `nn_it000` : first timestep (last timestep + 1)
   * `nn_itend` : final timestep
   * `cn_ocerst_in` : restart filename syntax (matches latest restart file)
   * `ln_rstart` : `.true.` to start from a restart file

   ![image](https://hackmd.io/_uploads/HJtsvsCPJe.png)

10. **Restart DINO** using the updated restart file.

---

## 10. Notes & TODOs

* hange cumsum axis = 1
* change update\_rho -> thetao\_new ⇒ Restart\["tn"]
* Add retstart and prepare

\# EMULATOR FRO NEMO MODEL

![img1](img/emulator.png)
