# swot_sda

Score-based data assimilation (SDA) for SWOT sea surface height (SSH) and sea surface temperature (SST) reconstruction. Fits pre-trained diffusion models to sparse satellite observations at inference time, producing probabilistic ensemble time series over 128×128 spatial tiles. Developed for the SWOT SUM03 project (Monkman et al., in preparation).

---

## Package structure

```
swot_sda/
├── model.py        — core SDA classes: eps_edm, VPSDE, GaussianScore, MyMCScoreNet,
│                     and load_diffusion_model checkpoint loader
├── inference.py    — run_inference() orchestrator: builds posterior SDE, runs
│                     ensemble sampling, returns xarray Dataset
├── transforms.py   — SSHNormalizer (mean/std) and SeasonalSSTNormalizer
│                     (daily climatology matching by day-of-year)
├── metrics.py      — PSD_WPSD_metrics(), rmse_based_scores(), psd_based_scores()
│                     following the OSSE SSH mapping challenge conventions
├── _sda/           — vendored: Rozet & Louppe (2023) sda library
│   ├── nn.py           residual networks and U-Net
│   ├── score.py        VPSDE, GaussianScore, MCScoreNet (upstream originals)
│   ├── mcs.py          Markov chain dynamics (Lorenz, KolmogorovFlow, ...)
│   └── utils.py        training loop, particle filter, EMD/MMD utilities
└── _modulus/       — vendored: NVIDIA Modulus 0.7.0a0 (Apache-2.0)
    ├── module.py       Module base class with from_checkpoint()
    ├── meta.py         ModelMetaData dataclass
    ├── model_registry.py  Borg-singleton model registry
    └── filesystem.py   _download_cached, _get_fs, Package
```

---

## Installation

```bash
pip install -e /path/to/swot_sda
```

The pre-trained model checkpoints (`.mdlus` files) require the original NVIDIA Modulus package to be installed alongside `swot_sda`, so that the UNet architecture class registered by Modulus can be looked up via entry points when loading a checkpoint:

```bash
pip install -e /path/to/GenDA/modulus   # NVIDIA Modulus (provides the UNet class)
```

---

## Pre-trained models

Two diffusion model checkpoints are available in `SDA_scripts/diff_models/`:

| Checkpoint | Variable | Window | Cadence | Domain |
|---|---|---|---|---|
| `training-state-diffusion-014_SSH_5tstep_12hrly-058293.mdlus` | SSH | 5 time steps | 12-hourly | North Atlantic |
| `training-state-diffusion-013_SST_5tstep_12hrly-117522.mdlus` | SST | 5 time steps | 12-hourly | North Pacific |

Both models operate on 128×128 spatial tiles. At inference time the sliding-window `MyMCScoreNet` wrapper extends them to arbitrary-length time series.

---

## Usage

### Example 1: SSH inference (with Modulus)

```python
import torch
import numpy as np
from swot_sda import load_diffusion_model, eps_edm, MyMCScoreNet, run_inference
from swot_sda.transforms import SSHNormalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and wrap the pre-trained diffusion model ---
net   = load_diffusion_model(
    "SDA_scripts/diff_models/training-state-diffusion-014_SSH_5tstep_12hrly-058293.mdlus",
    device,
)
eps   = eps_edm(net, shape=(5, 128, 128))   # EDM → SDA epsilon adaptor
score = MyMCScoreNet(eps, order=2)           # sliding-window MC score wrapper

# --- Normalise inputs ---
# x_true, x_obs: np.ndarray [T, 128, 128] in metres
# mask:          np.ndarray [T, 128, 128] binary (1 = observed)
normalizer  = SSHNormalizer(std=0.0454)
x_in = np.stack(
    [normalizer.normalize(x_true),
     normalizer.normalize(x_obs)],
    axis=1,                                  # [T, 2, 128, 128]
)

# --- Run posterior inference ---
result = run_inference(
    score, x_in, mask,
    N_MEMBERS    = 30,
    n_batch      = 5,
    sde_std      = 0.1,       # obs noise std in normalised units
    sde_gamma    = 5e-2,      # likelihood variance inflation
    sde_steps    = 256,
    l_corrections= 2,
    l_tau        = 0.1,
)

# result is an xr.Dataset with dims (sample, time, x, y)
ensemble_mean = normalizer.denormalize(result.x_sample_members.mean("sample").values)
```

### Example 2: SSH inference (without Modulus)

If you are working in an environment where the `modulus` package is not available, use
`load_weights_from_checkpoint` instead. This requires you to instantiate the correct
UNet architecture yourself beforehand — `swot_sda` then loads only the weight tensor.

```python
import torch
from swot_sda import load_weights_from_checkpoint, eps_edm, MyMCScoreNet, run_inference
from swot_sda.transforms import SSHNormalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Instantiate your own UNet (architecture must match the checkpoint)
from my_project.models import MyUNet
net = MyUNet(**my_unet_kwargs).to(device)

# 2. Load only the weights from the .mdlus file — no modulus required
net = load_weights_from_checkpoint(
    "SDA_scripts/diff_models/training-state-diffusion-014_SSH_5tstep_12hrly-058293.mdlus",
    net, device=device,
)

# 3. Everything downstream is identical
eps    = eps_edm(net, shape=(5, 128, 128))
score  = MyMCScoreNet(eps, order=2)
result = run_inference(score, x_in, mask, N_MEMBERS=30, n_batch=5)
```

> **What is inside a `.mdlus` file?**
> A `.mdlus` checkpoint is a tar archive containing three files:
> - `model.pt` — the PyTorch state dict (`load_weights_from_checkpoint` uses only this)
> - `args.json` — the constructor kwargs recorded by `modulus` when the model was saved;
>   use these to replicate the exact architecture if you need to re-instantiate the UNet
> - `metadata.json` — modulus version and checkpoint format version
>
> You can inspect a checkpoint with:
> ```bash
> python -c "
> import tarfile, json
> with tarfile.open('my_model.mdlus') as t:
>     print(json.load(t.extractfile('args.json')))
> "
> ```

---

### Example 3: SST inference

SST requires seasonal normalisation against a daily climatology rather than a fixed standard deviation.

```python
import numpy as np
from swot_sda import load_diffusion_model, eps_edm, MyMCScoreNet, run_inference
from swot_sda.transforms import SeasonalSSTNormalizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
net   = load_diffusion_model(
    "SDA_scripts/diff_models/training-state-diffusion-013_SST_5tstep_12hrly-117522.mdlus",
    device,
)
eps   = eps_edm(net, shape=(5, 128, 128))
score = MyMCScoreNet(eps, order=2)

# --- Normalise using actual observation datetimes ---
# Climatology is matched by day-of-year, so data from different years can be mixed.
normalizer = SeasonalSSTNormalizer(std=5.5, extra_mean_tuning=1.5)
times = np.array(["2012-03-01", "2012-03-02", "2012-03-03",
                  "2012-03-04", "2012-03-05"], dtype="datetime64[ns]")

x_true_norm = normalizer.normalize_from_datetime(x_true, data_times=times)
x_obs_norm  = normalizer.normalize_from_datetime(x_obs,  data_times=times)
x_in = np.stack([x_true_norm, x_obs_norm], axis=1)   # [T, 2, 128, 128]

# --- Run inference ---
# sde_std is the obs noise in *normalised* units: 0.025 K / 5.5 K ≈ 0.0045
result = run_inference(
    score, x_in, mask,
    N_MEMBERS    = 30,
    n_batch      = 5,
    sde_std      = 0.025 / 5.5,
    sde_gamma    = 5e-2,
    sde_steps    = 256,
    l_corrections= 2,
    time_slice   = np.arange(len(times)),   # used for SST denormalisation in metrics
)

# Denormalise using the same time coordinates
ensemble_mean = normalizer.denormalize_from_datetime(
    result.x_sample_members.mean("sample").values,
    data_times=times,
)
```

### Example 3: Saving and loading results

```python
import zarr

# Save to zarr (preferred for large ensembles)
result.to_zarr("my_ssh_experiment.zarr", mode="w")

# Reload in an analysis notebook
import xarray as xr
result = xr.open_zarr("my_ssh_experiment.zarr")
```

### Example 4: Evaluation metrics

```python
import xarray as xr
from swot_sda.metrics import PSD_WPSD_metrics

result = xr.open_zarr("my_ssh_experiment.zarr")

# Computes isotropic PSD, WPSD, RMSE, and OSSE leaderboard scores
# for each ensemble member and time step.
metrics = PSD_WPSD_metrics(result, denorm_field="SSH")

# Key output variables:
#   true_isospec          — isotropic PSD of the true field
#   WPSD_mean             — mean wavenumber PSD score across members
#   WPSD_sample           — per-member WPSD scores
#   leaderboard_nrmses    — OSSE leaderboard normalised RMSE per member
#   freq_cutoff_interp_mean — interpolated cutoff wavenumber (WPSD = 0.5)
#   RMSEs                 — per-(time, member) spatial RMSE

print(f"Leaderboard NRMSE  : {float(metrics.leaderboard_nrmses.mean()):.4f}")
print(f"Cutoff wavelength  : {float(1 / metrics.freq_cutoff_interp_mean.mean()):.1f} km")
```

---

## Provenance and attributions

This package consolidates code from three sources. If you use `swot_sda` in your work, please also cite the relevant upstream references.

| Component | Source | License | Reference |
|---|---|---|---|
| `model.py`, `inference.py`, `transforms.py`, `metrics.py` | This work | — | Monkman et al. (in prep.) |
| `_sda/` | [francois-rozet/sda](https://github.com/francois-rozet/sda) | MIT | Rozet & Louppe, NeurIPS 2023 |
| `_modulus/` | [NVIDIA/modulus](https://github.com/NVIDIA/modulus) | Apache-2.0 | — |

Additional credits:
- **EDM framework**: Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022. The `eps_edm` class converts an EDM-trained denoiser to the SDA epsilon parameterisation.
- **EDM→SDA adaptor formula**: Manshausen et al. (2024), "Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales", NeurIPS 2024. The specific formula implemented in `eps_edm.forward` follows the appendix of this work.
- **OSSE evaluation metrics**: Le Guillou et al. (2021), SSH mapping challenge. The leaderboard RMSE and PSD-based scores in `metrics.py` follow this benchmark's conventions.

---

## References

```
Rozet, F., & Louppe, G. (2023). Score-Based Data Assimilation.
    Advances in Neural Information Processing Systems, 36.
    https://arxiv.org/abs/2306.10574

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2022).
    Elucidating the Design Space of Diffusion-Based Generative Models.
    Advances in Neural Information Processing Systems, 35.

Manshausen, P., et al. (2024). Generative Data Assimilation of Sparse Weather Station
    Observations at Kilometer Scales.
    Advances in Neural Information Processing Systems, 38.
```
