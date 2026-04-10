# CLAUDE.md — swot_sda

Context for AI-assisted development sessions.

## What this package is

`swot_sda` is a clean, publication-ready Python package for score-based data assimilation
of SWOT satellite observations (SSH and SST). It was consolidated from a sprawling
experiment directory (`SDA_scripts/`) into a structured package in April 2026, while
the SDA_scripts notebooks remain in active use for ongoing publication experiments.

**Do not modify anything under `SDA_scripts/` or `SDA_scripts/src/`** — those files are
live imports for notebooks still being used for the paper.

## How the package was assembled

Three source locations were consolidated:

| `swot_sda` file | Source | Changes made |
|---|---|---|
| `model.py` | `SDA_scripts/src/tatsu_sda_final.py` | Dropped stale `from sda.mcs import *` / `from sda.utils import *` (unused); `from modulus import Module` moved inside `load_diffusion_model()` as a lazy import (so modulus is not required at import time); added `load_weights_from_checkpoint()` as a modulus-free alternative |
| `inference.py` | `SDA_scripts/src/inference.py` | `from . import tatsu_sda_final` → `from . import model as sda_model`; `time_slice=[]` default → `None`; coord construction uses `mask_in.shape[-2:]` |
| `transforms.py` | `SDA_scripts/src/transforms.py` | Verbatim copy |
| `metrics.py` | `SDA_scripts/src/metrics.py` | `from src import transforms` → `from . import transforms` |
| `_sda/` | `/sda/sda/` (Rozet & Louppe 2023) | `import jax/jnp/rng` in `mcs.py` gated behind `try/except`; `KolmogorovFlow.__init__` raises `ImportError` if JAX absent; attribution headers added |
| `_modulus/` | `GenDA/modulus/modulus/{models,registry,utils}/` | 4 absolute `modulus.*` imports → relative; `modulus.__version__` → local `__version__ = "0.7.0a0"`; `model_registry.py` `issubclass(model, modulus.Module)` check removed to break circular import; SPDX headers preserved |

Source paths (all relative to `/usr/home/tm3076/swot_SUM03/SWOT_project/sda/`):
- Experiment code: `tatsu_experiments/SDA_scripts/src/`
- `sda` package: `sda/` (installed via `pip install -e .`)
- Modulus: `tatsu_experiments/GenDA/modulus/` (installed via `pip install -e .`)

## Key design decisions

**Single package** (not split into inference + analysis): `metrics.py` depends on
`transforms.py`; splitting them would create a cross-package dependency for no benefit
at this publication scope.

**Underscore-prefixed vendored subpackages** (`_sda`, `_modulus`): signals to users
that these are internal and should not be imported directly.

**`model.py` has local modifications to `_sda/score.py` originals:**
- `GaussianScore.forward` — adds batched observation handling (the original assumes a
  single sample; the local version expands `y` to match batch dim)
- `VPSDE.sample` — adds `save_intermediates` flag for inspecting the diffusion trajectory
- `VPSDE` corrector — adds `1e-6` floor in the Langevin delta computation for stability

**`_modulus.model_registry` still uses `entry_points(group="modulus.models")`**: this
intentionally delegates architecture lookup to the installed `modulus` package so that
the UNet class embedded in the `.mdlus` checkpoint can be found via `importlib`. This
is only exercised when calling `load_diffusion_model()`.

**Two checkpoint loading paths exist** to support different environments:
- `load_diffusion_model(path, device)` — full reconstruction via `modulus`; instantiates
  the correct architecture automatically from `args.json` inside the checkpoint. Requires
  the `modulus` package installed. The import is lazy (inside the function body) so
  `import swot_sda` works fine without modulus.
- `load_weights_from_checkpoint(path, model, device)` — modulus-free; you instantiate
  the model yourself, then this function opens the `.mdlus` tar, extracts `model.pt`,
  and calls `load_state_dict`. Uses only stdlib + torch. Suitable for export to
  environments that have their own UNet loader.

## Module responsibilities

```
model.py      load_diffusion_model()         — loads .mdlus checkpoint via _modulus.Module
                                               (requires modulus installed; lazy import)
              load_weights_from_checkpoint() — modulus-free alternative: takes a
                                               pre-instantiated nn.Module, extracts
                                               model.pt from the tar and calls
                                               load_state_dict; uses only stdlib + torch
              eps_edm                        — EDM denoiser → SDA epsilon (Karras 2022 → Rozet 2023)
              GaussianScore                  — Gaussian likelihood score (batched, modified)
              VPSDE                          — VP-SDE sampler with Langevin corrector (modified)
              MyMCScoreNet                   — sliding-window MC score for time series

inference.py  run_inference()         — builds posterior SDE, ensemble loop, xr.Dataset output

transforms.py SSHNormalizer           — simple (x - mean) / std
              SeasonalSSTNormalizer   — climatology subtraction, matched by day-of-year

metrics.py    PSD_WPSD_metrics()      — isotropic PSD, WPSD, RMSE, leaderboard scores
              rmse_based_scores()     — OSSE challenge NRMSE
              psd_based_scores()      — OSSE challenge spectral score
```

## Input/output conventions

- Spatial tiles: 128×128 pixels
- `x_in` shape: `[T, 2, H, W]` — channel 0 = true field, channel 1 = masked/observed field
- `mask_in` shape: `[T, H, W]` — binary, 1 = observed
- All fields are **normalised** before passing to `run_inference`
- `run_inference` returns an `xr.Dataset` with dims `(sample, time, x, y)`
- Results are saved as zarr; `time_slice` coordinate is needed by SST metrics for denormalisation

## Dependency notes

| Dep | Why | Optional? |
|---|---|---|
| `zuko` | `_sda.nn` (LayerNorm) and `_sda.score` (broadcast) | No |
| `POT` | `_sda.utils.emd` | No (imported at module level) |
| `h5py` | `_sda.utils.TrajectoryDataset` | No (imported at module level) |
| `importlib_metadata` | `_modulus.model_registry` entry points | No |
| `fsspec`, `s3fs`, `requests` | `_modulus.filesystem` remote checkpoint fetch | No |
| `jax`, `jax-cfd` | `_sda.mcs.KolmogorovFlow` only | Yes — gated with try/except |
| `modulus` (installed) | UNet architecture lookup via entry points | Yes — only required if calling `load_diffusion_model`; `load_weights_from_checkpoint` needs no modulus at all |

## Quick smoke test

```bash
cd /usr/home/tm3076/swot_SUM03/SWOT_project/sda/swot_sda
pip install -e .
python -c "import swot_sda; print('top-level OK')"
python -c "from swot_sda._sda import score, nn, mcs, utils; print('_sda OK')"
python -c "from swot_sda._modulus import Module, ModelMetaData; print('_modulus OK')"
```
