"""
Inference runner for score-based data assimilation.
"""

import numpy as np
import torch
import xarray as xr

from . import model as sda_model


def run_inference(
    wrapped_score,
    x_in,
    mask_in,
    N_MEMBERS: int = 2,
    n_batch: int = 2,
    save_intermediates: bool = False,
    sde_std: float = 0.1,
    sde_gamma: float = 1e-2,
    sde_steps: int = 256,
    eta: float = 1e-3,
    l_corrections: int = 2,
    l_tau: float = 0.1,
    verbose: bool = True,
    time_slice=None,
    add_noise: bool = False,
):
    r"""Run score-based data assimilation inference and return an xarray Dataset.

    Parameters
    ----------
    wrapped_score : nn.Module
        Score network, typically ``MyMCScoreNet`` wrapping ``eps_edm``.
    x_in : array-like, shape ``[T, 2, H, W]``
        Input tensor where ``x_in[:, 0]`` is the true field and
        ``x_in[:, 1]`` is the (masked) observed field.
    mask_in : array-like, shape ``[T, H, W]``
        Binary observation mask corresponding to ``x_in[:, 1]``.
    N_MEMBERS : int
        Total number of ensemble members to generate.
    n_batch : int
        Members to generate per GPU batch (tune to GPU memory).
    save_intermediates : bool
        If True, saves the full diffusion trajectory.
    sde_std : float
        Observation noise standard deviation (normalised units).
    sde_gamma : float
        Likelihood variance inflation factor.
    sde_steps : int
        Number of reverse-diffusion steps.
    eta : float
        Minimum noise floor for the VP noise schedule.
    l_corrections : int
        Number of Langevin corrector steps per diffusion step.
    l_tau : float
        Langevin step size amplitude.
    verbose : bool
        Print inference configuration summary.
    time_slice : array-like, optional
        Time-index coordinates to attach to the output Dataset (used for
        SST seasonal denormalisation).
    add_noise : bool
        Add synthetic Gaussian noise (std=``sde_std``) to the observations.

    Returns
    -------
    xr.Dataset
        Contains ``x_sample_members``, ``x_true``, ``x_masked``, ``mask_in``,
        ``x_noise``, and optionally ``x_sample_intermediates``.
    """
    if time_slice is None:
        time_slice = []

    # ------------------------------------------------------------------
    # Observation operator: apply mask over all ensemble members
    # ------------------------------------------------------------------
    mask_in = torch.tensor(mask_in)
    x_in = torch.tensor(x_in)

    def A(x_s):
        if x_s.dim() == 3:
            return x_s[mask_in.bool()]
        mask_expanded = mask_in.bool().unsqueeze(0).expand(x_s.shape[0], -1, -1, -1)
        return x_s[mask_expanded].reshape(x_s.shape[0], -1)

    # ------------------------------------------------------------------
    # Optionally add synthetic noise to observations
    # ------------------------------------------------------------------
    noise = torch.zeros(x_in[:, 1].shape)
    if add_noise:
        noise = torch.normal(mean=0.0, std=sde_std, size=x_in[:, 1].shape)
        x_in[:, 1] += noise

    y = x_in[:, 1][mask_in.bool()]

    # ------------------------------------------------------------------
    # Build the posterior score network
    # ------------------------------------------------------------------
    sde = sda_model.VPSDE(
        sda_model.GaussianScore(
            y=y,
            A=A,
            std=sde_std,
            gamma=sde_gamma,
            sde=sda_model.VPSDE(wrapped_score, shape=(len(x_in), 128, 128)),
        ),
        shape=(len(x_in), 128, 128),
        eta=eta,
    ).cuda()

    if verbose:
        print(f"--------------------------------------------------")
        print(f"Running {N_MEMBERS} members")
        print(f"x_in.shape     {x_in.shape}")
        print(f"mask_in.shape  {mask_in.shape}")
        print(f"y.shape        {y.shape}\n")
        print(f"SDE hyperparameters:")
        print(f"  obs std        {sde_std}")
        print(f"  gamma          {sde_gamma}")
        print(f"  min eta        {eta}")
        print(f"  steps          {sde_steps}")
        print(f"  add noise      {add_noise}")
        print(f"  corrections    {l_corrections}")
        print(f"  langevin tau   {l_tau}")
        print(f"  intermediates  {save_intermediates}")
        print(f"--------------------------------------------------")

    # ------------------------------------------------------------------
    # Ensemble loop
    # ------------------------------------------------------------------
    x_sample_members = []
    x_sample_intermediates = []
    with torch.no_grad():
        for n_members in range(0, N_MEMBERS, n_batch):
            print(f"Running members {n_members}-{n_members + n_batch}")
            if save_intermediates:
                x_final, intermediates = sde.sample(
                    (n_batch,), steps=sde_steps,
                    corrections=l_corrections, tau=l_tau,
                    save_intermediates=True,
                )
                x_sample_members.append(x_final.cpu())
                x_sample_intermediates.append(intermediates.cpu())
            else:
                x_final = sde.sample(
                    (n_batch,), steps=sde_steps,
                    corrections=l_corrections, tau=l_tau,
                ).cpu()
                x_sample_members.append(x_final)

    x_out = torch.cat(x_sample_members, dim=0)

    # ------------------------------------------------------------------
    # Package results as xarray Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        {
            'x_sample_members': (['sample', 'time', 'x', 'y'], x_out),
            'mask_in':          (['time', 'x', 'y'], mask_in),
            'x_masked':         (['time', 'x', 'y'], x_in[:, 1]),
            'x_true':           (['time', 'x', 'y'], x_in[:, 0]),
            'x_noise':          (['time', 'x', 'y'], noise),
        },
        coords={
            'sample': np.arange(len(x_out)),
            'time':   np.arange(len(mask_in)),
            'x':      np.arange(mask_in.shape[-2]),
            'y':      np.arange(mask_in.shape[-1]),
        },
    )

    if len(time_slice) == len(ds.time):
        ds = ds.assign_coords(time_i=("time", time_slice))

    if save_intermediates:
        x_intermediates = torch.cat(x_sample_intermediates, dim=0)
        ds_inter = xr.Dataset(
            {'x_sample_intermediates': (['sigt', 'sample', 'time', 'x', 'y'], x_intermediates)},
            coords={
                'sigt':   np.arange(x_intermediates.shape[0]),
                'sample': np.arange(len(x_out)),
                'time':   np.arange(len(mask_in)),
                'x':      np.arange(mask_in.shape[-2]),
                'y':      np.arange(mask_in.shape[-1]),
            },
        )
        ds = xr.merge([ds, ds_inter])

    return ds
