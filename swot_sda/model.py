"""
Core score-based data assimilation model classes.

Provides:
  - eps_edm                    : EDM denoiser → SDA epsilon adaptor
  - GaussianScore              : Gaussian observation likelihood score
  - VPSDE                      : Variance-preserving SDE sampler with Langevin corrector
  - MyMCScoreNet               : Markov-chain sliding-window score network for time series
  - load_diffusion_model       : Full checkpoint loader via NVIDIA Modulus (requires modulus)
  - load_weights_from_checkpoint : Weights-only loader for a pre-instantiated model (no modulus)

References:
  Karras et al. (2022) — EDM framework
  Rozet & Louppe (2023) — Score-Based Data Assimilation
  Manshausen et al. (2024) — Application to geophysical fields
"""

import math
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor, Size
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_diffusion_model(checkpoint_path: str, device: torch.device, fp16: bool = True):
    """Load a diffusion model from a NVIDIA Modulus ``.mdlus`` checkpoint.

    Requires the ``modulus`` package to be installed — it provides both the
    ``Module.from_checkpoint`` mechanism and the UNet architecture class registered
    via entry points. Use :func:`load_weights_from_checkpoint` instead if you are
    working in an environment without ``modulus``.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.mdlus`` checkpoint file.
    device : torch.device
    fp16 : bool
        Enable half-precision inference (default True).
    """
    from ._modulus import Module  # lazy: keeps modulus optional at import time
    print(f"Loading diffusion model from {checkpoint_path}")
    net = Module.from_checkpoint(checkpoint_path)
    net = net.eval().to(device).to(memory_format=torch.channels_last)
    if fp16:
        net.use_fp16 = True
    return net


def load_weights_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device = None,
    strict: bool = True,
) -> nn.Module:
    """Load weights from a ``.mdlus`` checkpoint into an already-instantiated model.

    This function bypasses the ``modulus`` package entirely. It treats the
    ``.mdlus`` file purely as a tar archive and extracts the ``model.pt`` weight
    file, then calls ``model.load_state_dict``. You are responsible for
    instantiating ``model`` with the correct architecture beforehand.

    A ``.mdlus`` file is a tar archive containing:
      - ``model.pt``     — PyTorch state dict (all that this function uses)
      - ``args.json``    — constructor arguments used by ``modulus`` to rebuild the model
      - ``metadata.json``— modulus version and checkpoint format version

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.mdlus`` checkpoint file.
    model : nn.Module
        A pre-instantiated model whose architecture matches the checkpoint.
    device : torch.device, optional
        Map location for ``torch.load``. If None, uses the model's current device.
    strict : bool
        Whether to require an exact key match between checkpoint and model (default True).

    Returns
    -------
    nn.Module
        The same ``model`` object, with weights loaded and set to eval mode.

    Example
    -------
    >>> # Instantiate your own UNet (no modulus needed)
    >>> from my_project.models import MyUNet
    >>> model = MyUNet(**my_unet_kwargs).to(device)
    >>> model = load_weights_from_checkpoint(
    ...     "diff_models/training-state-diffusion-014_SSH_5tstep_12hrly-058293.mdlus",
    ...     model, device=device,
    ... )
    >>> # Then wrap for SDA inference as normal
    >>> eps   = eps_edm(model, shape=(5, 128, 128))
    >>> score = MyMCScoreNet(eps, order=2)
    """
    map_location = device if device is not None else next(model.parameters()).device
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp)
        with tarfile.open(checkpoint_path, "r") as tar:
            # Extract only model.pt to avoid processing unneeded files
            member = tar.getmember("model.pt")
            tar.extract(member, path=local_path)
        state_dict = torch.load(local_path / "model.pt", map_location=map_location)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


class eps_edm(nn.Module):
    r"""Converts a denoising model trained in the EDM framework (Karras et al. 2022)
    to the epsilon used in SDA (Rozet & Louppe 2023), following the formula from
    the appendix of Manshausen et al. 2024.

    Arguments:
        D     : denoising neural network trained in EDM framework
        shape : batch shape (e.g. ``(5, 128, 128)`` for 5 time steps)
        alpha : noise schedule type — ``'cos'`` (default), ``'lin'``, or ``'exp'``
        eta   : minimum noise floor

    Authors: Manshausen / Nvidia team (2025); modifications by Tatsu Monkman and Scott Martin
    """

    def __init__(
        self,
        D: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()
        self.D = D
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError("alpha must be one of {'lin', 'cos', 'exp'}")

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(t.shape + (1,) * len(self.shape))
        mu = self.mu(t)
        sigma = self.sigma(t)
        return (mu / sigma) * (x / mu - self.D(x / mu, sigma / mu))


class GaussianScore(nn.Module):
    r"""Score module for a Gaussian observation likelihood :math:`p(y | x) = \mathcal{N}(y | A(x), \Sigma)`.

    Note: returns :math:`-\sigma(t)\, s(x(t), t \mid y)` (i.e. the epsilon parameterisation).

    Arguments:
        y      : observed values, shape ``(num_observed,)``
        A      : observation operator mapping state → observed space
        std    : observation noise standard deviation
        sde    : VPSDE instance wrapping the prior score
        gamma  : inflation factor for the likelihood variance (default 1e-2)
        detach : detach score computation from grad tape (default False)

    Authors: Francois Rozet (Rozet & Louppe 2023); modifications by Tatsu Monkman
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: "VPSDE",
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()
        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))
        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        if self.detach:
            eps = self.sde.eps(x, t)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            if not self.detach:
                eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu

            A_x = self.A(x_)

            if A_x.dim() == 1:
                err = self.y - A_x
            else:
                y_expanded = self.y.unsqueeze(0).expand_as(A_x)
                err = y_expanded - A_x

            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2
            log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)
        return eps - sigma * s


class VPSDE(nn.Module):
    r"""Variance-preserving SDE sampler with optional Langevin corrector steps.

    Arguments:
        eps   : epsilon (denoiser) network
        shape : field shape, e.g. ``(T, H, W)``
        alpha : noise schedule — ``'cos'`` (default), ``'lin'``, or ``'exp'``
        eta   : minimum noise floor

    Authors: Francois Rozet (Rozet & Louppe 2023); modifications by Tatsu Monkman
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()
        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError("alpha must be one of {'lin', 'cos', 'exp'}")

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Sample from the perturbation kernel :math:`p(x(t) \mid x)`."""
        t = t.reshape(t.shape + (1,) * len(self.shape))
        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps
        if train:
            return x, eps
        return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
        save_intermediates: bool = False,
    ):
        r"""Sample from :math:`p(x(0))` via reverse-time SDE.

        Arguments:
            shape            : batch shape
            steps            : number of discrete time steps
            corrections      : number of Langevin corrector steps per time step
            tau              : Langevin step size amplitude
            save_intermediates: if True, returns ``(x_final, intermediates)``
        """
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)
        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps
        intermediates = [] if save_intermediates else None

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t)

                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt)
                    delta = tau / (eps.square().mean(dim=self.dims, keepdim=True) + 1e-6)
                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

                if save_intermediates:
                    intermediates.append(x.detach().clone().cpu())

        x_final = x.reshape(shape + self.shape)
        if save_intermediates:
            return x_final, torch.stack(intermediates, dim=0)
        return x_final

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Denoising score-matching loss."""
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)
        err = (self.eps(x, t) - eps).square()
        if w is None:
            return err.mean()
        return (err * w).mean() / w.mean()


class MyMCScoreNet(nn.Module):
    r"""Markov-chain sliding-window score network for arbitrary-length time series.

    Wraps a denoiser trained on fixed-length windows of size ``T = 2*order + 1``
    and iterates it over a sequence of arbitrary length ``L`` via overlapping
    windows (unfold → score → fold, keeping only the central prediction of each
    window).

    Arguments:
        base_model : pre-trained denoiser, expects input ``[B, T, H, W]``
        order      : half-window size; full window length = ``2*order + 1``

    Usage::

        wrapped_score = MyMCScoreNet(net, order=2)
        sde = VPSDE(GaussianScore(..., sde=VPSDE(wrapped_score, ...)), ...)

    Authors: Francois Rozet (Rozet & Louppe 2023); modifications by Tatsu Monkman
    """

    def __init__(self, base_model: nn.Module, order: int = 2):
        super().__init__()
        self.order = order
        self.kernel = base_model

    def forward(self, x: Tensor, t: Tensor = None, c: Tensor = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape ``[B, L, H, W]``
        t : scalar or ``[B]`` diffusion time
        """
        B, L, H, W = x.shape
        x_unfolded = self.unfold(x, self.order)        # [B, num_windows, T, H, W]
        B, num_windows, T, H, W = x_unfolded.shape
        x_flat = x_unfolded.reshape(-1, T, H, W)       # [B*num_windows, T, H, W]

        if t.dim() == 0:
            t_expanded = t.unsqueeze(0).expand(B * num_windows)
        else:
            t_expanded = t.unsqueeze(1).expand(B, num_windows).reshape(-1)

        s = self.kernel(x_flat, t_expanded)
        s = s.view(B, num_windows, T, H, W)
        return self.fold(s, self.order)

    @staticmethod
    def unfold(x: Tensor, order: int) -> Tensor:
        """Split ``[B, L, H, W]`` into overlapping windows of length ``2*order+1``."""
        T = 2 * order + 1
        x_unf = x.unfold(dimension=1, size=T, step=1)  # [B, L-T+1, H, W, T]
        return x_unf.permute(0, 1, 4, 2, 3)            # [B, L-T+1, T, H, W]

    @staticmethod
    def fold(x: Tensor, order: int) -> Tensor:
        """Reconstruct sequence by taking the central prediction of each window."""
        center = x[:, :, order]   # [B, num_windows, H, W]
        return torch.cat((
            x[:, 0, :order],      # leading edge frames
            center,
            x[:, -1, -order:],    # trailing edge frames
        ), dim=1)
