# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from NVIDIA Modulus: https://github.com/NVIDIA/modulus
# Source: modulus/models/module.py, version 0.7.0a0
#
# Changes from original:
#   - Replaced `import modulus` / `modulus.__version__` with local `__version__`
#   - Replaced `from modulus.models.meta import ModelMetaData` with relative import
#   - Replaced `from modulus.registry import ModelRegistry` with relative import
#   - Replaced `from modulus.utils.filesystem import ...` with relative import

import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import torch

from . import __version__
from .meta import ModelMetaData
from .model_registry import ModelRegistry
from .filesystem import _download_cached, _get_fs


class Module(torch.nn.Module):
    """The base class for all network models in Modulus.

    This should be used as a direct replacement for torch.nn.module and provides
    additional functionality for saving and loading models, as well as
    handling file system abstractions.

    There is one important requirement for all models in Modulus. They must
    have json serializable arguments in their __init__ function. This is
    required for saving and loading models and allow models to be instantiated
    from a checkpoint.

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None
    """

    _file_extension = ".mdlus"
    __model_checkpoint_version__ = "0.1.0"

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)

        sig = inspect.signature(cls.__init__)

        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )
        bound_args.apply_defaults()

        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            if k == "self":
                continue

            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(self, meta: Union[ModelMetaData, None] = None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0))
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _safe_members(tar, local_path):
        for member in tar.getmembers():
            if (
                ".." in member.name
                or os.path.isabs(member.name)
                or os.path.realpath(os.path.join(local_path, member.name)).startswith(
                    os.path.realpath(local_path)
                )
            ):
                yield member
            else:
                print(f"Skipping potentially malicious file: {member.name}")

    @classmethod
    def instantiate(cls, arg_dict: Dict[str, Any]) -> "Module":
        """Instantiate a model from a dictionary of arguments.

        Parameters
        ----------
        arg_dict : Dict[str, Any]
            Must contain ``__name__``, ``__module__``, and ``__args__`` keys.
        """
        _cls_name = arg_dict["__name__"]
        registry = ModelRegistry()
        if cls.__name__ == arg_dict["__name__"]:
            _cls = cls
        elif _cls_name in registry.list_models():
            _cls = registry.factory(_cls_name)
        else:
            try:
                _mod = importlib.import_module(arg_dict["__module__"])
                _cls = getattr(_mod, arg_dict["__name__"])
            except AttributeError:
                _cls = cls
        return _cls(**arg_dict["__args__"])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def save(self, file_name: Union[str, None] = None, verbose: bool = False) -> None:
        """Simple utility for saving just the model.

        Parameters
        ----------
        file_name : Union[str, None], optional
            File name (must end with ``.mdlus``). Defaults to ``<meta.name>.mdlus``.
        verbose : bool, optional
            Include git hash in metadata, by default False.
        """
        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            torch.save(self.state_dict(), local_path / "model.pt")

            with open(local_path / "args.json", "w") as f:
                json.dump(self._args, f)

            metadata_info = {
                "modulus_version": __version__,
                "mdlus_file_version": self.__model_checkpoint_version__,
            }

            if verbose:
                import git

                repo = git.Repo(search_parent_directories=True)
                try:
                    metadata_info["git_hash"] = repo.head.object.hexsha
                except git.InvalidGitRepositoryError:
                    metadata_info["git_hash"] = None

            with open(local_path / "metadata.json", "w") as f:
                json.dump(metadata_info, f)

            with tarfile.open(local_path / "model.tar", "w") as tar:
                for file in local_path.iterdir():
                    tar.add(str(file), arcname=file.name)

            if file_name is None:
                file_name = self.meta.name + ".mdlus"

            fs = _get_fs(file_name)
            fs.put(str(local_path / "model.tar"), file_name)

    @staticmethod
    def _check_checkpoint(local_path: str) -> bool:
        if not local_path.joinpath("args.json").exists():
            raise IOError("File 'args.json' not found in checkpoint")

        if not local_path.joinpath("metadata.json").exists():
            raise IOError("File 'metadata.json' not found in checkpoint")

        if not local_path.joinpath("model.pt").exists():
            raise IOError("Model weights 'model.pt' not found in checkpoint")

        with open(local_path.joinpath("metadata.json"), "r") as f:
            metadata_info = json.load(f)
            if (
                metadata_info["mdlus_file_version"]
                != Module.__model_checkpoint_version__
            ):
                raise IOError(
                    f"Model checkpoint version {metadata_info['mdlus_file_version']} is not "
                    f"compatible with current version {Module.__model_checkpoint_version__}"
                )

    def load(
        self,
        file_name: str,
        map_location: Union[None, str, torch.device] = None,
        strict: bool = True,
    ) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        file_name : str
            Checkpoint file name (``.mdlus``).
        map_location : Union[None, str, torch.device], optional
            Map location for loading weights. Defaults to model's device.
        strict : bool, optional
            Enforce strict key matching, by default True.
        """
        cached_file_name = _download_cached(file_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(Module._safe_members(tar, local_path))
                )

            Module._check_checkpoint(local_path)

            device = map_location if map_location is not None else self.device
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=device
            )
            self.load_state_dict(model_dict, strict=strict)

    @classmethod
    def from_checkpoint(cls, file_name: str) -> "Module":
        """Construct a model from a ``.mdlus`` checkpoint file.

        Parameters
        ----------
        file_name : str
            Checkpoint file name.

        Returns
        -------
        Module

        Raises
        ------
        IOError
            If ``file_name`` does not exist or is not a valid checkpoint.
        """
        cached_file_name = _download_cached(file_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(cls._safe_members(tar, local_path))
                )

            Module._check_checkpoint(local_path)

            with open(local_path.joinpath("args.json"), "r") as f:
                args = json.load(f)
            model = cls.instantiate(args)

            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )
            model.load_state_dict(model_dict)

        return model

    @staticmethod
    def from_torch(
        torch_model_class: torch.nn.Module, meta: ModelMetaData = None
    ) -> "Module":
        """Construct a Modulus module from a PyTorch module.

        Parameters
        ----------
        torch_model_class : torch.nn.Module
            PyTorch module class.
        meta : ModelMetaData, optional
            Meta data for the model, by default None.
        """

        class ModulusModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(
            zip(model_argnames[-len(model_defaults):], model_defaults)
        )

        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [
            inspect.Parameter(
                argname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults_dict.get(argname, inspect.Parameter.empty),
            )
            for argname in model_argnames
        ]
        init_signature = inspect.Signature(params)

        ModulusModel.__init__.__signature__ = init_signature

        new_class_name = f"{torch_model_class.__name__}ModulusModel"
        ModulusModel.__name__ = new_class_name

        registry = ModelRegistry()
        registry.register(ModulusModel, new_class_name)

        return ModulusModel

    @property
    def device(self) -> torch.device:
        """Get device the model is on."""
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Return the total number of learnable parameters."""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count
