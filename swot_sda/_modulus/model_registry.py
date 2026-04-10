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
# Source: modulus/registry/model_registry.py, version 0.7.0a0

from importlib.metadata import EntryPoint, entry_points
from typing import List, Union

import importlib_metadata


class ModelRegistry:
    """Model registry following a Borg singleton pattern.

    Discovers registered models via the ``modulus.models`` entry-point group,
    so models from the original ``modulus`` package installation remain
    accessible even when using this vendored copy.
    """

    _shared_state = {"_model_registry": None}

    def __new__(cls, *args, **kwargs):
        obj = super(ModelRegistry, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if cls._shared_state["_model_registry"] is None:
            cls._shared_state["_model_registry"] = cls._construct_registry()
        return obj

    @staticmethod
    def _construct_registry() -> dict:
        registry = {}
        entrypoints = entry_points(group="modulus.models")
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    def register(self, model, name: Union[str, None] = None) -> None:
        """Register a model class under ``name`` (defaults to ``model.__name__``)."""
        # Avoid a hard import of the Module base class here to prevent circular imports.
        if name is None:
            name = model.__name__

        if name in self._model_registry:
            raise ValueError(f"Name {name} already in use")

        self._model_registry[name] = model

    def factory(self, name: str):
        """Return a registered model class by name."""
        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, (EntryPoint, importlib_metadata.EntryPoint)):
                model = model.load()
            return model

        raise KeyError(f"No model is registered under the name {name}")

    def list_models(self) -> List[str]:
        """Return names of all registered models."""
        return list(self._model_registry.keys())

    def __clear_registry__(self):
        # NOTE: used for testing only
        self._model_registry = {}

    def __restore_registry__(self):
        # NOTE: used for testing only
        self._model_registry = self._construct_registry()
