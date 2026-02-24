# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import version

# Monkey-patch to support meta device in fused optimizer operations
import torch.utils._foreach_utils as _foreach_utils
import torch.optim.optimizer as _optimizer_module

_orig_get_fused_kernels_supported_devices = _foreach_utils._get_fused_kernels_supported_devices

def _patched_get_fused_kernels_supported_devices():
    """Add 'meta' device support for fused optimizer operations."""
    return _orig_get_fused_kernels_supported_devices() + ["meta"]

# Patch both the source module and the optimizer module's imported reference
_foreach_utils._get_fused_kernels_supported_devices = _patched_get_fused_kernels_supported_devices
_optimizer_module._get_fused_kernels_supported_devices = _patched_get_fused_kernels_supported_devices

# Import to register quantization modules.
import torchtitan.components.quantization  # noqa: F401

try:
    __version__ = version("torchtitan")
except Exception as e:
    __version__ = "0.0.0+unknown"
