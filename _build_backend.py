"""
PEP 517 build backend for flash-cosine-sim-attention.

Most of the work is delegated to setuptools, driven purely by pyproject.toml.
The cuda extension cannot be expressed declaratively, so when torch and nvcc
are available the backend temporarily materializes a minimal setup.py (the
only mechanism torch's cpp_extension supports) to compile the kernel into a
platform wheel. Without torch or nvcc - for example on macOS - a pure python
wheel is produced, and the cuda kernel is compiled just in time on first use.
"""

import shutil
from pathlib import Path

from setuptools import build_meta as _build_meta

# re-export everything from setuptools.build_meta

from setuptools.build_meta import *  # noqa: F401,F403
from setuptools.build_meta import (  # noqa: F401
    build_sdist,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_wheel,
)

_ROOT = Path(__file__).parent
_SETUP_PY = _ROOT / 'setup.py'

# linux binary wheels must carry a manylinux tag to be accepted by pypi -
# the kernel itself only needs a reasonably modern glibc, so tag with the
# widest manylinux version that this environment can honestly claim

_PLAT_NAME = 'manylinux_2_28_x86_64'

_SETUP_PY_SOURCES = '''
# temporary setup.py materialized by the _build_backend to compile the
# cuda extension - removed again after the wheel build finishes

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

exec(open('flash_cosine_sim_attention/version.py').read())

setup(
  packages = find_packages(exclude=[]),
  ext_modules = [
    CUDAExtension(
      __cuda_pkg_name__,
      sources = ["flash_cosine_sim_attention/flash_cosine_sim_attention_cuda.cu"]
    )
  ],
  cmdclass = {"build_ext": BuildExtension},
  options = {"bdist_wheel": {"plat_name": "__PLAT_NAME__"}},
)
'''


_SETUP_PY_SOURCES = _SETUP_PY_SOURCES.replace('__PLAT_NAME__', _PLAT_NAME)


def _cuda_available():
    if shutil.which('nvcc') is None:
        return False

    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if not _cuda_available():
        return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)

    _SETUP_PY.write_text(_SETUP_PY_SOURCES)

    try:
        return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        _SETUP_PY.unlink(missing_ok=True)
