import sys
from functools import lru_cache
from subprocess import DEVNULL, call
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# the following code was taken from
# https://github.com/teddykoker/torchsort/blob/main/setup.py
# which in turn was taken from
# https://github.com/idiap/fast-transformers/blob/master/setup.py

@lru_cache(None)
def cuda_toolkit_available():
  try:
    call(["nvcc"], stdout = DEVNULL, stderr = DEVNULL)
    return True
  except FileNotFoundError:
    return False

def compile_args():
  args = ["-fopenmp", "-ffast-math"]
  if sys.platform == "darwin":
    args = ["-Xpreprocessor", *args]
  return args

def ext_modules():
  if not cuda_toolkit_available():
    return []

  return [
    CUDAExtension(
      "flash_cosine_sim_attention_cuda",
      sources = ["flash_cosine_sim_attention/flash_cosine_sim_attention_cuda.cu"]
    )
  ]

# main setup code

setup(
  name = 'flash-cosine-sim-attention',
  packages = find_packages(exclude=[]),
  version = '0.0.17',
  license='MIT',
  description = 'Flash Cosine Similarity Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/flash-cosine-sim-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism'
  ],
  install_requires=[
    'torch>=1.10',
    'torchtyping'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  ext_modules = ext_modules(),
  cmdclass = {"build_ext": BuildExtension},
  include_package_data = True,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
