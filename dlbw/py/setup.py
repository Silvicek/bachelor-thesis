from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'nao',
  ext_modules = cythonize("nao.pyx",language='c++'), 
)
