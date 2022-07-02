from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('pairwise_dist', ['pairwise_dist.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))python setup.py build_ext --inplace