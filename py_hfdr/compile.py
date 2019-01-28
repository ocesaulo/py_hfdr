from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import scipy

ext_modules = [
    Extension("fast_chirp_handling",  ["fast_chirp_handling.pyx"],
              include_dirs=[numpy.get_include(), scipy.get_include()],
              ),


]

setup(
    name='Fast Chirp Handling',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules)
)
