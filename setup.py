from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="core_synchrotron",
        sources=["agnpy/core_synchrotron.pyx"],
        libraries=["m", "m"],  # Unix-like specific
    ),
    Extension(
        name="core_compton",
        sources=["agnpy/core_compton.pyx"],
        libraries=["m", "m"],  # Unix-like specific
    ),
]

setup(ext_modules=cythonize(ext_modules, annotate=True))
