from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "core_synch_cy",
        sources=["core_synch_cy.pyx"],
        libraries=["m"],  # Unix-like specific
    )
]

setup(ext_modules=cythonize(ext_modules, annotate=True))
