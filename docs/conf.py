# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "agnpy"
copyright = "2019, Cosimo Nigro"
author = "Cosimo Nigro"

# The full version, including alpha/beta/rc tags
release = "0.0.10"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "numpydoc",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

numfig = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# set main node of the documentation to index.rst (contents.rst is the default)
master_doc = "index"

# dictionary with external packages references
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("http://docs.astropy.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org", None),
}

# latex support
mathjax_config = {
    "TeX": {
        "Macros": {
            "Beta": r"{\mathcal{B}}",
            "uunits": r"{{\rm erg}\,{\rm cm}^{-3}}",
            "diff": r"{\mathrm{d}}",
            "utransform": r"{\Gamma^3 (1 + \Beta \mu')^3}",
            "rtilde": r"\tilde{r}",
            "ltilde": r"\tilde{l}",
        }
    }
}
