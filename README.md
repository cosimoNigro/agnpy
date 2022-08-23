# agnpy
<p align="center">
  <img width="60%" src="docs/_static/logo.png">
</p>

Modelling Active Galactic Nuclei radiative processes with python.    
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

## descritpion
`agnpy` focuses on the numerical computation of the photon spectra produced by leptonic radiative processes in jetted Active Galactic Nuclei (AGN).

## agnpy binder
Run this repository in binder    
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cosimoNigro/agnpy/HEAD)

## acknowledging and citing agnpy
As a general acknowledgement of `agnpy` usage, we recommend citing the agnpy release paper.
Additionaly, to specify which version of `agnpy` is being used, that version's zenodo record can be cited.
We recommend citing both.

At the following links you can find:

 * [the agnpy release paper (for a general citation)](https://ui.adsabs.harvard.edu/abs/2022A%26A...660A..18N/abstract);
 * the zenodo record (for citing a specific version) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4055175.svg)](https://doi.org/10.5281/zenodo.4055175)

## documentation and quickstart
You are invited to check the documentation at https://agnpy.readthedocs.io/en/latest/.    
To get familiar with the code you can run the notebooks in the `tutorials` section
of the documentation.

## dependencies
The only dependencies are:

* [numpy](https://numpy.org) managing the numerical computation;

* [astropy](https://www.astropy.org) managing physical units and astronomical distances.

* [matplotlib](https://matplotlib.org) for visualisation and reproduction of the tutorials.

* [scipy](https://www.scipy.org/) for interpolation 

## installation
The code is available in the [python package index](https://pypi.org/project/agnpy/) and can be installed via `pip`

```bash
pip install agnpy
```

The code can also be installed using `conda`

```bash
conda install -c conda-forge agnpy
```

## tests
A test suite is available in [`agnpy/tests`](https://github.com/cosimoNigro/agnpy/tree/master/agnpy/tests), to run it just type
`pytest` in the main directory.

## shields
[![CI test](https://github.com/cosimoNigro/agnpy/actions/workflows/test.yml/badge.svg)](https://github.com/cosimoNigro/agnpy/actions/workflows/test.yml)

[![Upload to PIP](https://github.com/cosimoNigro/agnpy/actions/workflows/pip-upload.yml/badge.svg)](https://github.com/cosimoNigro/agnpy/actions/workflows/pip-upload.yml)

[![Documentation Status](https://readthedocs.org/projects/agnpy/badge/?version=latest)](https://agnpy.readthedocs.io/en/latest/?badge=latest)

![](https://codecov.io/gh/cosimoNigro/agnpy/branch/master/graph/badge.svg)

![](http://img.shields.io/pypi/v/agnpy.svg?text=version)
