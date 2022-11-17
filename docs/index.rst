.. image:: _static/logo.png
    :width: 400px
    :align: center


agnpy Documentation
===================
`agnpy` focuses on the numerical computation of the photon spectra produced by particles radiative processes in jetted Active Galactic Nuclei (AGN).


Description
-----------
References
..........
Notation and basic formulas are borrowed from [DermerMenon2009]_ which constitutes the fundamental reference for this package.
The implementation of synchrotron and synchrotron self Compton radiative processes relies on [DermerMenon2009]_ and [Finke2008]_.
[Dermer2009]_ and [Finke2016]_ are instead the main references for the external Compton and :math:`\gamma\gamma` absorption implementation.

Implementation
..............
The numerical operations are delegated to `numpy arrays <https://numpy.org>`_, all the physical quantities computed are casted as `astropy Quantities <https://docs.astropy.org/en/stable/units/>`_.


License
-------
The code is licensed under a `BSD-3-Clause License <https://opensource.org/licenses/BSD-3-Clause>`_ (see ``LICENSE.md`` in the main repository).


Installation
------------
The code is available in the `python package index <https://pypi.org/project/agnpy/>`_ and can be installed via ``pip``

.. code-block:: bash

   pip install agnpy

The code can also be installed with ``conda``

.. code-block:: bash

   conda install -c conda-forge agnpy


Dependencies
------------
The only dependencies are:

* `numpy <https://numpy.org>`_ managing the numerical computation;

* `astropy <https://www.astropy.org>`_ managing physical units and astronomical distances;

* `scipy <https://scipy.org>`_ for interpolation;

* `matplotlib <https://matplotlib.org>`_ for visualisation and reproduction of the tutorials.

`sherpa <https://sherpa.readthedocs.io/en/latest/>`_ and `gammapy <https://gammapy.org/>`_ are additionaly required to use `agnpy` for fitting, i.e. to use the wrappers in the ``agnpy.fit`` module.


Overview
--------
The documentation includes several tutorial jupyter notebooks providing examples of applications of the code functionalities.

.. toctree::
   :maxdepth: 2

   spectra
   emission_regions
   synchrotron
   targets
   tutorials/dt_thermal_emission.ipynb
   tutorials/energy_densities.ipynb
   compton
   tutorials/synchrotron_self_compton.ipynb
   tutorials/external_compton.ipynb
   absorption
   tutorials/absorption_targets.ipynb
   spectral_constraints
   fit
   acknowledging
   bibliography
   agnpy


