.. _fit:


Using agnpy radiative processes to fit a MWL SED
================================================
``agnpy`` includes `sherpa <https://sherpa.readthedocs.io/>`_ and `gammapy <https://docs.gammapy.org/>`_ wrappers that allow the user to fit the broad-band emission of jetted AGN.
The wrappers consider the combination of several radiative processes and return a ``model`` object that can be used by the fitting routine of the package.
In this documentation page, we gather several examples showing how to obtain the best-fit parameters for these models via a :math:`\chi^2` minimisation.
The data to be fitted are flux points, representing the flux measurement of an instrument in one or more energy bins.


Prerequisites
-------------
Note that the ``agnpy.fit`` module, containing the wrappers, needs ``gammapy`` and ``sherpa`` to be installed.
These packages are not among the basic dependencies of ``agnpy`` and will not be installed automatically if ``agnpy`` is installed via ``conda`` or ``pip``.
Check the documentation of the relative packages for their installation instructions.
The following warning message will be printed by ``agnpy`` if the packages not being installed:

``WARNING:root:sherpa and gammapy are not installed, the agnpy.fit module cannot be used``

In this case, all the submodules, except ``agnpy.fit``, can be used without problems.


Fit using the ``sherpa`` wrapper
--------------------------------
.. toctree::
   :maxdepth: 1

   tutorials/ssc_sherpa_fit.ipynb
   tutorials/ec_dt_sherpa_fit.ipynb


Fit using the ``Gammapy`` wrapper
---------------------------------
For Gammapy we show, beside the simple :math:`\chi^2` minimisation, how to use a Monte Carlo Markov Chain (MCMC) to perform the fit.

.. toctree::
   :maxdepth: 1

   tutorials/ssc_gammapy_fit.ipynb
   tutorials/ssc_gammapy_mcmc_fit.ipynb
   tutorials/ec_dt_gammapy_fit.ipynb


Loading MWL SED data with ``Gammapy`` and ``sherpa``
----------------------------------------------------
We provide, in ``agnpy.fit``, functions that can diretly load flux points in ``sherpa`` and ``Gammapy`` data objects.
The SED format that these functions can read follows the one proposed within the `Data Formats for Gamma-ray Astronomy <https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html>`_.
More specifically, the flux point format that can be currently read by the ``load_sherpa_flux_points`` and ``load_gammapy_flux_points`` functions has the following **mandatory** columns:

* ``e_ref``, representing the energy of the center of the flux points (``eV``);
* ``e2dnde``, representing the energy flux (``erg / (cm2 s)``);
* ``e2dnde_errn``, representing the lower error on the energy flux (``erg / (cm2 s)``);
* ``e2dnde_errp``, representing the higher error on the energy flux (``erg / (cm2 s)``);
* ``instrument``, representing the instrument that performed the measurement.
