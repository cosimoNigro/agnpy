agnpy docs
=================================

Description
-----------
References
..........
Notation and basic formulas are borrowed from [DermerMenon2009]_ which constitutes the fundamental reference for this code. The implementation of synchrotron and synchrotron self-Compton radiative processes relies on [DermerMenon2009]_ and [Finke2008]_; for the external Compton radiative processes [Dermer2009]_ and [Finke2016]_ are the main references.

Implementation
..............
`agnpy` focuses on the numerical computation of the photon spectra produced by leptonic radiative processes in jetted Active Galactic Nuclei (AGN).    
The numerical operations are delegated to `numpy arrays <https://numpy.org>`_, all the physical quantities computed are casted as `astropy Quantities <https://docs.astropy.org/en/stable/units/>`_.

License
-------
The code is licensed under `GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.html>`_ (see `LICENSE.md` in the main directory).

Overview
--------

.. toctree::
   :maxdepth: 2

   emission_regions
   synchrotron
   compton
   references

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
