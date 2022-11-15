.. _synchrotron:


Synchrotron Radiation
=====================

The synchrotron radiation is computed following the approach of [DermerMenon2009]_ and [Finke2008]_.

It is here illustrated how to produce a synchrotron spectral energy distribution (SED) staring from a :class:`~agnpy.emission_regions.Blob`. 

.. plot:: snippets/synchro_snippet.py
   :include-source:

Note two aspects valid for all the radiative processes in ``agnpy``:

1. to initialise any radiative process in ``agnpy``, the instance of the emission region class (:class:`~agnpy.emission_regions.Blob` in this case) has to be passed to the initialiser of the radiative process class (:class:`~agnpy.synchrotron.Synchrotron` in this case)

.. code-block:: python

   synch = Synchrotron(blob)

2. the SEDs are always compute over an array of frequencies (astropy units), passed to the ``sed_flux`` function

.. code-block:: python

   nu = np.logspace(8, 23) * u.Hz
   sed = synch.sed_flux(nu)

this produces an array of :class:`~astropy.units.Quantity`.

For more examples of Synchrotron radiation and cross-checks of literature results, check the 
check the `tutorial notebook on synchrotron and sycnrotron self Compton <tutorials/synchrotron_self_compton.html>`_.
