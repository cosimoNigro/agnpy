.. _emission_regions:


Emission Regions
================

The only emission region currently available in the package is a simple spherical plasmoid, commonly referred to as
*blob* in the literature.

Blob
----

The blob represents a spherical region of plasma streaming along the jet.
The physical quantities *needed* to define the :class:`~agnpy.emission_regions.Blob` are:

- its radius, :math:`R_{\rm b}`;
- its distance from the observer, expressed through the redshift :math:`z` of the host galaxy;
- the Doppler factor produced by the motion of the jet, :math:`\delta_{\rm D} = \frac{1}{\Gamma(1 - \beta\cos(\theta_{\rm s}))}`. Where :math:`\beta` is the blob relativistic velocity, :math:`\Gamma` its bulk Lorentz factor, and :math:`\theta_{\rm s}` the angle between the jet axis and the observer's line of sight;
- the bulk Lorentz factor of the jet, :math:`\Gamma`;
- the magnetic field tangled to the blob, :math:`B`, assumed to be uniform; 
- the energy distributions of particles accelerated in the blob.

Follows a snippet initialising the :class:`~agnpy.emission_regions.Blob` (using ``astropy`` quantities) with its physical quantities and an electrons disitribtuion.

.. literalinclude:: snippets/blob_snippet.py
   :lines: 1-26

Note that defining :math:`\delta_{\rm D}` and :math:`\Gamma` implicitly defines the viewing angle :math:`\theta_{\rm s}`. If you want to set the Doppler factor from :math:`\Gamma` and :math:`\theta_{\rm s}`, you can do so by using the :py:meth:`~agnpy.emission_regions.Blob.set_delta_D` function;

.. literalinclude:: snippets/blob_snippet.py
   :lines: 29-32

.. code-block:: text

   3.04

Since version ``0.3.0``, we can also specify a non-thermal proton distribution

.. literalinclude:: snippets/blob_snippet.py
   :lines: 35-38

.. code-block:: text

   * Spherical emission region
    - R_b (radius of the blob): 1.00e+16 cm
    - t_var (variability time scale): 4.13e-01 d
    - V_b (volume of the blob): 4.19e+48 cm3
    - z (source redshift): 0.07 redshift
    - d_L (source luminosity distance):1.00e+27 cm
    - delta_D (blob Doppler factor): 1.00e+01
    - Gamma (blob Lorentz factor): 1.00e+01
    - Beta (blob relativistic velocity): 9.95e-01
    - theta_s (jet viewing angle): 5.74e+00 deg
    - B (magnetic field tangled to the jet): 1.00e+00 G
    - xi (coefficient for 1st order Fermi acceleration) : 1.00e+00
   * electrons energy distribution
    - broken power law
    - k: 1.00e-08 1 / cm3
    - p1: 1.90
    - p2: 2.60
    - gamma_b: 1.00e+04
    - gamma_min: 1.00e+01
    - gamma_max: 1.00e+06
   * protons energy distribution
    - power law
    - k: 1.00e-01 1 / cm3
    - p: 2.30
    - gamma_min: 1.00e+01
    - gamma_max: 1.00e+06

As shown above, the :class:`~agnpy.emission_regions.Blob` can be printed at any moment to obtain a resume of the blob characterisitcs

Additional quantities computed by the blob
..........................................

The quantities listed in the previous section, needed to initialise the blob, are then used to evaluate several other physical quantities.
Among the most interesting are the energy densities in electrons and protons :math:`u_{\rm e},\;u_{\rm p}`; the energy density of the magnetic field, :math:`U_B`, and of the synchrotron radiation, :math:`u_{\rm ph,\, synch}`.
You can examine all of the physical quantities automatically computed by the :class:`~agnpy.emission_regions.Blob` in the API.

.. literalinclude:: snippets/blob_snippet.py
   :lines: 41-45

.. code-block:: text

   5.37e-06 erg / cm3
   2.43e-04 erg / cm3
   3.98e-02 erg / cm3
   3.75e-05 erg / cm3

