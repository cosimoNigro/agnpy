.. _synchrotron:


Synchrotron Radiation
=====================

The synchrotron radiation is computed following the approach of [DermerMenon2009]_ and [Finke2008]_.

Expanding the example in :ref:`emission_regions`, it is here illustrated how to produce a synchrotron spectral energy distribution (SED) staring from a :class:`~agnpy.emission_regions.Blob`. The Synchrotron Self Absorption (SSA) mechanism can be considered. 

.. code-block:: python

	import numpy as np
	import astropy.units as u
	from astropy.coordinates import Distance
	from agnpy.emission_regions import Blob
	from agnpy.synchrotron import Synchrotron
	import matplotlib.pyplot as plt

	# set the spectrum normalisation (total energy in electrons in this case)
	spectrum_norm = 1e48 * u.Unit("erg") 
	# define the spectral function through a dictionary
	spectrum_dict = {
	    "type": "PowerLaw", 
	    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7}
	}
	R_b = 1e16 * u.cm
	B = 1 * u.G
	z = Distance(1e27, unit=u.cm).z
	delta_D = 10
	Gamma = 10
	blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

to initialise the synchrotron radiation the :class:`~agnpy.emission_regions.Blob` instance has to be passed to the :class:`~agnpy.synchrotron.Synchrotron` class initialiser

.. code-block:: python

	synch = Synchrotron(blob)

the optional argument `ssa` specifies if self absorption has to be taken into account (by default it is not)

.. code-block:: python

	synch_ssa = Synchrotron(blob, ssa=True)

to compute the spectral energy distribution (SED), an array of frequencies (astropy units) has to be passed to the :func:`~agnpy.synchrotron.Synchrotron.sed_flux` function

.. code-block:: python

	nu = np.logspace(8, 23) * u.Hz
	synch_sed = synch.sed_flux(nu)
	print(synch_sed)

this produces an array of :class:`~astropy.units.Quantity`

.. code-block:: text

	[9.07847669e-16 2.32031314e-15 5.92493269e-15 1.51066953e-14
	3.84226036e-14 9.73303142e-14 2.44919574e-13 6.09602590e-13
	1.49002575e-12 3.53274373e-12 7.95174501e-12 1.63760127e-11
	2.91395265e-11 4.20897124e-11 4.96023285e-11 5.36547089e-11
	5.75763018e-11 6.17811534e-11 6.62930892e-11 7.11345358e-11
	7.63295578e-11 8.19039772e-11 8.78855014e-11 9.43038616e-11
	1.01190960e-10 1.08581027e-10 1.16510791e-10 1.25019658e-10
	1.34149896e-10 1.43946820e-10 1.54458961e-10 1.65738143e-10
	1.77839337e-10 1.90819907e-10 2.04737266e-10 2.19642496e-10
	2.35563787e-10 2.52464517e-10 2.70139515e-10 2.87965730e-10
	3.04330969e-10 3.15437517e-10 3.13250591e-10 2.83748181e-10
	2.12000058e-10 1.06769402e-10 2.42794829e-11 1.12784249e-12
	2.22960744e-15 8.03667875e-21] erg / (cm2 s)
	
Let us examine the different SEDs produced by the normal and self-absorbed synchrotron processes

.. code-block:: python

	synch_sed_ssa = synch_ssa.sed_flux(nu)
	plt.loglog(nu, synch_sed, color="k", lw=2, label="synchr.")
	plt.loglog(nu, synch_sed_ssa, lw=2, ls="--", color="gray", label="self absorbed synchr.")
	plt.xlabel(r"$\nu\,/\,\mathrm{Hz}$")
	plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
	plt.legend()
	plt.show()

.. image:: _static/synch.png
    :width: 500px
    :align: center

For more examples of Synchrotron radiation and cross-checks of literature results, check the 
check the `tutorial notebook on synchrotron and sycnrotron self Compton <tutorials/synchrotron_self_compton.html>`_.


API
---

.. automodule:: agnpy.synchrotron
   :noindex:
   :members: Synchrotron 