.. _emission_regions:


Emission regions
================

At the moment the only emission region available in the code is a simple spherical plasmoid, commonly referred to as **blob** in the literature.
For more details on the electrons spectra you can read the :ref:`spectra` API.

Follows an example of how to initialise a `Blob` using `astropy` quantities:

.. code-block:: python

	import astropy.units as u
	from astropy.coordinates import Distance
	from agnpy.emission_regions import Blob

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

API
---

.. automodule:: agnpy.emission_regions
   :noindex:
   :members: Blob 