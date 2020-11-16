.. _absorption:


Absorption by :math:`\gamma`-:math:`\gamma` pair production
===========================================================
The photon fields that represent the target for the Compton scattering might re-absorb 
the scattered photons via :math:`\gamma`-:math:`\gamma` pair production. `agnpy` computes the 
optical depth (or opacity) :math:`\tau_{\gamma \gamma}` as a function of the frequency :math:`\nu` 
of the photon hitting the target. Photoabsorption results in an attenuation of the photon flux 
by a factor :math:`\exp(-\tau_{\gamma \gamma})`.

Absorption on target photon fields
----------------------------------

In the following example we compute the optical depths produced by the disk, the broad line region and the dust torus photon fileds

.. code-block:: python

	import numpy as np
	import astropy.units as u
	import astropy.constants as const
	from agnpy.emission_regions import Blob
	from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
	from agnpy.absorption import Absorption

	# define the blob
	spectrum_norm = 1e47 * u.erg
	parameters = {"p": 2.8, "gamma_min": 10, "gamma_max": 1e6}
	spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
	R_b = 1e16 * u.cm
	B = 0.56 * u.G
	z = 0
	delta_D = 40
	Gamma = 40
	blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

	# disk parameters
	M_BH = 1.2 * 1e9 * const.M_sun.cgs
	L_disk = 2 * 1e46 * u.Unit("erg s-1")
	eta = 1 / 12
	R_in = 6 
	R_out = 200 
	disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)

	# blr definition
	csi_line = 0.024
	R_line = 1e17 * u.cm
	blr = SphericalShellBLR(L_disk, csi_line, "Lyalpha", R_line)

	# dust torus definition
	T_dt = 1e3 * u.K
	csi_dt = 0.1
	dt = RingDustTorus(L_disk, csi_dt, T_dt)

as for the :class:`~agnpy.compton.ExternalCompton` radiation, the absortpion can 
be computed passing to the :class:`~agnpy.absorption.Absorption` class the 
:class:`~agnpy.emission_regions.Blob` and :class:`~agnpy.targets.SSDisk` 
(or any other target) instances. 
Remember also to set the distance between the blob and the target photon field (:math:`r`)

.. code-block:: python

	# consider a fixed distance of the blob from the target fields 
	r = 1.1e16 * u.cm

	absorption_disk = Absorption(blob, disk, r=r)
	absorption_blr = Absorption(blob, blr, r=r)
	absorption_dt = Absorption(blob, dt, r=r)

	E = np.logspace(0, 5) * u.GeV
	nu = E.to("Hz", equivalencies=u.spectral())

	tau_disk = absorption_disk.tau(nu)
	tau_blr = absorption_blr.tau(nu)
	tau_dt = absorption_dt.tau(nu)


	# plot the absorption
	import matplotlib.pyplot as plt
	from agnpy.utils.plot import load_mpl_rc
	# matplotlib adjustments
	load_mpl_rc()

	fig, ax = plt.subplots()
	ax.loglog(E, tau_disk, lw=2, ls="-", label = "SS disk")
	ax.loglog(E, tau_blr, lw=2, ls="--", label = "spherical shell BLR")
	ax.loglog(E, tau_dt, lw=2, ls="-.", label = "ring dust torus")
	ax.legend()
	ax.set_xlabel("E / GeV")
	ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
	ax.set_xlim([1, 1e5])
	ax.set_ylim([1e-3, 1e5])
	plt.show()

.. image:: _static/tau.png
    :width: 500px
    :align: center


API
---

.. automodule:: agnpy.absorption
   :noindex:
   :members: Absorption 