# module containing the synchrotron self-Compton (SSC) radiative process
# for a conical jet emission region.


import numpy as np
import astropy.units as u
from astropy.constants import c, h, sigma_T

from ..compton.kernels import isotropic_kernel
from ..utils.math import axes_reshaper
from ..utils.conversion import nu_to_epsilon_prime, mec2
from ..radiative_process import RadiativeProcess
from ..utils.synchrotron import nu_synch_peak
from ..synchrotron.synchrotron_cone import SynchrotronCone

__all__ = ["SynchrotronSelfComptonCone"]

c = c.cgs
h = h.cgs
sigma_T = sigma_T.cgs
mec2 = mec2.cgs


class SynchrotronSelfComptonCone(RadiativeProcess):
    r"""Synchrotron self-Compton (SSC) emission from a conical jet.

    Parameters
    ----------
    cone : :class:`~agnpy.emission_regions.cone.Cone`
        conical emission region and electron distribution
    ssa : bool
        whether to attenuate the local seed photon field by synchrotron
        self-absorption before it is Compton up-scattered. Default ``False``.
    integrator : func
        function to be used for integration (default `numpy.trapz`)
    nu_seed_size : int
        number of points in the internal fluid-frame seed-photon
        frequency grid (default 100)
    """

    def __init__(self, cone, ssa=False, integrator=np.trapz, nu_seed_size=100):
        self.cone = cone
        self.ssa = ssa
        self.integrator = integrator
        self.nu_seed_size = nu_seed_size
        # re-used to compute (and, if requested, self-absorb) the local
        # synchrotron seed photon field

    @property
    def nu_seed(self):
        r"""Fluid-frame frequency grid used to sample the local synchrotron
        seed photon field.
        """
        cone = self.cone
        nu_pk_max = nu_synch_peak(cone.B_x[0], cone.gamma_e[-1])
        nu_pk_min = nu_synch_peak(cone.B_x[-1], cone.gamma_e[0])
        return np.logspace(
            np.log10(nu_pk_min.to_value("Hz")) - 2, 
            # the base of the jet (largest B) together with the highest electron 
            # energies) sets the upper end of the range
            np.log10(nu_pk_max.to_value("Hz")) +2,
            # the far end of the jet (smallest B) lowest surviving energies,
            # sets the lower end
            self.nu_seed_size,
        ) * u.Hz

    def local_seed_photon_field(self):
        r"""Local (fluid-frame) synchrotron seed photon energy density in
        every slice of the jet, :math:`u(\epsilon, x)`.

        Returns
        -------
        epsilon_seed : :class:`~numpy.ndarray`
            dimensionless (fluid-frame) seed photon energies, shape ``(N_nu,)``
        u_epsilon_x : :class:`~astropy.units.Quantity`
            local photon energy density, shape ``(N_x, N_nu)``, units
            ``erg cm-3``
        """
        nu_seed = self.nu_seed

        R_x = self.cone.R_x
        B_x_reshaped = self.cone.B_x[:, None, None]
        gamma_e_reshaped = self.cone.gamma_e[None, :, None]
        nu_seed_reshaped = nu_seed[None, None, :]
        N_e_xg_reshaped = self.cone.N_e_xg[:, :, None]
        t_esc = 3 * R_x[:,None] / c

        epsilon_seed = nu_to_epsilon_prime(nu_seed, z=0, delta_D=1)

        L_x_nu_fluid, _, _ = SynchrotronCone.evaluate_sed_flux(
            nu_seed_reshaped,
            B_x_reshaped,
            self.cone.x,
            gamma_e_reshaped,
            N_e_xg_reshaped,
            integrator=self.integrator,
        )
        u_nu_x = ( L_x_nu_fluid * t_esc / (np.pi * R_x[:, None]**2))

        u_epsilon_x = (u_nu_x * mec2 / h).to("erg cm-3")
        
        return epsilon_seed, u_epsilon_x

    def sed_flux(self, nu_obs):
        r"""Observer-frame SSC SED, :math:`\nu F_\nu`.

        Parameters
        ----------
        nu_obs : :class:`~astropy.units.Quantity`
            observed frequencies (Hz) at which to evaluate the SED

        Returns
        -------
        :class:`~astropy.units.Quantity`
            SSC SED :math:`\nu F_\nu` in ``erg cm-2 s-1``, shape ``(N_nu,)``
        """
        gamma = self.cone.gamma_e

        epsilon_seed, u_epsilon_x = self.local_seed_photon_field()
        epsilon_s = nu_to_epsilon_prime(nu_obs, self.cone.z, self.cone.delta_D)

        # kernel depends only on (gamma, epsilon_seed, epsilon_s), not on x,
        # so it is computed once and re-used for every slice
        _gamma, _epsilon, _epsilon_s = axes_reshaper(gamma, epsilon_seed, epsilon_s)
        kernel = isotropic_kernel(_gamma, _epsilon, _epsilon_s)

        N_x = len(self.cone.x)
        local_ssc = []
        for i in range(N_x):
            N_e_i = self.cone.N_e_xg[i][:, None, None]
            u_epsilon_i = u_epsilon_x[i][None, :, None]

            integrand = (
                u_epsilon_i / np.power(_epsilon, 2)
                * N_e_i / np.power(_gamma, 2)
                * kernel
            )
            integral_gamma = self.integrator(integrand, gamma, axis=0)
            integral_epsilon = self.integrator(
                integral_gamma, epsilon_seed, axis=0
            ).reshape(epsilon_s.shape)

            local_ssc.append(
            0.75 * c * sigma_T * np.power(epsilon_s, 2) * integral_epsilon 
            )

        # shape (N_x, N_nu), units erg s-1 cm-1 (emissivity per unit jet length)
        local_ssc = u.Quantity(local_ssc)

        L_ssc_fluid = self.integrator(local_ssc, self.cone.x, axis=0)
        prefactor = np.power(self.cone.delta_D, 4) / (4 * np.pi * np.power(self.cone.d_L, 2))

        self.sed = (prefactor * L_ssc_fluid).to("erg cm-2 s-1")
        return self.sed

    def sed_luminosity(self, nu_obs):
        r"""SSC luminosity SED, :math:`\nu L_\nu` in erg s-1."""
        sphere = 4 * np.pi * np.power(self.cone.d_L, 2)
        return (sphere * self.sed_flux(nu_obs)).to("erg s-1")

    def sed_peak_flux(self, nu_obs):
        """Peak flux of the SSC SED over the given frequency grid."""
        return self.sed_flux(nu_obs).max()

    def sed_peak_nu(self, nu_obs):
        """Frequency at which the SSC SED peaks over the given grid."""
        idx_max = self.sed_flux(nu_obs).argmax()
        return nu_obs[idx_max]