# tests on the utils module
import pytest
import numpy as np
import agnpy.utils.geometry as geom
import astropy.units as u
from astropy.constants import m_e, c, sigma_T, k_B
from agnpy.targets import RingDustTorus
from agnpy.absorption import Absorption
from agnpy.utils.math import axes_reshaper

mec2 = m_e.to("erg", equivalencies=u.mass_energy())


def sigma_pp(b):
    """pair production cross section"""
    return (
        sigma_T
        * 3.0
        / 16.0
        * (1 - b ** 2)
        * (2 * b * (b ** 2 - 2) + (3 - b ** 4) * np.log((1 + b) / (1 - b)))
    )


class TestAbsorptionMuS:
    """test absorption functions for calculations with mu_s !=1"""

    def test_tau_dt_mu_s_simple(self):
        """
        order of magnitude test comparing with simplified calculations
        the case of a perpendicularly moving photon starting at r~R_re
        """
        r = 1.0e17 * u.cm  # distance at which the photon starts
        mu_s = 0.0  # angle of propagation

        L_disk = 2e46 * u.Unit("erg s-1")
        xi_DT = 0.1
        temp = 1000 * u.K
        R_DT = r  # radius of DT: assume the same as the distance r
        dt = RingDustTorus(L_disk, xi_DT, temp, R_dt=R_DT)

        nu_ref = np.logspace(26, 32, 60) * u.Hz

        # absorption at mu_s
        abs_dt_mu_s = Absorption(dt, r, z=0, mu_s=mu_s)
        tau_dt_mu_s = abs_dt_mu_s.tau_mu_s(nu_ref)

        eps = (2.7 * temp * k_B).to("eV")  # energy of soft photons
        # soft photon density
        nph = (L_disk * xi_DT / (4 * np.pi * (r ** 2 + R_DT ** 2)) / c / eps).to("cm-3")

        E = nu_ref.to("eV", equivalencies=u.spectral())
        cospsi = 0  # assume perpendicular scattering
        beta2 = 1 - 2 * mec2 ** 2 / (E * eps * (1 - cospsi))
        beta2[beta2 < 0] = 0  # below the threshold
        # for tau calculations we assume that gamma ray moves
        # roughtly the characteristic distance of ~r~R_DT
        tau_my = (sigma_pp(np.sqrt(beta2)) * nph * r * (1 - cospsi)).to("")

        max_agnpy = max(tau_dt_mu_s)
        max_my = max(tau_my)
        max_pos_agnpy = nu_ref[np.argmax(tau_dt_mu_s)]
        max_pos_my = nu_ref[np.argmax(tau_my)]
        # very rough calculations so allowing for
        # 15% accuracy in peak maximum and 30% in peak position
        assert np.isclose(max_agnpy, max_my, atol=0, rtol=0.15)
        assert np.isclose(max_pos_agnpy, max_pos_my, atol=0, rtol=0.3)

    @pytest.mark.parametrize("mu_s", np.linspace(0, 0.9, 5))
    def test_tau_dt_mu_s_far(self, mu_s):
        """
        comparing the DT absorption for mu_s !=1 with point source simplification
        """
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_DT = 0.1
        temp = 1000 * u.K
        R_DT = 1.0e17 * u.cm  # radius of DT
        dt = RingDustTorus(L_disk, xi_DT, temp, R_dt=R_DT)

        r = 10 * R_DT  # distance at which the photon starts

        nu_ref = np.logspace(26, 32, 60) * u.Hz

        # absorption at mu_s
        abs_dt_mu_s = Absorption(dt, r, z=0, mu_s=mu_s)
        tau_dt_mu_s = abs_dt_mu_s.tau_mu_s(nu_ref)

        uu = np.logspace(-5, 5, 100) * r
        _u, _nu_ref = axes_reshaper(uu, nu_ref)

        eps = (2.7 * temp * k_B).to("eV")  # energy of soft photons
        # distance from the DT
        _x = np.sqrt(r * r + _u * _u + 2 * mu_s * _u * r)

        # soft photon density
        _nph = (L_disk * xi_DT / (4 * np.pi * _x ** 2) / c / eps).to("cm-3")

        _E = _nu_ref.to("eV", equivalencies=u.spectral())
        _cospsi = (_u ** 2 + _x ** 2 - r ** 2) / (2 * _u * _x)

        _beta2 = 1 - 2 * mec2 ** 2 / (_E * eps * (1 - _cospsi))
        _beta2[_beta2 < 0] = 0

        integrand = sigma_pp(np.sqrt(_beta2)) * _nph * (1 - _cospsi)
        tau_my = (np.trapz(integrand, uu, axis=0)).to("")

        print(tau_my / tau_dt_mu_s)

        max_agnpy = max(tau_dt_mu_s)
        max_my = max(tau_my)
        max_pos_agnpy = nu_ref[np.argmax(tau_dt_mu_s)]
        max_pos_my = nu_ref[np.argmax(tau_my)]

        # if r>>R_re this should be pretty precise, allowing for 10% accuracy
        assert np.isclose(max_agnpy, max_my, atol=0, rtol=0.1)
        assert np.isclose(max_pos_agnpy, max_pos_my, atol=0, rtol=0.1)
