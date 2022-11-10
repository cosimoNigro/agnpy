# tests the emission region modules
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
import pytest
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.utils.conversion import Gauss_cgs_unit


class TestBlob:
    """Class grouping all tests related to the Blob emission region."""

    def test_blob_properties(self):
        """Test that the blob properties are properly updated if the basic
        attributes are modified."""
        blob = Blob()
        blob.R_b = 1e17 * u.cm
        blob.z = 2.1
        blob.delta_D = 8
        blob.Gamma = 5
        blob.B = 2 * u.G
        assert u.isclose(
            blob.V_b, 4.1887902e51 * u.Unit("cm3"), atol=0 * u.Unit("cm3"), rtol=1e-3
        )
        assert u.isclose(blob.d_L, 5.21497473e28 * u.cm, atol=0 * u.cm, rtol=1e-3)
        assert np.isclose(blob.Beta, 0.9798, atol=0, rtol=1e-3)
        assert u.isclose(blob.theta_s, 5.67129265 * u.deg, atol=0 * u.deg, rtol=1e-3)
        assert u.isclose(
            blob.B_cgs,
            2 * u.Unit(Gauss_cgs_unit),
            atol=0 * u.Unit(Gauss_cgs_unit),
            rtol=1e-3,
        )
        # test the manual setting of delta_D
        blob.set_delta_D(Gamma=10, theta_s=20 * u.deg)
        assert np.allclose(blob.delta_D, 1.53804, atol=0, rtol=1e-3)

    def test_particles_spectra(self):
        """Test for the blob properties related to the particle spectra."""
        n_e = BrokenPowerLaw()
        n_p = PowerLaw(mass=m_p)
        # first we initialise the blob without the protons distribution
        blob = Blob(n_e=n_e)
        # assert that the proton distribution is not set
        with pytest.raises(AttributeError):
            assert blob.gamma_p
            assert blob.n_p
        # now let us set the proton distribution, should automatically set gamma_p
        blob.n_p = n_p
        assert blob.gamma_p is not None
        # change the grid of Lorentz factors
        gamma = np.logspace(2, 6, 50)
        blob.set_gamma_e(len(gamma), gamma[0], gamma[-1])
        assert np.array_equal(blob.gamma_e, gamma)
        blob.set_gamma_p(len(gamma), gamma[0], gamma[-1])
        assert np.array_equal(blob.gamma_p, gamma)

    def test_particles_densities(self):
        """Test different methods to initialise the protons and electrons
        distributions. Check the attributes related to the particles
        distributions computed by the blob.
        """
        blob = Blob()

        # intialise from total particle density
        n_tot = 1e-5 * u.Unit("cm-3")
        N_tot = blob.V_b * n_tot

        n_e = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_e, p=2.1, gamma_min=1e3, gamma_max=1e6
        )
        n_p = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_p, p=2.1, gamma_min=1e3, gamma_max=1e6
        )

        blob = Blob(n_e=n_e, n_p=n_p)
        assert u.isclose(blob.n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=0.01)
        assert u.isclose(blob.n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=0.01)
        assert np.isclose(blob.N_e_tot, N_tot, atol=0, rtol=0.01)
        assert np.isclose(blob.N_p_tot, N_tot, atol=0, rtol=0.01)

        # intialise from total particle density
        u_tot = 3e-4 * u.Unit("erg cm-3")

        n_e = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_e, p=2.5, gamma_min=1e2, gamma_max=1e6
        )
        n_p = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_p, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        blob = Blob(n_e=n_e, n_p=n_p)
        assert u.isclose(blob.u_e, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=0.01)
        assert u.isclose(blob.u_p, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=0.01)

        # intialise from total energy
        W = 1e48 * u.erg

        n_e = PowerLaw.from_total_energy(
            W=W, V=blob.V_b, mass=m_e, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        n_p = PowerLaw.from_total_energy(
            W=W, V=blob.V_b, mass=m_p, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        blob = Blob(n_e=n_e, n_p=n_p)
        assert u.isclose(blob.W_e, W, atol=0 * u.Unit("erg"), rtol=0.01)
        assert u.isclose(blob.W_p, W, atol=0 * u.Unit("erg"), rtol=0.01)

    def test_total_energies_powers(self):
        """Test the computation of the various energy densities / powers in the
        blob / jet."""
        # energy density of the magnetic field
        blob = Blob(B=2 * u.G)
        U_B_expected = 0.159155 * u.Unit("erg cm-3")
        # 2 pi R**2 Beta Gamma**2 c (prefactor from energy density to power)
        prefactor_expected = 1.87420965e45 * u.Unit("cm3 / s")

        assert np.allclose(
            blob.U_B, U_B_expected, atol=0 * u.Unit("erg cm-3"), rtol=1e-3
        )

        # power in kinetic energy of the particles
        u_tot = 1e-3 * u.Unit("erg cm-3")

        n_e = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_e, p=2.5, gamma_min=1e2, gamma_max=1e6
        )
        n_p = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_p, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        blob.n_e = n_e
        P_jet_ke_expected = prefactor_expected * u_tot

        assert u.isclose(
            blob.P_jet_ke, P_jet_ke_expected, atol=0 * u.Unit("erg s-1"), rtol=0.02
        )

        assert np.isclose(
            blob.k_eq, (u_tot / U_B_expected).to_value(""), atol=0, rtol=0.02
        )

        # now add protons with the same total energy density
        blob.n_p = n_p

        # the jet power in kinetic energy should double
        assert u.isclose(
            blob.P_jet_ke, 2 * P_jet_ke_expected, atol=0 * u.Unit("erg s-1"), rtol=0.02
        )

        # and so the equipartition
        assert np.isclose(
            blob.k_eq, (2 * u_tot / U_B_expected).to_value(""), atol=0, rtol=0.02
        )

    def test_plot_particles_distribution(self):
        """Test the plotting functions for the particles distributions."""
        n_e = BrokenPowerLaw()
        n_p = PowerLaw(mass=m_p)

        # first we initialise the blob without the protons distribution
        blob = Blob(n_e=n_e)
        blob.plot_n_e()
        with pytest.raises(AttributeError):
            assert blob.plot_n_p()

        # second blob with electrons and protons
        blob = Blob(n_e=n_e, n_p=n_p)
        blob.plot_n_e()
        blob.plot_n_p()
