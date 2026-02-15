# test the kernels
from pathlib import Path
import pytest

import numpy as np
import astropy.units as u
from astropy.constants import c, h, m_p
from agnpy.photo_meson.kernels import PhiKernel, secondaries, eta_0
from agnpy.utils.math import axes_reshaper, ftiny, fmax, log10
from agnpy.utils.conversion import mpc2

# to be used only for the the validation
# import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.spectra import PowerLaw, ExpCutoffPowerLaw
from agnpy.photo_meson.photo_meson import PhotoMesonProduction
from agnpy.targets.targets import CMB


from agnpy.utils.validation_utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
    clean_and_make_dir,
)

eta_0 = 0.313

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures, clean-up before making the new
figures_dir = clean_and_make_dir(agnpy_dir.parent, "crosschecks/figures/photo_meson")


class TestKernels:
    """Class to test the phi functions representing the integration kernels."""

    @pytest.mark.parametrize(
        "particle", ["gamma", "positron", "muon_neutrino", "muon_antineutrino"]
    )
    @pytest.mark.parametrize("eta_eta0", ["1.5", "30"])
    def test_phi_interpolation(self, particle, eta_eta0):
        """Test the interpolations against the plots presented in the [KelnerAharonian2008]_ paper."""
        # read the reference from file, note the y axis is already phi multiplied by x
        x_ref, x_phi_ref = extract_columns_sample_file(
            f"{data_dir}/photo_meson/kelner_aharonian_2008/phi_values/phi_{particle}_eta_{eta_eta0}_eta0.csv"
        )

        phi = PhiKernel(particle)
        eta = float(eta_eta0) * eta_0
        phi_agnpy = phi(eta, x_ref).to_value("cm3 s-1")

        # comparison plot
        x_max_comparison = 0.1 if eta_eta0 == "1.5" else 0.2
        x_range = [2e-4, x_max_comparison]
        make_comparison_plot(
            x=x_ref,
            y_comp=x_ref * phi_agnpy,
            y_ref=x_phi_ref,
            comp_label="agnpy",
            ref_label="Kelner and Aharonian (2008)",
            fig_title=r"$\phi$" + f" {particle.replace('_', ' ')}",
            fig_path=f"{figures_dir}/phi_comparison_particle_{particle}_eta_{eta_eta0}_eta0.png",
            plot_type="custom",
            x_label=r"$x = E_{\gamma} / E_{\rm p}$",
            y_label=r"$x \phi(\eta, x)$",
            y_range=None,
            comparison_range=x_range,
        )
        # requires that the SED points deviate less than 25% from the figure
        assert check_deviation(
            x_ref, x_ref * phi_agnpy, x_phi_ref, 0.25, x_range=x_range
        )
    @pytest.mark.parametrize(
        "particle", ["gamma", "electron", "positron", "muon_neutrino", "muon_antineutrino", "electron_neutrino", "electron_antineutrino"]
    )
    @pytest.mark.parametrize("fig_number", ["14", "15", "16", "17"])
    def test_spectrum(self, particle, fig_number):
        """Test the interpolations against the plots presented in the [KelnerAharonian2008]_ paper."""
        
        factor = 1.0e3

        if fig_number == "14":
            factor = 1e-1
        if fig_number == "15":
            factor = 1e0
        if fig_number == "16":
            factor = 1e1

        # Blob with proton population
        E_star = 3e20 * u.Unit("eV")
        gamma_star = (E_star / mpc2).to_value("")
        
        n_p = ExpCutoffPowerLaw.from_total_energy_density(
            1.0*u.Unit("erg/cm3"),
            mass = m_p,
            p = 2,
            gamma_c = factor*gamma_star, # change fig_number!
            gamma_min = (1.0*u.Unit("GeV")/mpc2).to_value(""),
            gamma_max = 30.0*factor*gamma_star
            )

        blob = Blob(n_p = n_p)

        cmb = CMB(z = 0.0)
        cmb_target = lambda nu: cmb.du_dnu(nu)

        E_i, spectrum_ref = np.genfromtxt(f"{data_dir}/photo_meson/kelner_aharonian_2008/fig{fig_number}_values/{particle}.txt", 
                          dtype="float", 
                          comments="#", 
                          usecols=(0, 1), 
                          unpack="True")

        E_i = np.power(10,E_i)*u.Unit("eV")
        spectrum_ref = np.power(10,spectrum_ref)#*u.Unit("cm-3 s-1")

        pmp_cmb = PhotoMesonProduction(blob, cmb_target)

        spectrum = ((pmp_cmb.evaluate_spectrum(E_i, particle = particle)*E_i).to_value(f"cm-3 s-1"))

        E_i = E_i.to_value("eV")

        E_range = [E_i.min(), E_i.max()]

        # requires that the SED points deviate less than 50% from the figure
        assert check_deviation(
            E_i, spectrum, spectrum_ref, 0.50, x_range=E_range
        )
