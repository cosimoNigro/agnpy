# test the kernels
import numpy as np
from pathlib import Path
import pytest
from agnpy.photo_meson.kernels import PhiKernel
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
    """Class to test the phi functions representing the integration kernels"""

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
        phi_agnpy = phi(eta, x_ref)

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
