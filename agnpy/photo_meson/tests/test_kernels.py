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

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures, clean-up before making the new
figures_dir = clean_and_make_dir(agnpy_dir, "crosschecks/figures/photo_meson")


class TestKernels:
    """Class to test the phi functions representing the integration kernels"""

    @pytest.mark.parametrize("particle", ["gamma", "positron", "muon_neutrino", "muon_antineutrino"])
    @pytest.mark.parametrize("eta", ["1.5", "30"])
    def test_phi_interpolation(self, particle, eta):
        """Test the interpolations against the plots presented in the [KelnerAharonian2008]_ paper."""
        # read the reference from file
        x_ref, phi_ref = extract_columns_sample_file(
            f"{data_dir}/photo_meson/kelner_aharonian_2008/phi_values/phi_{particle}_eta_{eta}.csv"
        )

        phi = PhiKernel(particle)
        eta = float(eta)
        phi_agnpy = phi(eta, x_ref)

        # comparison plot
        make_comparison_plot(
            x=x_ref,
            y_comp=x_ref * phi_agnpy,
            y_ref=x_ref * phi_ref,
            comp_label="agnpy",
            ref_label="Kelner and Aharonian (2008)",
            fig_title=r"$\phi$" + f" {particle.replace('_', ' ')}",
            fig_path=f"{figures_dir}/phi_comparison_particle_{particle}_eta_{eta}.png",
            plot_type="custom",
            x_label=r"$x = E_{\gamma} / E_{\rm p}$",
            y_label=r"$x \phi(\eta, x)$",
            y_range=None,
            comparison_range=None
        )
        # requires that the SED points deviate less than 25% from the figure
        assert check_deviation(x_ref, phi_agnpy, phi_ref, 0.25)
