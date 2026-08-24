from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import c, m_e

from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.spectra import PowerLaw
from agnpy.time_evolution import (
    BlobLTTIntegrator, BlobExpansion, TimeEvolution, synchrotron_loss, calc_seds_over_time,
)
import agnpy.time_evolution.blob_ltt_integration as blob_ltt_integration
from agnpy.time_evolution.blob_ltt_integration import (
    _constant_kernel_cgs, _expanding_kernel_shape_cgs,
)

C_CGS = c.to_value("cm/s")
SED_UNIT = u.Unit("erg / (cm2 s)")


def make_blob(R_b=1e16 * u.cm, B=0.1 * u.G):
    n_e = PowerLaw(k=1e-8 * u.Unit("cm-3"), p=2.1, gamma_min=1e2, gamma_max=1e6, mass=m_e)
    return Blob(R_b, B=B, n_e=n_e)


def flat_sed(value):
    """A sed_flux_fn with a constant, frequency-independent flux."""
    return lambda blob, nu: np.full(len(nu), value) * SED_UNIT


class TestKernel:
    """The geometric kernel: shape, limits and normalisation."""

    def test_shape_and_normalisation(self):
        R_cm = 1e16
        n = 41
        tau, W = _constant_kernel_cgs(R_cm, n)

        tau_max = R_cm / C_CGS
        assert np.isclose(tau[0], -tau_max, rtol=1e-12)
        assert np.isclose(tau[-1], tau_max, rtol=1e-12)
        assert len(tau) == len(W) == n

        # normalised to 1, peaked at 3c/4R, vanishing at both ends
        assert np.isclose(np.trapz(W, tau), 1.0, rtol=1e-3)
        assert np.isclose(W[n // 2], 0.75 * C_CGS / R_cm, rtol=1e-12)
        assert W[0] == 0
        assert W[-1] == 0

    def test_kernel_is_symmetric(self):
        tau, W = _constant_kernel_cgs(1e16, 51)
        assert np.allclose(W, W[::-1])
        assert np.allclose(tau, -tau[::-1])

    @pytest.mark.parametrize("n", [10, 50, 200, 2000])
    def test_quadrature_error_is_second_order(self, n):
        """
        The kernel integrates to 1 analytically. Composite trapezoid on a parabola has an exact
        error of (b-a) h^2 |f''| / 12, which here works out to 1/(n-1)**2 -- so the shortfall is
        fully predictable, and this pins both the normalisation and the convergence rate.
        """
        tau, W = _constant_kernel_cgs(1e16, n)
        assert np.isclose(np.trapz(W, tau), 1.0 - 1.0 / (n - 1) ** 2, rtol=1e-9)

    def test_expanding_shape_reduces_to_constant_kernel_at_zero_beta(self):
        rho, shape = _expanding_kernel_shape_cgs(0.0, 41)
        tau, W = _constant_kernel_cgs(1e16, 41)
        assert np.allclose(rho, tau * C_CGS / 1e16)
        assert np.allclose(shape, W * 1e16 / C_CGS)


class TestWindow:
    """The blob-state span reported by for_time."""

    def test_window_is_centred_on_the_requested_time_and_reaches_negative_start(self):
        R = 1e16 * u.cm
        t = 5e2 * u.s  # early enough that the window reaches back before 0
        integrator = BlobLTTIntegrator(R)
        window = integrator.for_time(t)

        assert window.start_time < 0 * u.s
        assert u.isclose(window.end_time - t, (R / c).to("s"), rtol=1e-12)
        assert u.isclose(t - window.start_time, (R / c).to("s"), rtol=1e-12)

    def test_window_width_is_independent_of_time(self):
        """With a fixed radius every window spans the same 2R/c, wherever it is centred."""
        integrator = BlobLTTIntegrator(1e16 * u.cm)
        widths = [
            (integrator.for_time(t).end_time - integrator.for_time(t).start_time)
            for t in [0, 3e5, 1e7] * u.s
        ]
        assert u.allclose(u.Quantity(widths), 2 * (1e16 * u.cm / c).to("s"), rtol=1e-12)

    def test_expanding_window_has_asymmetric_limits(self):
        R = 1e16 * u.cm
        integrator = BlobLTTIntegrator(R, expansion=(BlobExpansion(0.3 * c)))
        t0 = 2 * (R / c).to("s")
        window = integrator.for_time(t0)
        assert (window.end_time - t0) > (t0 - window.start_time)


class TestCalcSed:
    """Integration of blob states over the window."""

    def test_constant_state_reproduces_the_plain_sed(self):
        """With a frozen blob the smearing must be a no-op"""
        nu = np.logspace(11, 24, 6) * u.Hz
        R = 1e16 * u.cm
        blob = make_blob(R)
        integrator = BlobLTTIntegrator(R)

        t0 = 2 * (R / c).to("s")
        window = integrator.for_time(t0)
        times = np.linspace(0, 4 * (R / c).to_value("s"), 4) * u.s
        snapshots = [(t, deepcopy(blob)) for t in times]

        sed = window.calc_sed(snapshots, nu)
        expected = Synchrotron(blob).sed_flux(nu) + SynchrotronSelfCompton(blob).sed_flux(nu)

        assert u.allclose(sed, expected, rtol=1e-3)

    def test_state_linear_in_time_integrates_to_the_central_value(self):
        """The kernel is symmetric, so its first moment vanishes."""
        nu = [1e15] * u.Hz
        R = 1e16 * u.cm
        light_crossing = (R / c).to_value("s")
        offset, slope = 3.0, 2e-3

        def sed_flux_fn(blob, freq):
            return np.full(len(freq), offset + slope * blob.marker_time) * SED_UNIT

        integrator = BlobLTTIntegrator(R, sed_flux_fn=sed_flux_fn)
        t0 = 2 * light_crossing * u.s
        window = integrator.for_time(t0)

        times = np.linspace(0, 4 * light_crossing, 40) * u.s
        snapshots = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            snapshots.append((t, blob))

        sed = window.calc_sed(snapshots, nu)
        assert np.isclose(
            sed[0].to_value(SED_UNIT), offset + slope * t0.to_value("s"), rtol=1e-3
        )

    def test_smearing_lags_a_step_change(self):
        """A step in the state is seen smeared over the whole light-crossing window."""
        nu = [1e15] * u.Hz
        R = 1e16 * u.cm
        lc = (R / c).to_value("s")

        def sed_flux_fn(blob, freq):
            return np.full(len(freq), 0.0 if blob.marker_time < 5 * lc else 1.0) * SED_UNIT

        integrator = BlobLTTIntegrator(R, sed_flux_fn=sed_flux_fn)
        times = np.linspace(0, 10 * lc, 400) * u.s
        snapshots = []
        for t in times:
            blob = make_blob(R)
            blob.marker_time = t.to_value("s")
            snapshots.append((t, blob))

        def flux(t0_over_lc):
            return integrator.for_time(t0_over_lc * lc * u.s).calc_sed(
                snapshots, nu
            )[0].to_value(SED_UNIT)

        # fully before the step, halfway through it, fully after.
        # The tolerance on the plateau is set by the kernel quadrature error (~4e-4 at the
        # default 50 points), not by the smearing itself.
        assert np.isclose(flux(3.0), 0.0, atol=1e-6)
        assert np.isclose(flux(5.0), 0.5, rtol=2e-2)
        assert np.isclose(flux(7.0), 1.0, rtol=1e-3)
        # monotonically rising across the transition
        rising = [flux(x) for x in np.linspace(4.0, 6.0, 9)]
        assert np.all(np.diff(rising) >= -1e-12)


class TestRadiusValidation:
    """calc_sed refuses states whose R_b disagrees with the integrator's radius."""

    def _setup(self, snapshot_radius):
        R = 1e16 * u.cm
        integrator = BlobLTTIntegrator(R, sed_flux_fn=flat_sed(1.0))
        lc = (R / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 4 * lc, 5) * u.s
        snapshots = [(t, make_blob(snapshot_radius)) for t in times]
        return window, snapshots

    def test_consistent_radii_pass(self):
        window, snapshots = self._setup(1e16 * u.cm)
        sed = window.calc_sed(snapshots, [1e15] * u.Hz)
        assert np.all(np.isfinite(sed.to_value(SED_UNIT)))

    def test_mismatched_radius_raises(self):
        window, snapshots = self._setup(2e16 * u.cm)
        with pytest.raises(ValueError, match="radius model"):
            window.calc_sed(snapshots, [1e15] * u.Hz)

class TestCoverage:
    """The snapshots must bracket the window: interpolation only, never extrapolation."""

    def _integrator(self):
        return BlobLTTIntegrator(1e16 * u.cm, sed_flux_fn=flat_sed(1.0))

    def test_missing_states_before_the_window_raise(self):
        integrator = self._integrator()
        window = integrator.for_time(0 * u.s)  # start_time < 0
        times = np.linspace(0, 1e6, 10) * u.s
        snapshots = [(t, make_blob()) for t in times]

        with pytest.raises(ValueError, match="missing before the window"):
            window.calc_sed(snapshots, [1e15] * u.Hz)

    def test_missing_states_after_the_window_raise(self):
        integrator = self._integrator()
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 2.5 * lc, 10) * u.s  # ends before end_time
        snapshots = [(t, make_blob()) for t in times]

        with pytest.raises(ValueError, match="missing after the window"):
            window.calc_sed(snapshots, [1e15] * u.Hz)

    def test_two_snapshots_exactly_at_the_window_edges(self):
        """
        The minimal legal input: one snapshot at start_time, one at end_time. This is also the
        case most sensitive to the interaction between the coverage tolerance and the clip that
        pulls the sample times back into the interpolation domain.
        """
        integrator = self._integrator()
        window = integrator.for_time(3e5 * u.s)
        snapshots = [(window.start_time, make_blob()), (window.end_time, make_blob())]

        sed = window.calc_sed(snapshots, [1e15] * u.Hz)
        # flat state, kernel integrates to 1
        assert np.isclose(sed[0].to_value(SED_UNIT), 1.0, rtol=1e-3)

    def test_time_preceding_the_blob_existence(self):
        """A time so far in the past that R(t) <= 0 must raise an error"""
        R = 1e16 * u.cm
        v_exp = 0.1 * c
        integrator = BlobLTTIntegrator(R, expansion=BlobExpansion(v_exp))
        t_negative = -2 * (R / v_exp).to("s")  # R + v_exp * t <= 0 here

        with pytest.raises(ValueError, match="non-positive"):
            integrator.for_time(t_negative)


class TestSedCache:
    @staticmethod
    def _counting_integrator(calls):
        def counting_sed(blob, nu):
            calls.append(id(blob))
            return np.full(len(nu), 1.0) * SED_UNIT

        return BlobLTTIntegrator(1e16 * u.cm, sed_flux_fn=counting_sed)

    def test_each_snapshot_is_evaluated_once(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")
        nu = [1e15] * u.Hz

        times = np.linspace(0, 10 * lc, 9) * u.s
        snapshots = [(t, make_blob()) for t in times]

        # two overlapping windows over the same growing snapshot list
        integrator.for_time(2 * lc * u.s).calc_sed(snapshots[:6], nu)
        assert len(calls) == 6
        integrator.for_time(5 * lc * u.s).calc_sed(snapshots, nu)
        # only the three new snapshots are evaluated
        assert len(calls) == 9

    def test_replaced_snapshot_is_recomputed(self):
        calls = []
        integrator = self._counting_integrator(calls)
        lc = (1e16 * u.cm / c).to_value("s")
        nu = [1e15] * u.Hz
        window = integrator.for_time(2 * lc * u.s)

        times = np.linspace(0, 5 * lc, 6) * u.s
        snapshots = [(t, make_blob()) for t in times]
        window.calc_sed(snapshots, nu)
        assert len(calls) == 6

        # same time, but one state replaced by a different object -> must be re-evaluated
        snapshots[3] = (snapshots[3][0], make_blob())
        window.calc_sed(snapshots, nu)
        assert len(calls) == 7

    def test_calc_sed_uses_the_given_nu_obs(self):
        """calc_sed evaluates the SED at exactly the frequencies it is given."""
        def freq_echo_sed(blob, nu):
            return nu.to_value("Hz") * SED_UNIT

        integrator = BlobLTTIntegrator(1e16 * u.cm, sed_flux_fn=freq_echo_sed)
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 4 * lc, 5) * u.s
        snapshots = [(t, make_blob()) for t in times]

        nu_a = np.array([2e15, 3e15]) * u.Hz
        sed_a = window.calc_sed(snapshots, nu_a)
        assert sed_a.shape == nu_a.shape
        assert u.allclose(sed_a, nu_a.to_value("Hz") * SED_UNIT, rtol=1e-3)

        # a different grid gives a correspondingly different result
        nu_b = np.array([5e15]) * u.Hz
        sed_b = window.calc_sed(snapshots, nu_b)
        assert sed_b.shape == nu_b.shape
        assert u.allclose(sed_b, nu_b.to_value("Hz") * SED_UNIT, rtol=1e-3)

    def test_switching_nu_obs_does_not_contaminate_and_reuses_by_value(self):
        """
        Switching frequency arrays between calls must not use rows cached for a
        different array (in either direction), but two calls that happen to reconstruct an
        equal array from scratch should still hit the cache.
        """
        calls = []

        def counting_sed(blob, nu):
            calls.append(1)
            return np.full(len(nu), 1.0) * SED_UNIT

        integrator = BlobLTTIntegrator(1e16 * u.cm, sed_flux_fn=counting_sed)
        lc = (1e16 * u.cm / c).to_value("s")
        window = integrator.for_time(2 * lc * u.s)
        times = np.linspace(0, 4 * lc, 5) * u.s
        snapshots = [(t, make_blob()) for t in times]

        array_a = [1e15] * u.Hz
        window.calc_sed(snapshots, array_a)
        assert len(calls) == 5

        # switching to a different array must not reuse array_a's cached rows
        array_b = np.logspace(11, 20, 5) * u.Hz
        window.calc_sed(snapshots, array_b)
        assert len(calls) == 10

        # a different object with the SAME values as array_b, called immediately after, hits the cache
        array_b_again = np.logspace(11, 20, 5) * u.Hz
        window.calc_sed(snapshots, array_b_again)
        assert len(calls) == 10

class TestValidation:
    def _window(self):
        integrator = BlobLTTIntegrator(1e16 * u.cm, sed_flux_fn=flat_sed(1.0))
        return integrator.for_time(0 * u.s)

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_snapshots(self, n):
        """Interpolation needs two points; one cannot bracket a window anyway."""
        snapshots = [(t * u.s, make_blob()) for t in range(n)]
        with pytest.raises(ValueError, match="at least two"):
            self._window().calc_sed(snapshots, [1e15] * u.Hz)

    @pytest.mark.parametrize(
        "times",
        [
            pytest.param([0, 2, 1], id="non_monotonic"),
            pytest.param([0, 1, 1], id="duplicate"),
        ],
    )
    def test_non_increasing_times_are_rejected(self, times):
        snapshots = [(t, make_blob()) for t in times * u.s]
        with pytest.raises(ValueError, match="strictly increasing"):
            self._window().calc_sed(snapshots, [1e15] * u.Hz)

    def test_non_scalar_snapshot_time(self):
        """A snapshot time must be a single instant, not an array."""
        snapshots = [(0 * u.s, make_blob()), (np.array([1.0, 2.0]) * u.s, make_blob())]
        with pytest.raises(ValueError, match="must be a scalar Quantity"):
            self._window().calc_sed(snapshots, [1e15] * u.Hz)

    def test_non_positive_radius(self):
        with pytest.raises(ValueError, match="positive finite length"):
            BlobLTTIntegrator(0 * u.cm)

    @pytest.mark.parametrize("R", [np.array([1e16]) * u.cm, np.array([1e16, 2e16]) * u.cm])
    def test_non_scalar_radius(self, R):
        """A shape-(1,) radius used to slip through, silently and with a numpy deprecation."""
        with pytest.raises(ValueError, match="scalar length"):
            BlobLTTIntegrator(R)

    def test_too_few_kernel_points(self):
        with pytest.raises(ValueError, match="at least 2"):
            BlobLTTIntegrator(1e16 * u.cm, kernel_points_size=1)


class TestDocumentedWorkflow:
    """
    The module docstring's usage pattern, driven by a real TimeEvolution using its `t0` offset
    so the callback reports times on the same clock as `for_time`.
    """

    R = 1e15 * u.cm

    def _run(self):
        integrator = BlobLTTIntegrator(self.R, sed_flux_fn=flat_sed(1.0))
        window = integrator.for_time(3 * (self.R / c).to("s"))
        start, end = window.start_time, window.end_time

        blob = make_blob(self.R)
        # advance the blob to the start of the window first
        TimeEvolution(blob, total_duration_time=start,
                      energy_change_functions=synchrotron_loss(Synchrotron(blob)),
                      max_energy_change_per_interval=0.2).evaluate()

        blobs, times = ([deepcopy(blob)], [start])

        def callback(result):
            blobs.append(deepcopy(blob))
            times.append(result.blob_time)

        TimeEvolution(blob, total_duration_time=(end - start), t0=start,
                               energy_change_functions=synchrotron_loss(Synchrotron(blob)),
                               max_energy_change_per_interval=0.2,
                               distribution_change_callback=callback).evaluate()

        times = u.Quantity(times)
        return window, list(zip(times, blobs)), times

    def test_workflow_produces_a_covered_window(self):
        window, snapshots, times = self._run()

        # t0 aligned the clocks: the callback times already bracket the window
        assert len(snapshots) >= 2
        assert times[0] <= window.start_time
        assert times[-1] >= window.end_time - 1e-6 * u.s

        sed = window.calc_sed(snapshots, [1e15] * u.Hz)
        assert np.all(np.isfinite(sed.to_value(SED_UNIT)))
        assert np.isclose(sed[0].to_value(SED_UNIT), 1.0, rtol=1e-3)


class _FakeTimeEvolution:
    """
    A stub for TimeEvolution in call-sequence tests: runs no real physics, just fires the
    distribution_change_callback once, at t0 + total_duration_time, so calc_seds_over_time's
    logic (snapshot bracketing, `now` advancement) does not fail.
    """

    def __init__(self, blob, total_duration_time, t0=0 * u.s,
                 distribution_change_callback=None, **kwargs):
        self._callback = distribution_change_callback
        self._end = t0 + total_duration_time

    def evaluate(self):
        if self._callback is not None:
            self._callback(SimpleNamespace(blob_time=self._end))


class TestCalcSedsOverTime:

    R = 1e16 * u.cm
    nu = np.logspace(11, 24, 6) * u.Hz

    def _lc(self):
        """ Radius light-crossing time """
        return (self.R / c).to_value("s")

    def test_matches_the_manual_workflow(self):
        """The automated loop must correctly proceed to start/end of window for each time point of interest:
         0, 3, 4, 6 and 9 (in units or radius light-crossing), according to this timeline:

              *           *   *       *           *
         -1   0   1   2   3   4   5   6   7   8   9   10   (time, in R/c)
              ---->                                        (run simulation till end of window for time=0; steady state is assume for time < 0)
                   ===>                                    (fast forward to start of window for time=3)
                       ------->                            (run simulation till end of window for time=3)
                               --->                        (run simulation till end of window for time=4, which is also start of window for time=6)
                                   ------->                (run simulation till end of window for time=6)
                                           ===>            (fast forward to start of window for time=9)
                                               ------->    (run simulation till end of window for time=9)

        This is checked by mocking TimeEvolution and asserting the sequence of
        (total_duration_time, t0, distribution_change_callback) it is constructed with.
        "---->" segments run to the end of a window and must have a callback
        attached, to gather the snapshots that window's SED needs. "===>" segments are
        fast-forwards through an already-covered gap with no window boundary inside it; their
        `evaluate()` result is never read back, so their exact t0 and whether a callback is
        attached are left unconstrained here (marked None below) -- only their duration matters.
        """
        lc = self._lc()
        times = np.array([0, 3 * lc, 4 * lc, 6 * lc, 9 * lc]) * u.s

        # (total_duration_time, t0, has_callback) for every segment in the diagram above, in units of lc
        expected_calls = [
            (1, 0, True),
            (1, None, None),  # fast-forward: t0/callback unconstrained
            (2, 2, True),
            (1, 4, True),
            (2, 5, True),
            (1, None, None),  # fast-forward
            (2, 8, True),
        ]

        blob = make_blob(self.R)
        with patch.object(
            blob_ltt_integration, "TimeEvolution", side_effect=_FakeTimeEvolution
        ) as mock_time_evolution:
            calc_seds_over_time(
                blob, times, self.nu,
                energy_change_functions=synchrotron_loss(Synchrotron(blob)),
            )

        assert mock_time_evolution.call_count == len(expected_calls)
        for call, (expected_duration, expected_t0, expects_callback) in zip(
            mock_time_evolution.call_args_list, expected_calls
        ):
            _, kwargs = call
            assert u.isclose(kwargs["total_duration_time"], expected_duration * lc * u.s)
            if expected_t0 is not None:
                assert u.isclose(kwargs.get("t0", 0 * u.s), expected_t0 * lc * u.s)
            if expects_callback is not None:
                callback = kwargs.get("distribution_change_callback")
                assert (callback is not None) == expects_callback


    def test_overlapping_windows_reuse_snapshots_instead_of_resimulating(self):
        """Heavily overlapping windows must not be independently re-simulated from scratch."""
        lc = self._lc()
        calls = []

        def counting_sed(blob, nu):
            calls.append(1)
            return np.full(len(nu), 1.0) * SED_UNIT

        blob = make_blob(self.R)
        tight_times = np.array([5 * lc, 5.1 * lc, 5.2 * lc, 5.3 * lc]) * u.s

        calc_seds_over_time(
            blob, tight_times, self.nu, sed_flux_fn=counting_sed,
            energy_change_functions=synchrotron_loss(Synchrotron(blob)),
            max_energy_change_per_interval=0.3,
        )

        # if each window re-simulated independently it would need dozens of evaluations per
        # window; heavy reuse keeps the total far below that
        assert len(calls) < 10 * len(tight_times)
