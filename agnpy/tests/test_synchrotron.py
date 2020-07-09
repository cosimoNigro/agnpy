# test on synchrotron module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, h
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from io import StringIO
import pytest


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]
# global blob, same parameters as the one defined to produce Figure 7.4 of
# Dermer Menon 2009
SPECTRUM_NORM = 1e48 * u.Unit("erg")
PWL_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_B = 1e16 * u.cm
B = 1 * u.G
Z = Distance(1e27, unit=u.cm).z
DELTA_D = 10
GAMMA = 10
PWL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, PWL_DICT)


# the following points are taken with webplotdigitizer from Figure 7.4
# of Dermer and Menon 2009
sampled_synch_string = StringIO(
    """19281002208.06707, 1.0517919063817168e-12
31081114937.37773, 1.8893646675076006e-12
102544896679.54123, 7.768745191781145e-12
300254668040.4201, 2.1763026869027324e-11
1051545583501.7872, 4.368686466212362e-11
3078958391175.496, 5.2927844856926507e-11
10783063321401.867, 6.035340185482218e-11
29743837423956.23, 6.609603971655472e-11
104168241858427.1, 7.613405252175284e-11
305007873549266.3, 8.422453006484642e-11
1006302410404936.6, 9.507584094653489e-11
2946484963836226.5, 1.0517919063817168e-10
10319117670334754, 1.1753722651306415e-10
30214699618441012, 1.2614707735098417e-10
105817285532851420, 1.187302558090956e-10
309836325057640770, 7.927254606285227e-11
1085102261187576200, 1.1402958502838823e-11"""
)

sampled_synch_table = np.loadtxt(sampled_synch_string, delimiter=",")
sampled_synch_nu = sampled_synch_table[:, 0] * u.Hz
sampled_synch_sed = sampled_synch_table[:, 1] * u.Unit("erg cm-2 s-1")

# there is no reference self-absorbed synchrotron SED in the book, so we
# generate it with agnpy version 0.0.6 and check against it in each new
# implementation, generated with nu grid nu = np.logspace(8, 23)
agnpy_reference_ssa_sed = np.asarray(
    [
        2.04937033e-022,
        1.69848057e-021,
        1.40782007e-020,
        1.16710604e-019,
        9.67832246e-019,
        8.02970580e-018,
        6.66727561e-017,
        5.54351775e-016,
        4.61994675e-015,
        3.86621886e-014,
        3.25747813e-013,
        2.72555211e-012,
        1.64371176e-011,
        3.82843660e-011,
        4.92351546e-011,
        5.37893562e-011,
        5.77799005e-011,
        6.19994383e-011,
        6.65268480e-011,
        7.13841705e-011,
        7.65943671e-011,
        8.21803201e-011,
        8.81621204e-011,
        9.45500389e-011,
        1.01326764e-010,
        1.08403514e-010,
        1.15514784e-010,
        1.21978163e-010,
        1.26195349e-010,
        1.24813561e-010,
        1.12085130e-010,
        8.22428776e-011,
        3.98979768e-011,
        8.39207810e-012,
        3.29969971e-013,
        4.51555572e-016,
        7.43516333e-022,
        1.57237333e-033,
        2.94065972e-057,
        1.49115684e-105,
        1.34947028e-203,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
        0.00000000e000,
    ]
) * u.Unit("erg cm-2 s-1")


class TestSynchrotron:
    """class grouping all tests related to the Synchrotron class"""

    def test_synch_sed(self):
        """test agnpy synchrotron SED against the one from the book"""
        synch = Synchrotron(PWL_BLOB)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_synch_sed = synch.sed_flux(sampled_synch_nu)
        deviation = np.abs(1 - agnpy_synch_sed.value / sampled_synch_sed.value)
        # requires that the SED points deviate less than 20 % from the figure
        assert np.all(deviation < 0.2)

    def test_synch_ssa_sed(self):
        """test this version SSA SED against the one generated in version 0.0.6"""
        ssa = Synchrotron(PWL_BLOB, ssa=True)
        nu = np.logspace(8, 23) * u.Hz
        agnpy_ssa_sed = ssa.sed_flux(nu)
        assert np.allclose(agnpy_ssa_sed.value, agnpy_reference_ssa_sed.value, atol=0)


if __name__ == "__main__":
    # TestSynchrotron.test_synch_ssa_sed()
    nu = np.logspace(8, 23) * u.Hz
    ssa = Synchrotron(PWL_BLOB, ssa=True)
    epsilon = nu.to("", equivalencies=epsilon_equivalency)
    tau = ssa.tau_ssa(epsilon)
    print("tau: ", tau)
    print(np.isclose(tau, 0, atol=0))
    print("done")
