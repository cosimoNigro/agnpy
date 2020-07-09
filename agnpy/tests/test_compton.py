# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from io import StringIO
import pytest

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
# global PWL blob
SPECTRUM_NORM = 1e48 * u.Unit("erg")
PWL_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7},
}
R_B = 1e16 * u.cm
B = 1 * u.G
Z = Distance(1e27, unit=u.cm).z
DELTA_D = 10
GAMMA = 10
PWL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, PWL_DICT)

# the following points are taken from Figure 7.4 of Dermer and Menon 2009
sampled_ssc_string = StringIO(
    """2320651998432363, 1.0034000379911887e-12
10211190588495366, 3.3783858712488843e-12
29674768813983508, 8.064205774624566e-12
103382613573372500, 1.4798922827809465e-11
301597912834841700, 2.2636945912731082e-11
991837572512144800, 3.2919683927534396e-11
3074932384991272000, 4.415180530485312e-11
10119321782947789000, 5.921711804500588e-11
29564967431579832000, 7.627190143409357e-11
103315591794494950000, 9.824260194294287e-11
302008784167628000000, 1.1908659984844048e-10
1.0555624696559661e+21, 1.5031884246909748e-10
3.0858555962550093e+21, 1.803782147857804e-10
1.0786434641030331e+22, 2.2539393928780104e-10
2.9724490111816544e+22, 2.519766274458472e-10
1.0401838319940773e+23, 2.76068908136978e-10
3.044885941941356e+23, 2.84644300864069e-10
1.066091358668202e+24, 2.934982154792073e-10
2.940427221453963e+24, 2.9655121817328477e-10
1.0298776524388784e+25, 2.936521423067126e-10
3.016561780979584e+25, 2.8207949954627346e-10
1.0575596921309445e+26, 2.4991298987697746e-10
3.103062394121982e+26, 1.961009177107105e-10
1.0286231918640271e+27, 1.1360856488568951e-10
3.03667650907864e+27, 4.391727921550145e-11
6.296557127257376e+27, 9.732917651399568e-12
9.751019298876886e+27, 1.010165141117916e-12"""
)

sampled_ssc_table = np.loadtxt(sampled_ssc_string, delimiter=",")

sampled_ssc_nu = sampled_ssc_table[:, 0]
sampled_ssc_sed = sampled_ssc_table[:, 1]


class TestSynchrotronSelfCompton:
    """class grouping all tests related to the Synchrotron Slef Compton class"""

    def test_ssc_sed(self):
        """test agnpy synchrotron SED against the one in Figure 7.4 of Dermer Menon"""
        synch = Synchrotron(PWL_BLOB)
        ssc = SynchrotronSelfCompton(PWL_BLOB, synch)
        # compute the SED as the same frequency at whcih the SED was sampled
        agnpy_ssc_sed = ssc.sed_flux(sampled_ssc_nu * u.Hz).value
        assert np.allclose(agnpy_ssc_sed, sampled_ssc_sed)
