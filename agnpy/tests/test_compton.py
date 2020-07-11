# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.targets import RingDustTorus
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from io import StringIO
import matplotlib.pyplot as plt
import pytest

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
# global PWL blob
PWL_SPECTRUM_NORM = 1e48 * u.Unit("erg")
PWL_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_B_PWL = 1e16 * u.cm
B_PWL = 1 * u.G
Z_PWL = Distance(1e27, unit=u.cm).z
DELTA_D_PWL = 10
GAMMA_PWL = 10
PWL_BLOB = Blob(
    R_B_PWL, Z_PWL, DELTA_D_PWL, GAMMA_PWL, B_PWL, PWL_SPECTRUM_NORM, PWL_DICT
)

# global BPL blob
BPL_SPECTRUM_NORM = 6e42 * u.Unit("erg")
BPL_DICT = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.0,
        "p2": 3.5,
        "gamma_b": 1e4,
        "gamma_min": 20,
        "gamma_max": 5e7,
    },
}
R_B_BPL = 1e16 * u.cm
B_BPL = 0.56 * u.G
Z_BPL = 1
DELTA_D_BPL = 40
GAMMA_BPL = 40
BPL_BLOB = Blob(
    R_B_BPL, Z_BPL, DELTA_D_BPL, GAMMA_BPL, B_BPL, BPL_SPECTRUM_NORM, BPL_DICT
)
BPL_BLOB.set_gamma_size(500)

# the following points are taken from Figure 7.4 of Dermer and Menon 2009
sampled_ssc_string = StringIO(
    """2614944543238824, 1.0517919063817168e-12
10319117670334754, 3.605932093335428e-12
30214699618441012, 7.927254606285227e-12
105817285532851420, 1.5129499502366456e-11
309836325057640770, 2.266016969803532e-11
1085102261187576200, 3.3260580486968143e-11
2993129529441971700, 4.368686466212362e-11
10482475286977960000, 5.974695679970744e-11
30693015834511335000, 7.613405252175284e-11
101264454118678560000, 9.701571868867811e-11
296505492133032170000, 1.1753722651306415e-10
1.0384152984910122e+21, 1.482697760456359e-10
3.040512505569831e+21, 1.7604108438655598e-10
1.0648418949421582e+22, 1.90854214400669e-10
2.9372436441613516e+22, 1.9474830399087612e-10
1.091941020968905e+23, 1.889364667507593e-10
3.0119934601316744e+23, 1.7604108438655598e-10
9.937370612056698e+23, 1.5438193898779022e-10
3.0886455816865645e+24, 1.1635618505359118e-10
1.0190266426796636e+25, 7.311981563944311e-11
2.9837419143177227e+25, 2.8013567611988813e-11
5.1056273492242735e+25, 1.0841458689358372e-11"""
)

# the following points are taken from Figure 11 of Finke 2016
sampled_ec_dt_string = StringIO(
    """11343232716353988, 2.439948851000885e-26
30126409040604520, 2.0956123869705675e-25
106246783089404300, 3.280052945898815e-24
300534778602682300, 3.113597393093638e-23
1059895767530234900, 4.873403290158684e-22
2998072323456430600, 4.400592102463717e-21
10245338593872138000, 6.232359417932816e-20
14489474389457713000, 1.3872323120265444e-19
29908144736082194000, 2.177710653534621e-19
102205362805401240000, 4.392693153993351e-19
307908056601890500000, 8.016125396170055e-19
1.0195794009610904e+21, 1.538046658768363e-18
3.071626608272833e+21, 2.6699326577462e-18
1.0832703145633775e+22, 5.3858449193112536e-18
3.0641906953569677e+22, 9.348435821541323e-18
1.04712854805089e+23, 1.793770198106569e-17
3.056772783587732e+23, 3.113685585645011e-17
1.078031817378934e+24, 5.974828648410431e-17
2.5240924812633494e+24, 7.681684439003113e-17
1.1098471161392183e+25, 3.825519754908847e-17
3.041990789283074e+25, 1.903626575293829e-17
1.0072978215950157e+26, 6.6786744321175614e-18
1.0370255908667412e+27, 3.1805899933554136e-19
3.0272802780002223e+27, 4.7706392282394245e-20
1.0024267134963427e+28, 3.9291877858624616e-21
3.01995172040197e+28, 3.235641972506863e-22"""
)

sampled_ssc_table = np.loadtxt(sampled_ssc_string, delimiter=",")
sampled_ec_dt_table = np.loadtxt(sampled_ec_dt_string, delimiter=",")

sampled_ssc_nu = sampled_ssc_table[:, 0] * u.Hz
sampled_ssc_sed = sampled_ssc_table[:, 1] * u.Unit("erg cm-2 s-1")
sampled_ec_dt_nu = sampled_ec_dt_table[:, 0] * u.Hz
# there is a factor 2 missing in Finke 2016 expression of the energy density
# and consequently of the SED of DT
sampled_ec_dt_sed = 2 * sampled_ec_dt_table[:, 1] * u.Unit("erg cm-2 s-1")

# dust torus definition
L_disk = 2 * 1e46 * u.Unit("erg s-1")
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)


class TestSynchrotronSelfCompton:
    """class grouping all tests related to the Synchrotron Slef Compton class"""

    def test_ssc_reference_sed(self):
        """test agnpy SSC SED against the one in Figure 7.4 of Dermer Menon"""
        synch = Synchrotron(PWL_BLOB)
        ssc = SynchrotronSelfCompton(PWL_BLOB, synch)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_ssc_sed = ssc.sed_flux(sampled_ssc_nu)
        deviation = np.abs(1 - agnpy_ssc_sed.value / sampled_ssc_sed.value)
        # requires that the SED points deviate less than 20 % from the figure
        assert np.all(deviation < 0.15)


class TestExternalCompton:
    """class grouping all tests related to the External Compton class"""

    def test_ec_dt_reference_sed(self):
        ec_dt = ExternalCompton(BPL_BLOB, dt, r=1e19 * u.cm)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_ec_dt_sed = ec_dt.sed_flux(sampled_ec_dt_nu)
        # as the sampled SED is computed with an approximation by Finke
        # check that the integrals of the two SED have 60% deviation
        reference_sed_integral = np.trapz(sampled_ec_dt_sed, sampled_ec_dt_nu).to(
            "erg cm-2 s-2"
        )
        agnpy_sed_integral = np.trapz(sampled_ec_dt_sed, sampled_ec_dt_nu).to(
            "erg cm-2 s-2"
        )
        deviation = np.abs(1 - agnpy_sed_integral.value / reference_sed_integral.value)
        assert deviation < 0.6
