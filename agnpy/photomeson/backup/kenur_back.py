import astropy.units as u
from naima.radiative import _validate_ene
from naima.models import PowerLaw as NPL
from naima.models import ExponentialCutoffBrokenPowerLaw, Synchrotron
from naima.models import ExponentialCutoffPowerLaw as ECPL
from naima.models import BrokenPowerLaw as BrokenNaima
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
# File with all available soft photon distributions:
# to be used in the future to make the code faster:
import numba as nb

''' Photomeson process.
    Reference for all expressions:
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).
'''

__all__ = ['PhotomesonGamma', 'PhotomesonPositron', 'PhotomesonElectron', 'PhotomesonNeutrinoMu']

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')

# Added a soft photon distribution function like this, necessary for the class to be
# able to take a soft photon distribution as an input.

def soft_photon_distribution(gamma, particle_dist):
     return soft_particle_dist(gamma).to('cm-3').value

def particle_distribution(gamma, particle_dist):
     return particle_dist(gamma).to('cm-3').value

# The file changes according to the type of particle
def lookup_tab1(eta, particle):
    """
    Interpolate the values of s, delta, B for the
    parametric form of _phi_gamma.
    Interpolation of tables I, II, III Kelner2008
    Parameters
    ----------
    eta : float
          (4 * E_soft * E_proton) / mpc2
    interp_file : string
                  interpolation table according
                  to Kelner2008
    Returns
    -------
    s, delta, B : float
                Return these quantities as function of eta
    """
    if particle == 'photon':
        interp_file = "../data/interpolation_tables/gamma_tab1_ka08.txt"

    elif particle == 'positron':
        interp_file = "../data/interpolation_tables/muonantinu_tab2_ka08.txt"

    if particle == 'electron':
        interp_file = "../data/interpolation_tables/elecantinu_tab3_ka08.txt"

    if particle == 'muonnu':
        interp_file = "../data/interpolation_tables/muonnu_tab2_ka08.txt"

    interp_table = open(interp_file, "r")
    rows = interp_table.readlines()
    eta_eta0 = []
    s = []
    delta = []
    B = []

    for row in rows:
        if not row.startswith("#"):
            entries = re.split(r"\s{1,}", row)
            eta_eta0.append(float(entries[0]))
            s.append(float(entries[1]))
            delta.append(float(entries[2]))
            B.append(float(entries[3]))

    eta_arr = np.array(eta_eta0)
    s_arr = np.array(s)
    delta_arr = np.array(delta)
    B_arr = np.array(B)

    s_int = interp1d(eta_arr, s_arr,
                    kind='linear', bounds_error=False, fill_value="extrapolate")
    delta_int = interp1d(eta_arr, delta_arr,
                    kind='linear', bounds_error=False, fill_value="extrapolate")
    B_int = interp1d(eta_arr, B_arr,
                    kind='linear', bounds_error=False, fill_value="extrapolate")

    s_new = s_int(eta)
    delta_new = delta_int(eta)
    B_new = B_int(eta)

    return s_new, delta_new, B_new

# The values change according to the type of particle
def x_plus_minus(eta, particle):

    r = 0.146 # r = m_pi / M_p
    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    if particle == 'photon':
        return x_plus, x_minus

    elif particle == 'positron':
        return x_plus, x_minus / 4

    if particle == 'electron':
        r = 0.146
        x_1 = 2 * (1 + eta)
        x_2 = eta - (2 * r)
        x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

        x_plus = (x_2 + x_3) / x_1
        x_minus = (x_2 - x_3) / x_1

        return x_plus, x_minus / 2

    if particle == 'muonnu':
        rho = eta / 0.313
        if rho < 2.14:
            xp = 0.427 * x_plus
        elif rho > 2.14 and rho < 10:
            xp = (0.427 + 0.0729 * (rho - 2.14)) * x_plus
        elif rho > 10:
            xp = x_plus

        return xp, (xminus * 0.427)

# The power index "ψ" according to the particle type.
def phi_gamma(eta, x, particle): # CHANGE: you put out power (psi) since you wanted to make it as an input for every species

    x_p, x_n = x_plus_minus(eta, particle)

    s, delta, B = lookup_tab1(eta / 0.313, particle) # eta_0 = 0.313

    if particle == 'photon':
        psi = 2.5 + 0.4 * np.log(eta / 0.313)
    elif particle == 'positron' or 'neutrinomu':
        psi = 2.5 + 1.4 * np.log(eta / 0.313)
    elif particle == 'electron':
        6 * (1 - np.exp(1.5 * (4 - eta/0.313))) * (np.sign(eta/0.131 - 4) + 1) / 2.
        # the np.sign part is the heavinside function of (rho - 4) where rho = eta/eta0

    if x > x_n and x < x_p:
        y = (x - x_n) / (x_p - x_n)
        ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
        ln2 = np.log(2. / (1 + y**2))
        return B * ln1 * ln2 ** psi

    elif x < x_n:
        return B * (np.log(2)) ** psi

    elif x > x_p:
        return 0

# The more essential change: Integration now in Ep. I don't use x, except inside the function Φ(η,χ)
def H_integrand(gamma, eta, gamma_limit, particle_dist, soft_photon_dist, particle):

    return (1 / gamma ** 2  *
        particle_distribution(gamma, particle_dist) *
        soft_photon_dist((eta /  (4*gamma))) *
        phi_gamma(eta, gamma_limit/gamma , particle))


class PhotomesonGamma:

    """ Production spectra of secondary photons from
    neutral pion decay produced as secondaries from p-gamma interaction.

    References
    ----------
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).
    """

    # Type of the product particle
    particle = "photon"

    def __init__(self,particle_dist, soft_photon_dist):
        self.particle_dist = particle_dist
        self.soft_photon_dist = soft_photon_dist

    def dNdE(self, gamma):
        gamma_limit = gamma
        gamma_range = [gamma_limit, 1e13]
        eta_range = [0.3443, 31.3]

        dNdE = (1 / 4) * (mpc2.value) *  nquad(H_integrand,
                                    [gamma_range, eta_range],
                                    args=[gamma_limit,
                                    self.particle_dist,
                                    self.soft_photon_dist,
                                    self.particle]
                                    )[0]

        return dNdE

    def spectrum(self,gamma):

        # I should make our own validate energy function
        outspecene = gamma
        spectrum_array = np.zeros(len(outspecene))

        for i, gamma in enumerate(outspecene):
            spectrum_array[i] = self.dNdE(gamma)
            print("Executing {} out of {} steps...\n dNdE={}"
                        .format(i + 1, len(outspecene), spectrum_array[i]))

        return spectrum_array * u.Unit('eV-1 cm-3 s-1')


class PhotomesonElectron:

    """
    Production spectra of secondary electrons from
    charged pion decay produced as secondaries from p-gamma interaction.
    """

    particle = 'electron'

    def __init__(self,particle_dist, soft_photon_dist):

        self.particle_dist = particle_dist
        self.soft_photon_dist = soft_photon_dist

    def dNdE(self, gamma):
        gamma_limit = (gamma * mec2 / mpc2)  #hν / mpc2
        gamma_range = [gamma_limit, 1e14]
        eta_range = [0.945, 31.3]


        dNdE = (1 / 4) * (mpc2.value) *  nquad(H_integrand,
                                    [gamma_range, eta_range],
                                    args=[gamma_limit,
                                    self.particle_dist,
                                    self.soft_photon_dist,
                                    self.particle]
                                    )[0]
        return dNdE

    def spectrum(self, gamma):

        # I should make our own validate energy function
        outspecene = gamma
        spectrum_array = np.zeros(len(outspecene))

        for i, gamma in enumerate(outspecene):
            spectrum_array[i] = self.dNdE(gamma)
            print("Executing {} out of {} steps...\n dNdE={}"
                        .format(i + 1, len(outspecene), spectrum_array[i]))

        return spectrum_array * u.Unit('eV-1 cm-3 s-1')


if __name__ == '__main__':
    start = timeit.default_timer()

    def BlackBody(gamma):
        kT = (k_B * 2.7 *u.K).to('eV').value
        c1 = c.to('cm s-1').value
        h1 = h.to('eV s').value
        norm = 8*np.pi/(h1**3*c1**3)

        num = (mpc2.value * gamma) ** 2
        denom = (np.exp((mpc2.value * gamma / kT)) - 1)
        return (norm * (num / denom))

    def __call__(self, gamma):
        return self.evaluate(gamma,self.kT)
    # Define source parameters
    B = 80 * u.G
    redshift = 0.117
    distPKS = Distance(z=redshift) # Already inside blob definition
    doppler_s = 30
    Gamma_bulk = 16
    R = 5.2e14 * u.cm #radius of the blob
    V = (4. / 3) * np.pi * R ** 3
    A = (4 * np.pi * distPKS ** 2)
    # FOR THE EXAMPLE OF AHARONIAN
    Ec = 3*1e20 * u.eV # characteristic energy of protons
    Ecut = Ec #Aharonian example has different diagrams with cut off energies: Ecut = 0.1Ec, Ecut = Ec, Ecit = 10Ec etc...

    ''' Tutotial '''
    # The class takes as an input 3 things: the energy array, the proton distr and the soft photon dist

    g_dist = BlackBody # eV -1 cm -3

    mpc2 = (m_p * c ** 2).to('eV')

    gammas = np.logspace(1,20,15)
    energies = gammas * mpc2
    energies2 = gammas * mec2

    delta_D = doppler_s

    gammas_prime = gammas / delta_D
    energies_prime = energies / delta_D
    energies2_prime = energies2 / delta_D

    from agnpy.spectra import ExpCutoffPowerLaw as ECPL

    p_dist = ECPL(0.265*1e11/mpc2.value**2 * u.Unit('cm-3'), 2., Ec/mpc2, 1e1, 1e13)

    # proton_gamma = PhotomesonGamma(p_dist, g_dist)
    # sed = proton_gamma.spectrum(gammas_prime)
    # sed_blob = sed * (V/A) * energies_prime ** 2

    proton_gamma = PhotomesonElectron(p_dist, g_dist)
    sed2 = proton_gamma.spectrum(gammas_prime)
    sed_blob2 = sed2 * (V/A) * (gammas_prime * mec2) ** 2

    # plt.loglog((energies), (sed_blob ), lw=2.2, ls='-', color='orange',label = 'agnpy')
    # plt.loglog((energies2), (sed_blob2 ), lw=2.2, ls='-', color='blue',label = 'agnpy')
    # plt.show()
    plt.loglog((energies ), (sed_blob * doppler_s ** 4  ), lw=2.2, ls='-', color='orange',label = 'agnpy')
    # plt.loglog((energies2 ), (sed_blob2 * doppler_s ** 4), lw=2.2, ls='-', color='blue',label = 'agnpy')
    plt.show()
    # plt.loglog((energies), (sed * energies ), lw=2.2, ls='-', color='orange',label = 'agnpy')
    # plt.loglog((energies2), (sed2 * energies2 ), lw=2.2, ls='-', color='blue',label = 'agnpy')
    # plt.show()

    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))


    # But like AGNpy, if you want the value for just one specific point and not the whole thing,
    # you just call the class for the value of energy for which you want the value of the dN/dE
    # Maybe it is not useful so we can consider cutting this feature
    # sed = proton_gamma(energies)
    # print (sed)

    # By the way: I made this BlackBody class which works exactly the same: you just call it for a specific
    # energy and it gives you back the value of the spectrum on that specific point.
