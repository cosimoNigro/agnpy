import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
from ..utils.math import axes_reshaper, gamma_p_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e
# to be used in the future to make the code faster:
import numba as nb

''' Photomeson process.
    Reference for all expressions:
    Kelner, S.R., Aharonian, 2008, Phys.Rev.D 78, 034013
    (`arXiv:astro-ph/0803.0688 <https://arxiv.org/abs/0803.0688>`_).

    Product particles and symbol:
        photons: photon
        positrons: positron
        muon antineutrinos: antinu_muon
        electron neutrinos: nu_electron
        muon neutrinos: nu_muon
        electrons: electron
        electron antineutrino: antinu_electron

'''

__all__ = ['PhotoHadronicInteraction']

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')


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

    elif particle in ('positron', 'antinu_muon', 'nu_electron'):
        return x_plus, x_minus / 4

    if particle in ('electron', 'antinu_electron'):
        r = 0.146
        x_1 = 2 * (1 + eta)
        x_2 = eta - (2 * r)
        x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

        x_plus = (x_2 + x_3) / x_1
        x_minus = (x_2 - x_3) / x_1

        return x_plus, x_minus / 2

    if particle == 'nu_muon':
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

    elif particle in ('positron', 'antinu_muon', 'nu_electron', 'nu_muon'):
        psi = 2.5 + 1.4 * np.log(eta / 0.313)

    elif particle in ('electron', 'antinu_electron'):
        psi = 6 * (1 - np.exp(1.5 * (4 - eta/0.313))) * (np.sign(eta/0.131 - 4) + 1) / 2.
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



def H_integrand(gamma, eta, gamma_limit, particle_distribution, soft_photon_dist, particle, *args):

    return (1 / gamma ** 2  *
        particle_distribution(gamma, *args).to('cm-3').value *
        soft_photon_dist((eta /  (4*gamma))).to('cm-3').value *
        phi_gamma(eta, gamma_limit/gamma , particle)
    )


class PhotoHadronicInteraction:

    def __init__(self, particle, blob, soft_photon_distribution, integrator = np.trapz):

        self.blob = blob
        self.soft_photon_distribution = soft_photon_distribution
        self.integrator = integrator

    @staticmethod
    def spectrum(
        particle,
        gammas,
        soft_photon_distribution,
        n_p,
        *args
    ):
        outspecene = gammas
        spectrum_array = np.zeros(len(outspecene))

        for i, g in enumerate(outspecene):

            if particle in ('electron','positron'):
                gamma_limit = g * (mec2/mpc2)
            else:
                gamma_limit = g

            gamma_range = [gamma_limit, n_p.gamma_max]

            if particle in ('electron','nu_muon'):
                eta_range = [0.945, 31.3]
            else:
                eta_range = [0.3443, 31.3]


            dNdE = (1 / 4) * (mpc2.value) *  nquad(H_integrand,
                                        [gamma_range, eta_range],
                                        args=[gamma_limit,
                                        n_p.evaluate,
                                        soft_photon_distribution,
                                        particle,
                                        *args]
                                        )[0]

            spectrum_array[i] = dNdE

            print("Executing {} out of {} steps...\n dNdE={}"
                        .format(i + 1, len(gammas), spectrum_array[i]))

        return (spectrum_array * u.Unit('eV-1 cm-3 s-1')).to('erg-1 cm-3 s-1')


    @staticmethod
    def evaluate_sed_flux(
        nu,
        soft_photon_distribution,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        integrator=np.trapz,
        gamma=gamma_p_to_integrate,
    ):
        #input_array here is either the frequencies in the case of photons or gammas in the case of the other particles

        vol = ((4. / 3) * np.pi * R_b ** 3)
        area = (4 * np.pi * d_L ** 2)

        epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)

        sed_source_frame = (
                PhotoHadronicInteraction.spectrum(
                'photon', epsilon, soft_photon_distribution, n_p, *args
                ) * (vol / area) * (epsilon * mpc2.to('erg')) ** 2
        ).to("erg cm-2 s-1")

        sed = sed_source_frame * np.power(delta_D, 4)

        return sed

    @staticmethod
    def evaluate_sed_flux_particle(
        gammas,
        particle,
        soft_photon_distribution,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        integrator=np.trapz,
        gamma=gamma_p_to_integrate,
    ):
        #input_array here is either the frequencies in the case of photons or gammas in the case of the other particles
        vol = ((4. / 3) * np.pi * R_b ** 3)
        area = (4 * np.pi * d_L ** 2)

        gamma_prime =  gammas / delta_D
        print (gamma_prime)

        sed_source_frame = (
                PhotoHadronicInteraction.spectrum(
                particle, gamma_prime, soft_photon_distribution, n_p, *args
                ) * (vol / area) * (gamma_prime * mec2.to('erg')) ** 2
        ).to("erg cm-2 s-1")

        sed = sed_source_frame * np.power(delta_D, 4)

        return sed

    def sed_flux(self, nu):
        r"""Evaluates the photomeson flux SED for a photomeson object built
        from a Blob."""
        return self.evaluate_sed_flux(
            nu,
            self.soft_photon_distribution,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    def sed_flux_particle(self, gammas, particle):
        r"""Evaluates the photomeson flux SED for a photomeson object built
        from a Blob."""
        return self.evaluate_sed_flux_particle(
            gammas,
            particle,
            self.soft_photon_distribution,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    # JUST FOR PHOTONS:
    def sed_luminosity(self, nu):
        r"""Evaluates the synchrotron luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a a Synchrotron object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")

    # JUST FOR PHOTONS:
    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    # JUST FOR PHOTONS:
    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]


if __name__ == '__main__':

    from agnpy.spectra import ExpCutoffPowerLaw as ECPL

    # SoftPhotonDistribution
    def BlackBody(gamma):
        T = 2.7 *u.K
        kT = (k_B * T).to('eV').value
        c1 = c.to('cm s-1').value
        h1 = h.to('eV s').value
        norm = 8*np.pi/(h1**3*c1**3)
        num = (mpc2.value * gamma) ** 2
        denom = np.exp(mpc2.value * gamma / kT) - 1
        return norm * (num / denom)

    start = timeit.default_timer()

    Ec = 3*1e20 * u.eV

    ''' Tutotial '''
    # The class takes as an input 3 things: the energy array, the proton distr and the soft photon dist
    mpc2 = (m_p * c ** 2).to('eV')
    gammas = np.logspace(1,13,3)
    energies = gammas * mpc2

    A = (0.265*1e11)/(mpc2.value**2) * u.Unit('cm-3')
    p = 2.

    p_dist = ECPL(A, p, Ecut/mpc2, min(gammas), max(gammas))

    proton_gamma = PhotomesonGamma(energies, p_dist, BlackBody)

    sed = proton_gamma.spectrum()


    plt.loglog((energies), (sed * energies ), lw=2.2, ls='-', color='orange',label = 'agnpy')
    plt.show()

    stop = timeit.default_timer()
    print("Elapsed time for computation = {} secs".format(stop - start))
