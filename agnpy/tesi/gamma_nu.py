from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
import astropy.units as u
import numpy as np

mpc2 = (m_p * c ** 2).to('eV')
h1 = h.to('eV s').value

gamma = float(input('gamma: '))
nu = gamma * (mpc2/h1)
print (nu)
