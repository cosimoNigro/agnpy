import numpy as np
import astropy.units as u
from agnpy.absorption import EBL
import matplotlib.pyplot as plt

ebl_1 = EBL("dominguez")
ebl_2 = EBL("franceschini")
ebl_3 = EBL("finke")

z = 1.0
nu = np.logspace(15, 30) * u.Hz

absorption_1 = ebl_1.absorption(z, nu)
absorption_2 = ebl_2.absorption(z, nu)
absorption_3 = ebl_3.absorption(z, nu)

# plt.semilogx(nu, absorption_1, ls="-", label="Dominguez")
plt.semilogx(nu, absorption_2, ls="--", label="Franceschini")
# plt.semilogx(nu, absorption_3, ls=":", label="Finke")
plt.legend()
plt.show()
