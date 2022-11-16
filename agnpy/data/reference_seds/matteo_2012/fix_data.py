import numpy as np
from matplotlib import pyplot as plt

pks_sed = np.loadtxt('PKS2155-304_data_circa.txt')
lognu = pks_sed[:,0]
lognuFnu = pks_sed[:,1]
nu_data = 10**lognu
nuFnu_data = 10**lognuFnu

plt.loglog(nu_data, nuFnu_data, '.')
plt.show()
