"""test Aharonian 2010 expression for R(x)"""
import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from agnpy.synchrotron import R

fig, ax = plt.subplots()

x_tab = np.asarray(
    [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
    ]
)

R_tab = np.asarray(
    [
        0.371,
        0.455,
        0.508,
        0.546,
        0.576,
        0.600,
        0.620,
        0.636,
        0.650,
        0.661,
        0.711,
        0.705,
        0.678,
        0.641,
        0.600,
        0.559,
        0.517,
        0.477,
        0.439,
        0.403,
        0.370,
        0.338,
        0.309,
        0.283,
        0.258,
        0.235,
        0.214,
        0.195,
        0.178,
        0.111,
        0.068,
        0.042,
        0.026,
        0.016,
        0.010,
        0.0059,
        0.0036,
        0.0022,
        0.0013,
        0.00081,
        0.00049,
        0.00030,
        0.00018,
        0.00011,
        0.000068,
    ]
)

x = np.logspace(-4, 3, 100)

ax.loglog(x, R(x), lw=2.5, color="crimson", label="Aharonian 2010")
ax.loglog(x_tab, R_tab, lw=2.5, ls="-.", color="k", label="Crusius 1986")
# ax.grid(which="both")
ax.set_xlabel("x")
ax.set_ylabel("R(x)")
ax.set_xlim([1e-2, 10])
ax.set_ylim([1e-2, 10])
plt.legend()
plt.show()
fig.savefig("results/R_x.pdf")
