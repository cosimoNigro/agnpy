.. _spectral_constraints:


Self-consistent modelling
=========================
``agnpy`` does not foresee time evolution of the electron spectra, it allows though for self-consistent modelling.

In particular, the break and maximum Lorentz factor of the electron distribution (:math:`\gamma_{\rm b}` and 
:math:`\gamma_{\rm max}`) can be constrained accounting for the interplay between acceleration, cooling and escape processes.
See the documentation of the specific function for the particular considerations applied.
Follows a snippet showing the constraint usage. As for the radiative processes it is sufficient to pass
the instance of the emission region to the constraint class:

.. code-block:: python

    from agnpy.emission_regions import Blob
    from agnpy.constraints import SpectralConstraints

    # let us cosider a default blob
    blob = Blob()
    # class defining the spectral constraints
    constraints = SpectralConstraints(blob)

    # max Lorentz factor for a Larmor radius smaller than the blob
    gamma_max_larmor = constraints.gamma_max_larmor
    # max Lorentz factor comparing 1st order Fermi acceleration with synchrotron energy losses
    gamma_max_synch = constraints.gamma_max_synch
    # break Lorentz factor comparing synchrotron cooling time scale with ballistic time scale
    gamma_break_synch = constraints.gamma_break_synch
    
    print(f"gamma_max_larmor = {gamma_max_larmor:.2e}")
    print(f"gamma_max_synch = {gamma_max_synch:.2e}")
    print(f"gamma_break_synch = {gamma_break_synch:.2e}")

should return

.. code-block:: shell

    gamma_max_larmor = 5.87e+12
    gamma_max_synch = 1.17e+08
    gamma_break_synch = 2.32e+03
 
Self-consistent modelling is available also considering cooling due to external Compton on line and thermal emitters.
Check the API for more possibilties.
