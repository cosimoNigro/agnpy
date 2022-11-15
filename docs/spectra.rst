.. _spectra:


Non-thermal Particle Energy Distributions
=========================================

The ``agnpy.spectra`` module provides classes describing the energy distributions of particles accelerated in the jets.
The energy distribution is commonly represented by an analytical function, usually a power law, returning the volume
density of particles, :math:`n [{\rm cm}]^{-3}`, as a function of their Lorentz factor, :math:`\gamma`.
The following analytical functions are available:

- :class:`~agnpy.spectra.PowerLaw`;
- :class:`~agnpy.spectra.BrokenPowerLaw`;
- :class:`~agnpy.spectra.LogParabola`;
- :class:`~agnpy.spectra.ExpCutoffPowerLaw`.

Additionaly, an :class:`~agnpy.spectra.InterpolatedDistribution` is available to interpolate an array of densities and
Lorentz factors. This might be useful if you have obtained an electron distribution from another software (e.g. performing
the time evolution of the particles distribution accounting for acceleration and energy losses) and want to use this
result in ``agnpy``.

Since ``v0.3.0``, ``agnpy`` includes both electrons and protons energy distributions.

We can initialise a particle distribution specifying directly the parameters of the analytical function and the mass of
the particle, for example:

.. plot:: snippets/spectra_snippet.py
   :include-source:


Different ways to initialise (or normalise) the particle energy distribution
----------------------------------------------------------------------------
Authors use different approaches to define the particles distribution :math:`n(\gamma)`.
A *normalisation* of the distribution is often provided, which can be of different types.

Some authors use an *integral* normalisation. That is, the normalisation value provided might represent:

- the total volume density, :math:`n_{\rm tot} = \int {\rm d \gamma} \, n(\gamma)`, in :math:`{\rm cm}^{-3}`;
- the total energy density, :math:`u_{\rm tot} = \int {\rm d \gamma} \, \gamma \, n(\gamma)`, in :math:`{\rm erg}\,{\rm cm}^{-3}`;
- the total energy in particles, :math:`W = m c^2 \, \int {\rm d \gamma} \, \gamma \, n(\gamma)`, in :math:`{\rm erg}`.

Others use a *differential* normalisation, that is, the normalisation value provided directly represents the constant,
:math:`k`, multiplying the analytical function, e.g. for a power law

.. math::
    n(\gamma) = k \, \gamma^{-p}.

Finally, some authors use a normalisation *at* :math:`\gamma=1`, that means the normalisation value provided represents
the value of the denisty at :math:`\gamma=1`.

We offer all of the aforementioned alternatives to initialise a particles distribution in ``agnpy``.
Here follows an example illustrating them:

.. plot:: snippets/spectra_init_snippet.py
   :include-source:
