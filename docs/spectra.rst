.. _spectra:


Non-thermal Particles Energy Distributions
==========================================

The ``agnpy.spectra`` module provides classes describing the energy distributions of particles accelerated in the jets.
The energy distribution is commonly represented by an analytical function, usually a power law, returning the volume
density of particles, :math:`n [{\rm cm}]^{-3}`, as a function of their Lorentz factor, :math:`\gamma`.
The following analyitcal functions are available:

- :class:`~agnpy.spectra.PowerLaw`;
- :class:`~agnpy.spectra.BrokenPowerLaw`;
- :class:`~agnpy.spectra.LogParabola`;
- :class:`~agnpy.spectra.ExpCutoffPowerLaw`.

Additionaly, an :class:`~agnpy.spectra.InterpolatedDistribution` is available to interpolate an array of densities and
Lorentz factors (e.g. you might have obtained an electron distribution from a software performing the time evolution of
the particles accounting for acceleration and energy losses).

Since ``v0.3.0``, ``agnpy`` includes both electrons and protons energy distributions.

We can initialise a particle distribution specifying directly the parameters of the analytical function and the mass of
the particle, for example:

.. plot:: snippets/spectra_snippet.py
   :include-source:


Different ways to initialise (or normalise) the particle energy distributions
-----------------------------------------------------------------------------
Authors use different approaches to define the particle distributions :math:`n(\gamma)`.
A *normalisation* of the distribution is often provided, which can be of different types.

Some authors use an *integral* normalisation. That is, the normalisation value provided might represent:

- the total volume density, :math:`n_{\rm tot} = \int {\rm d \gamma} \, n(\gamma)`, in :math:`{\rm cm}^{-3}`;
- the total energy density, :math:`u_{\rm tot} = \int {\rm d \gamma} \, \gamma \, n(\gamma)`, in :math:`{\rm erg}\,{\rm cm}^{-3}`;
- the total energy in particles, :math:`W = m c^2 \, \int {\rm d \gamma} \, \gamma \, n(\gamma)`, in :math:`{\rm erg}`;

Others use a *differential* normalisation, that is, the normalisation value provided directly represents the constant :math:`k`
multiplying the particle distribution, e.g. for a power law

.. math::
    n(\gamma) = k \, \gamma^{-p}

Finally, some authors use a normalisation *at* :math:`\gamma=1`, that means the normalisation value provided represents
the value of the denisty at :math:`\gamma=1`.

We offer all of the aforementioned alternatives to initialise a particle distribution in ``agnpy``, here an example demonstrating it:

.. plot:: snippets/spectra_init_snippet.py
   :include-source:
