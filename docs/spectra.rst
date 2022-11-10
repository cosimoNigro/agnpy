.. _spectra:


Non-thermal Electrons Spectra
=============================

The electrons distributions can be described with simple, broken or curved power laws dependent on the electrons Lorentz factor.
For modelling, the user is not supposed to interact directly with these classes, and the electron spectra will be defined from the :ref:`emission_regions`.
Only for the fitting functionalities, the electron distribution has to be specified before being passed to the wrapper of a particular physical scenario, see :ref:`fit`.

The available analytic functions describing the electron distributions are:

* :class:`~agnpy.spectra.PowerLaw`;
* :class:`~agnpy.spectra.BrokenPowerLaw`;
* :class:`~agnpy.spectra.LogParabola`;
* :class:`~agnpy.spectra.ExpCutoffPowerLaw`.


These classes are built on the :class:`~agnpy.spectra.ElectronDistribution`
class, containing methods to numerically integrate and set the normalisation of 
the electron distributions.

API
---

.. automodule:: agnpy.spectra
   :noindex:
   :members: PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, ElectronDistribution