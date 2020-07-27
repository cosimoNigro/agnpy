.. _spectra:


Non-thermal Electrons Spectra
=============================

The electrons distributions can be described with simple, broken or curved power laws dependent on the electrons Lorentz factor.
The user is not supposed to interact directly with this classes, the electrons spectra will be defined from the :ref:`emission_regions`.

The available analytical functions describing the electron distributions are:

* :class:`~agnpy.spectra.PowerLaw`
* :class:`~agnpy.spectra.BrokenPowerLaw`
* :class:`~agnpy.spectra.LogParabola`

API
---

.. automodule:: agnpy.spectra
   :noindex:
   :members: PowerLaw, BrokenPowerLaw, LogParabola