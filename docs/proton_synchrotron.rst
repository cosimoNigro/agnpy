.. _proton_synchrotron:


Hadronic models: Proton Synchrotron Radiation
=============================================

The proton synchrotron radiation is the simplest of the hadronic processes.
`agnpy` implements it following the same calculation adopted for the :ref:`synchrotron radiation from electrons <synchrotron>`.
[Cerruti2015]_ is used as a reference for the validation.

The :class:`~agnpy.synchrotron.ProtonSynchrotron` class works exactly as the :class:`~agnpy.synchrotron.Synchrotron` one.
The blob we pass as argument though needs to have a proton particle distribution defined.
The following example illustrates how to compute the proton synchrotron radiation from a simple power law of protons,
and compares its radiation with that of the electrons.

.. plot:: snippets/p_synchro_snippet.py
   :include-source:

Note that in this case, having not specified an electron distribution, a default one has been initialised by the :class:`~agnpy.emission_regions.Blob`.
