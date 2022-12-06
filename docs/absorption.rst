.. _absorption:


Absorption by :math:`\gamma\gamma` pair production
===========================================================
The photon fields that provide the targets for the Compton scattering might re-asbsorb the scattered photons via 
:math:`\gamma\gamma` pair production. Similarly, on their path towards Earth, the highest-energy photons might be
absorbed by the extragalactic background light (EBL).
``agnpy`` computes the optical depth (or opacity) :math:`\tau_{\gamma \gamma}` as a function of the frequency :math:`\nu` 
produced by the AGN line and thermal emitters and by the EBL.
Photon absorption results in an attenuation of the flux by a factor :math:`\exp(-\tau_{\gamma \gamma})`.

Absorption calculation on line and thermal emitters
---------------------------------------------------
The :math:`\gamma\gamma` absorption for a photon field with specific energy density :math:`u(\epsilon, \mu, \phi; l)` reads:

.. math::
   \tau_{\gamma \gamma}(\nu) = \int_{r}^{\infty} {\rm d}l \; \int_{0}^{2\pi} {\rm d}\phi \; 
                               \int_{-1}^{1} {\rm d}\mu \; (1 - \cos\psi) \int_{0}^{\infty} {\rm d}\epsilon \;
                               \frac{u(\epsilon, \mu, \phi; l)}{\epsilon m_{\rm e} c^2} \, \sigma_{\gamma \gamma}(s),

where: 
    - :math:`\cos\psi = \mu\mu_s + \sqrt{1 - \mu^2}\sqrt{1 - \mu_s^2} \cos\phi` is the cosine of the angle between the hitting and the absorbing photon;
    - :math:`u(\epsilon, \mu, \phi; l)` is the energy density of the target photon field with :math:`\epsilon` dimensionless energy, :math:`(\mu, \phi)` angles, :math:`l` distance of the blob from the photon field;
    - :math:`\sigma_{\gamma \gamma}(s)` is the pair-production cross section, with :math:`s = \epsilon_1 \epsilon \, (1 - \cos\psi)\,/\,2` and :math:`\epsilon_1 = h \nu\,/\,(m_e c^2)` the dimensionless energy of the hitting photon.

We employed the notation [Finke2016]_. The approach presented therein (and in [Dermer2009]_) simplifies the
integration by assuming that the hitting photons travels in the direction parallel to the jet axis (:math:`\mu_s \rightarrow 1`),
decoupling the cross section and the :math:`(1 - \cos\psi)` term from the integral in :math:`\phi`.
The optical depths thus calculated are therefore valid only for blazars. ``agnpy`` instead allows to consider an
arbitrary viewing angle can be used. Optical depths thus obtained are valid for any jetted AGN.

Absorption on target photon fields
----------------------------------
In the following example we compute the optical depth produced by the the broad line region and the dust torus photon fields.

The :class:`~agnpy.absorption.Absorption` requires as input:

- the type of :class:`~agnpy.targets`;
- the distance between the blob and the target photon field (:math:`r`);
- the redshift of the source to correct the observed energies.

.. plot:: snippets/absorption_snippet.py
    :include-source:

For more examples of absorption calculation, especially for cases of "misaligned" sources, i.e. :math:`\mu_s \neq 1`, 
check the `tutorial notebook on Absorption <tutorials/absorption_targets.html>`_.

Extragalactic Background Light
------------------------------
The :math:`\gamma\gamma` absorption produced by the EBL models of [Franceschini2008]_, [Finke2010]_, [Dominguez2011]_ and [Saldana-Lopez2021]_, is available in ``data/ebl_models``.
The absorption values, tabulated as a function of redshift and energy, are interpolated by ``agnpy`` and can be later evaluated for a given redshift and range of frequencies.

.. plot:: snippets/ebl_models.py
    :include-source:

