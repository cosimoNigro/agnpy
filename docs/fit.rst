.. _fit:


Use agnpy radiative processes to fit a MWL SED
==============================================
agnpy is a code for numerical modelling: routines for data handling and fitting are not only outside of its scope, but are currently already implemented in several python data-analysis packages.   
Agnpy interfaceability with the astropy ecosystem allows it to be seamlessly interfaced with packages such as `sherpa` and `Gammapy`, containing these routines.
In this documentation we provide several tutorial notebooks illustrating how to build wrappers with these two data-analysis packages to perform a :math:`\chi^2` fit of a MWL SED. 
We consider different science tools and different science cases (radiative processes). The interested user can find:

* `a tutorial notebook wrapping agnpy with sherpa and fitting Mrk421 MWL SED <tutorials/ssc_sherpa_fit.html>`_;

* `a tutorial notebook wrapping agnpy with Gammapy and fitting Mrk421 MWL SED <tutorials/ssc_gammapy_fit.html>`_;

* `a tutorial notebook wrapping agnpy with sherpa and fitting PKS 1510-089 MWL SED <tutorials/ec_dt_sherpa_fit.html>`_;

* `a tutorial notebook wrapping agnpy with Gammapy and fitting PKS 1510-089 MWL SED <tutorials/ec_dt_gammapy_fit.html>`_.