////////////////////////////////////////////////////////////////////////////////
// class to compute the Compton emissivities
// References:
// [1] : Dermer, Menon; High Energy Radiation From Black Holes; Princeton Series in Astrophysics
////////////////////////////////////////////////////////////////////////////////
#ifndef Compton_H
#define Compton_H

#ifndef Particles
#include "Particles.h"
#endif

class Compton
{
  // class with the Compton radiative processes implementation
  public:
    // see constructor for parameters definition
    Particles baseElectrons;

    Compton(Particles t_baseElectrons);
    Double_t isotropicPhotonsEmissivityIntegral();
};
#endif
