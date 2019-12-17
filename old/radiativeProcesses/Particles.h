////////////////////////////////////////////////////////////////////////////////
// class containing particles distributions
// References:
// [1] : Dermer, Menon; High Energy Radiation From Black Holes; Princeton Series in Astrophysics
////////////////////////////////////////////////////////////////////////////////
#ifndef Particles_H
#define Particles_H

#ifndef ROOT_TF1
#include <TF1.h>
#endif

Double_t MEC2 = 8.18710565e-07; // electron rest mass energy (erg)

class Particles
{
  // class containing particles distribution, for now PowerLaw only
  public:
    // see constructor for parameters definition
    Double_t gammaMin, gammaMax, p, ke, ue;
    TF1 tf1GammaDistribution;

    Particles(); // default constructor to import it in other classes
    ~Particles();
    Particles(Double_t t_gammaMin, Double_t t_gammaMax, Double_t t_p, Double_t t_ue);
};
#endif
