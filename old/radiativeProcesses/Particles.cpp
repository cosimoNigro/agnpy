////////////////////////////////////////////////////////////////////////////////
// class containing particles distributions
// References:
// [1] : Dermer, Menon; High Energy Radiation From Black Holes; Princeton Series in Astrophysics
////////////////////////////////////////////////////////////////////////////////
#include "Particles.h"
#include <TMath.h>
#include <TF1.h>
#include <TH1.h>
#include <TCanvas.h>

using namespace std;


////////////////////////////////////////////////////////////////////////////////
// auxiliary functions
////////////////////////////////////////////////////////////////////////////////
Double_t powerLaw(Double_t *x, Double_t *par){
  Double_t gamma = x[0];
  Double_t ke = par[0];
  Double_t p = par[1];
  return ke * TMath::Power(gamma, -p);
}


////////////////////////////////////////////////////////////////////////////////
// class implementation
////////////////////////////////////////////////////////////////////////////////
Particles::Particles(){
} // default constructor added to import it other classes


Particles::~Particles(){
  cout << "particle object distroyed" << endl;
} // default destructor added to import it other classes


Particles::Particles(Double_t t_gammaMin, Double_t t_gammaMax, Double_t t_p, Double_t t_ue){
  // constructor for the particle class
  gammaMin = t_gammaMin; // minimum Lorentz factor of the particle distribution
  gammaMax = t_gammaMax; // maximum Lorentz factor of the particle distribution
  p = t_p; // spectral index
  ue = t_ue; // total energy density in electrons
  ke = (p - 2) * ue / (MEC2 * (TMath::Power(gammaMin, 2 - p) - TMath::Power(gammaMax, 2 - p)));
  // next member is a TF1 with the particle distribution as a function of the
  // Lorentz factor
  TF1 particleDistribution("particleDistribution", powerLaw, gammaMin, gammaMax, 2);
  particleDistribution.SetParameters(ke, p);
  tf1GammaDistribution = particleDistribution;
} // end of Particles constructor
