// class to compute features of the Broad Line Region (BLR) of Flat Spectrum
// Radio Quasar (FSRQ)
// References:
// [1] : Dermer, Menon; High Energy Radiation From Black Holes; Princeton Series in Astrophysics
// [2] : http://adsabs.harvard.edu/abs/2009ApJ...692...32D
// [3] : http://cdsads.u-strasbg.fr/abs/2016ApJ...821..102B
#ifndef BroadLineRegion_H
#define BroadLineRegion_H

#ifndef ROOT_TF1
#include <TF1.h>
#endif

#ifndef ROOT_TH1
#include <TH1.h>
#endif

#ifndef ROOT_Math_WrappedTF1
#include <Math/WrappedTF1.h>
#endif

#ifndef ROOT_Math_GSLIntegrator
#include <Math/GSLIntegrator.h>
#endif

#ifndef ROOT_Math_AllIntegrationTypes
#include "Math/AllIntegrationTypes.h"
#endif

Double_t SIGMA_T = 6.65245872e-25; // Thomson cross section (cm2)
Double_t MEC2 = 8.18710565e-07; // electron rest mass energy (erg)
Double_t MEC3 = 24544.32526614;
Double_t C = 2.99792458e+10; // speed of light (cm s-1)
Double_t R_E = 2.8179403227e-13; // radius of electron (cm)
Double_t TEV_TO_ERG = 1.60218;
Int_t NBINS_TAUYY = 1000;

class BroadLineRegion
{
	// class describing the Broad Line Region
	// see the constructor for the parameters definition
	public:
		Double_t Rin, Rout, zeta, tauT, L0, epsilonBLR, z, n0;
		TF1 *tf1IntegrandGeomFact;
		ROOT::Math::WrappedTF1 *wrappedIntegrandGeomFact;
		ROOT::Math::GSLIntegrator *gslIntegratorGeomFact;
		TF1 *tf1IntegrandTauYYMu;
		ROOT::Math::WrappedTF1 *wrappedIntegrandTauYYMu;
		ROOT::Math::GSLIntegrator * gslIntegratorTauYYMu;
		TF1 *tf1dTauYYdLog10X;
		TH1D *th1dTauYYdLog10X;

		BroadLineRegion(Double_t BroadLineRegion_Rin, Double_t BroadLineRegion_Rout, Double_t BroadLineRegion_zeta, Double_t BroadLineRegion_tauT, Double_t BroadLineRegion_L0, Double_t BroadLineRegion_epsilon, Double_t BroadLineRegion_z);
		Double_t densityGasBLR(Double_t *x, Double_t *par);
		Double_t geomFactorIntegrand(Double_t *x, Double_t *par);
		Double_t geomFactor(Double_t *x, Double_t *par);
		Double_t nScattered(Double_t *x, Double_t *par);
		Double_t tauYYIntegrandMu(Double_t *x, Double_t *par);
		Double_t dTauYYdLog10X(Double_t *x, Double_t *par);
		Double_t tauYY(Double_t *x, Double_t *par);
};
#endif
