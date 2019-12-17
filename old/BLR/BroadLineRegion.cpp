#include "BroadLineRegion.h"
#include "TMath.h"
#include "TF1.h"
#include "TH1.h"
#include "Math/WrappedTF1.h"
#include "Math/GSLIntegrator.h"
#include "Math/AllIntegrationTypes.h"

////////////////////////////////////////////////////////////////////////////////
// auxiliary functions
////////////////////////////////////////////////////////////////////////////////
Double_t sigmaYY(Double_t x){
	// Eq. 10.1 of Dermer, gamma gamma annihilation cross
	// Let's put a threshold on s otherwise we will have value that tends to infinity
	// x is s, the energy in the center of mass of the system
	if (x >= 1){ // Threshold for gamma gamma pair production
		Double_t beta_cm = TMath::Sqrt(1 - 1 / x);
		Double_t prefactor = 0.5 * TMath::Pi() * TMath::Power(R_E, 2) * (1 - TMath::Power(beta_cm, 2));
		Double_t addend1 = (3 - TMath::Power(beta_cm, 4)) * TMath::Log((1 + beta_cm) / (1 - beta_cm));
		Double_t addend2 = -2 * beta_cm * (2 - TMath::Power(beta_cm, 2));
		return prefactor * (addend1 + addend2);
	}
	else{
		return 0;
	}
}

TH1D* getTH1FromTF1(TF1 *tf1){
		// same binning and extent of TF1
		Int_t NbinsX = tf1->GetNpx();
		Double_t Xmin = tf1->GetXmin();
		Double_t Xmax = tf1->GetXmax();
    TH1D *h1 = new TH1D("h1", "histo", NbinsX, Xmin, Xmax);
    for(Int_t i = 0; i < h1->GetNbinsX(); i++){
        Double_t x = h1->GetBinCenter(i);
        Double_t y = tf1->Eval(x);
        h1->SetBinContent(i, y);
    }
    return h1;
}

////////////////////////////////////////////////////////////////////////////////
// class implementation
////////////////////////////////////////////////////////////////////////////////
BroadLineRegion::BroadLineRegion(Double_t BLR_Rin, Double_t BLR_Rout, Double_t BLR_zeta, Double_t BLR_tauT, Double_t BLR_L0, Double_t BLR_epsilon, Double_t BLR_z){
	// constructor for Broad Line Region class
	Rin = BLR_Rin; // inner radius of BroadLineRegion shell (cm)
	Rout = BLR_Rout; // outer radius of BroadLineRegion shell (cm)
	zeta = BLR_zeta; // spectral index of the shell radial power-law desnity
	tauT = BLR_tauT; // optical dept for BroadLineRegion scattering of central source photons
	L0 = BLR_L0; // luminosity of central source photons (erg s-1)
	epsilonBLR = BLR_epsilon; // monochromatic energy of the target in units of electron mass
	z = BLR_z; // redshift of the source
	// normalization for the radial density, based on tauT
	n0 = (tauT * TMath::Power(Rin, zeta) * (zeta + 1)) / (SIGMA_T * (TMath::Power(Rout, zeta + 1) - TMath::Power(Rin, zeta + 1)));

	// integrand for the geometric factor and wrapper for GSL integration
	tf1IntegrandGeomFact = new TF1("integrandGeomFact", this, &BroadLineRegion::geomFactorIntegrand, -1, 1, 2, "BroadLineRegion", "geomFactorIntegrand");
	wrappedIntegrandGeomFact = new ROOT::Math::WrappedTF1(*tf1IntegrandGeomFact);
	gslIntegratorGeomFact = new ROOT::Math::GSLIntegrator(ROOT::Math::IntegrationOneDim::kADAPTIVE);
	gslIntegratorGeomFact->SetFunction(*wrappedIntegrandGeomFact);
	gslIntegratorGeomFact->SetRelTolerance(1e-4);

	// integrand for the optical depth
	tf1IntegrandTauYYMu = new TF1("integrandTauYYMu", this, &BroadLineRegion::tauYYIntegrandMu, -1, 1, 3, "BroadLineRegion", "tauYYIntegrandMu");
	wrappedIntegrandTauYYMu = new ROOT::Math::WrappedTF1(*tf1IntegrandTauYYMu);
	gslIntegratorTauYYMu = new ROOT::Math::GSLIntegrator(ROOT::Math::IntegrationOneDim::kADAPTIVE);
	gslIntegratorTauYYMu->SetFunction(*wrappedIntegrandTauYYMu);
	gslIntegratorTauYYMu->SetRelTolerance(1e-4);

	// optical depth differential in X
	// 1e10 to 1e50 cm is a sufficient energy range to contain the BroadLineRegion
	tf1dTauYYdLog10X = new TF1("dTauYYdLog10X", this, &BroadLineRegion::dTauYYdLog10X, TMath::Log10(Rin)-2, TMath::Log10(Rout)+2, 2, "BroadLineRegion", "dTauYYdLog10X");
	tf1dTauYYdLog10X->SetNpx(NBINS_TAUYY);
	Double_t log10Rmin = tf1dTauYYdLog10X->GetXmin();
	Double_t log10Rmax = tf1dTauYYdLog10X->GetXmax();
	th1dTauYYdLog10X = new TH1D("dTauYYdLog10X", "th1dTauYYdLog10X", NBINS_TAUYY, log10Rmin, log10Rmax);
}

Double_t BroadLineRegion::densityGasBLR(Double_t *x, Double_t *par){
	// Eq. 90 in ref. [2], radial density of gas in the BroadLineRegion
	Double_t r = x[0];
	if(r >= Rin && r <= Rout){
		return n0 * TMath::Power(r / Rin, zeta);
	}
	else{
		return 0;
	}
}

Double_t BroadLineRegion::geomFactorIntegrand(Double_t *x, Double_t *par){
	// integrand of Eq. 97 in ref. [2]
	Double_t mu = x[0];
	Double_t muS = par[0];
	Double_t r = par[1];
	// Eq. 98 in ref. [2]
	Double_t gBar_num = -mu * (1 - TMath::Power(muS, 2)) + muS * TMath::Sqrt((1 - TMath::Power(mu, 2)) * (1 - TMath::Power(muS, 2)));
	Double_t gBar_denom = TMath::Power(muS, 2) - TMath::Power(mu, 2);
	Double_t gBar = gBar_num / gBar_denom;
	Double_t multiplier = TMath::Sqrt(1 + TMath::Power(gBar, 2) -2 * gBar * mu) / (gBar * (1 - TMath::Power(mu, 2)));
	Double_t gBarr[1] = {gBar*r};
	return densityGasBLR(gBarr, NULL) * multiplier;
}

Double_t BroadLineRegion::geomFactor(Double_t *x, Double_t *par){
	// Eq. 6.171 of ref.[1]
	Double_t muS = x[0];
	Double_t r = x[1];
	tf1IntegrandGeomFact->SetParameters(muS, r);
	// Eq. 97 in ref. [2]
	return gslIntegratorGeomFact->Integral(-muS, 1);
}

Double_t BroadLineRegion::nScattered(Double_t *x, Double_t *par){
	// Eq. 96 if ref.[2]
	Double_t muS = x[0];
	Double_t r = par[0];
	Double_t Ndot = L0 / (MEC2 * epsilonBLR);
	Double_t prefactor = SIGMA_T * Ndot / (8 * TMath::Pi() * C * r);
	// remember in geomFactor muS and r are both variables, here we put r as a parameter to reproduce figure 9 of the paper
	Double_t vars_geomFactor[2] = {muS, r};
	return prefactor * geomFactor(vars_geomFactor, NULL);
}

Double_t BroadLineRegion::tauYYIntegrandMu(Double_t *x, Double_t *par){
	// Cosine dependent Integrand of Eq. 102 in ref. [2], logarithmic in x
	Double_t muS = x[0];
	Double_t r = par[0];
	Double_t epsilon = par[1];
	Double_t epsilon1 = par[2];
	Double_t s = epsilon * epsilon1 * (1 + z) * (1 - muS) / 2;
	Double_t vars_geomFactor[2] = {muS, r};
	return (1 - muS) * geomFactor(vars_geomFactor, NULL) * sigmaYY(s);
}

Double_t BroadLineRegion::dTauYYdLog10X(Double_t *x, Double_t *par){
	// logarithmic in z
	Double_t log10r = x[0];
	Double_t r = TMath::Power(10, log10r);
	Double_t epsilon = par[0];
	Double_t epsilon1 = par[1];
	Double_t muMax = 1 - 2 / (epsilon * epsilon1 * (1 + z));
	Double_t prefactor = SIGMA_T * L0 / (8 * TMath::Pi() * MEC3 * epsilon1);
	tf1IntegrandTauYYMu->SetParameters(r, epsilon, epsilon1);
	Double_t dTauYYdLog10X = prefactor * TMath::Log(10) * gslIntegratorTauYYMu->Integral(-1, muMax);
	Int_t log10rBin = th1dTauYYdLog10X->FindBin(log10r);
	th1dTauYYdLog10X->SetBinContent(log10rBin, dTauYYdLog10X);
	return dTauYYdLog10X;
}

Double_t BroadLineRegion::tauYY(Double_t *x, Double_t *par){
	// invoke right after you call dTauYYdLog10X
	Double_t r = x[0];
	Double_t log10r = TMath::Log10(r);
	Int_t log10rBin = th1dTauYYdLog10X->FindBin(log10r);
	return th1dTauYYdLog10X->Integral(log10rBin, NBINS_TAUYY+1, "width"); // integrate up to the maxmimum bin
}
