#include <iostream>
#include "TMath.h"
#include "TF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TH1.h"
#include "TH2.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TString.h"
#include "Math/WrappedTF1.h"
#include "Math/GSLIntegrator.h"
#include "Math/AllIntegrationTypes.h"
// to work with root compiler include the .cpp
#include "BroadLineRegion.cpp"

using namespace std;

Double_t radiusSchw(Double_t massBH){
	// return a Schwarzschild radius in cm given the BH mass in solar mass unit
	Double_t G = 6.67408*1e-8; // (cm3 g-1 s-2)
	Double_t massSun = 1.9884754*1e33; // (g)
	return 2 * G * massBH * massSun / TMath::Power(C, 2);
}

void tauBLR(Double_t energy, Double_t Rin, Double_t Rout, Double_t zeta, Double_t tauT, Double_t L0, Double_t epsilonBLR, Double_t z){
	// macro to compute the BLR absorption
	// energy in TeV!
	// tauBLR(energy, Rin, Rout, zeta, tauT, epsilonBLR, L0, z);
	// reproduce results in ref. 2
	// tauBLR(0.112, 2.95325e+15, 2.95325e+18, -2, 0.1, 2e39, 2e-5, 0.36)
	// PKS1510 ref 3. results
	// tauBLR(0.112, 6.9e17, 8.4e17, 0, 0.1, 1e46, 2e-5, 0.36)

	BroadLineRegion *BLR = new BroadLineRegion(Rin, Rout, zeta, tauT, L0, epsilonBLR, z);
	cout << "Rin: " << BLR->Rin << " cm" << endl;
	cout << "Rout: " << BLR->Rout << " cm" << endl;
	cout << "normalization of gas density: " << BLR->n0 << " cm-3" << endl;

	cout << "displaying radial profile of the gas in the BLR... " << endl;
	TCanvas *c1 = new TCanvas("c1", "BLR radial density");
	TF1 *radialDensity = new TF1("radialDensity", BLR, &BroadLineRegion::densityGasBLR, 1e-1*Rin, 1e1*Rout, 0, "BroadLineRegion", "densityGasBLR");
	radialDensity->Draw();
	radialDensity->GetHistogram()->GetYaxis()->SetTitle("n_{e} / cm^{-3}");
	radialDensity->GetHistogram()->GetXaxis()->SetTitle("R / cm");
	c1->SetLogx();

	// useful only for ref [2]
	Double_t rSchw =  radiusSchw(1e8);
	TCanvas *c2 = new TCanvas("c2", "Figure 9 in ref. [2]");
	cout << "let's try to reproduce Figure 9 in ref. [2]" << endl;
	TLegend *legend = new TLegend(0.1, 0.1, 0.2, 0.3);
	legend->SetTextSize(0.03);
	Double_t radiuses[4] = {10*rSchw, 1e2*rSchw, 1e4*rSchw, 1e5*rSchw};
	Double_t colorsc2[4] = {9, 1, 8, 46};

	for(Int_t i=0; i<4; i++){
		TF1 *nSc1 = new TF1("nSc1", BLR, &BroadLineRegion::nScattered, -1, 1, 1, "BroadLineRegion", "nScattered");
		nSc1->SetParameter(0, radiuses[i]);
		nSc1->SetLineColor(colorsc2[i]);
		nSc1->SetLineWidth(2);
		nSc1->SetLineStyle(9);
		nSc1->SetTitle("");
		legend->AddEntry(nSc1, TString(Form("%.0e r_{s}", radiuses[i] / rSchw)));
		nSc1->GetHistogram()->GetXaxis()->SetTitle("#mu_{*}");
		nSc1->GetHistogram()->GetYaxis()->SetTitle("n_{sc} / cm^{-3}");
		nSc1->GetHistogram()->GetYaxis()->SetRangeUser(1e-6, 1e8);
		if (i==0){
			nSc1->Draw("");
		}
		else{
			nSc1->Draw("same");
		}
	}
	legend->Draw("same");
	c2->SetLogy();
	c2->SaveAs("results/ref2_nScattered.png");

	TLegend *legend2 = new TLegend(0.15, 0.15, 0.65, 0.25);
	legend2->SetTextSize(0.03);
	TCanvas *c3 = new TCanvas("c3", "optical depths for agn in ref. [2]", 1000, 500);
	c3->Divide(2,1);
	// note here distances are Log10s
	TF1 *dTauYYdLogXtf1 = new TF1("dTauYYdLogXtf1", BLR, &BroadLineRegion::dTauYYdLog10X, TMath::Log10(Rin)-1, TMath::Log10(Rout)+1, 2, "BroadLineRegion", "dTauYYdLog10X");
	TF1 *tauYYtf1 = new TF1("tauYYtf1", BLR, &BroadLineRegion::tauYY, 1e-1*Rin, 1e+1*Rout, 0, "BroadLineRegion", "tauYY");
	// let's do it for several epsilons
	Double_t epsilon = energy * TEV_TO_ERG / MEC2;
	dTauYYdLogXtf1->SetParameters(epsilon, BLR->epsilonBLR);
	dTauYYdLogXtf1->SetLineColor(1);
	dTauYYdLogXtf1->SetLineWidth(2);
	dTauYYdLogXtf1->GetHistogram()->GetYaxis()->SetTitle("d #tau_{#gamma #gamma}/d log_{10}(x)");
	dTauYYdLogXtf1->GetHistogram()->GetXaxis()->SetTitle("log10(R / cm)");
	dTauYYdLogXtf1->SetTitle("");

	tauYYtf1->SetLineColor(1);
	tauYYtf1->SetLineWidth(2);
	tauYYtf1->GetHistogram()->GetYaxis()->SetTitle("#tau_{#gamma #gamma}");
	tauYYtf1->GetHistogram()->GetXaxis()->SetTitle("R / cm");
	tauYYtf1->SetTitle("");
	// draw the differential optical depths
	// draw the lines delimiting the region
	TLine *RinLineLog = new TLine(TMath::Log10(BLR->Rin), dTauYYdLogXtf1->GetMinimum(), TMath::Log10(BLR->Rin), dTauYYdLogXtf1->GetMaximum());
	TLine *RoutLineLog = new TLine(TMath::Log10(BLR->Rout), dTauYYdLogXtf1->GetMinimum(), TMath::Log10(BLR->Rout), dTauYYdLogXtf1->GetMaximum());
	RinLineLog->SetLineColor(22);
	RinLineLog->SetLineStyle(2);
	RinLineLog->SetLineWidth(2);
	RoutLineLog->SetLineColor(22);
	RoutLineLog->SetLineWidth(2);
	RoutLineLog->SetLineStyle(2);
	TLine *RinLine = new TLine(BLR->Rin, tauYYtf1->GetMinimum(), BLR->Rin, tauYYtf1->GetMaximum());
	TLine *RoutLine = new TLine(BLR->Rout, tauYYtf1->GetMinimum(), BLR->Rout, tauYYtf1->GetMaximum());
	RinLine->SetLineColor(22);
	RinLine->SetLineStyle(2);
	RinLine->SetLineWidth(2);
	RoutLine->SetLineColor(22);
	RoutLine->SetLineWidth(2);
	RoutLine->SetLineStyle(2);

	c3->cd(1);
	dTauYYdLogXtf1->Draw("");
	RinLineLog->Draw("same");
	RoutLineLog->Draw("same");
	legend2->AddEntry(dTauYYdLogXtf1, TString(Form("E = %.2f TeV, #epsilon = %.3e", energy, epsilon)));
	legend2->Draw("same");
	gPad->SetLeftMargin(0.15);
	gPad->SetLogy();

	// draw the optical depths
	c3->cd(2);
	tauYYtf1->Draw("");
	RinLine->Draw("same");
	RoutLine->Draw("same");
	gPad->SetLeftMargin(0.15);
	gPad->SetLogy();
	gPad->SetLogx();
	c3->SaveAs(TString(Form("results/ref2_dtauYYdLog10X_E_%.2f.png", energy)));
}
