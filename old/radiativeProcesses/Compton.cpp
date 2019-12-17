////////////////////////////////////////////////////////////////////////////////
// class to compute the Compton emissivities
// References:
// [1] : Dermer, Menon; High Energy Radiation From Black Holes; Princeton Series in Astrophysics
////////////////////////////////////////////////////////////////////////////////
#include "Compton.h"
#include "Particles.h"
#include <TMath.h>
#include <TString.h>
#include <TF1.h>
#include <TLegend.h>
#include <TCanvas.h>

using namespace std;


////////////////////////////////////////////////////////////////////////////////
// auxiliary functions
////////////////////////////////////////////////////////////////////////////////
Double_t Fc(Double_t *x, Double_t *par){
  // Compton Kernel in Eq. 6.75 in [1]
  Double_t q = x[0];
  Double_t GammaE = par[0];
  Double_t term1 = 2 * q * TMath::Log(q);
  Double_t term2 = (1 + 2 * q) * (1 - q);
  Double_t term3 = 1. / 2 * (TMath::Power(GammaE * q, 2) / (1 + GammaE * q) * (1 - q));
  return term1 + term2 + term3;
}


void reproduceFigure_6_6(){
  // reprodue Figure 6.6 in [1]
  TCanvas *c1 = new TCanvas("c1", "Figure 6.6 Dermer and Menon", 800., 600.);
  Double_t GammaE[3] = {10., 3., 1.};
  Int_t lineStyles[3] = {9, 2, 7};
  TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
  legend->SetTextSize(0.035);
  for(Int_t i = 0; i < 3; i++){
    TF1 *tf1Fc = new TF1("Compton kernels", Fc, 0, 1 ,1);
    tf1Fc->SetParameter(0, GammaE[i]);
    tf1Fc->SetLineWidth(3);
    tf1Fc->SetLineColor(1);
    tf1Fc->SetLineStyle(lineStyles[i]);
    legend->AddEntry(tf1Fc, Form("#Gamma_{e} = %.0f", GammaE[i]));
    if(i == 0){
      tf1Fc->Draw();
      tf1Fc->GetHistogram()->GetXaxis()->SetTitle("q");
      tf1Fc->GetHistogram()->GetYaxis()->SetTitle("F_{c}(q)");
    }
    else{tf1Fc->Draw("same");}
  }
  legend->Draw("same");
  c1->SaveAs("figures/figure_6.6_Dermer.png");
}
////////////////////////////////////////////////////////////////////////////////
// class implementation
////////////////////////////////////////////////////////////////////////////////
Compton::Compton(Particles t_baseElectrons){
  baseElectrons = t_baseElectrons;
} // end of Compton constructor
