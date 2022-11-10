// script to extract the data from the Mrk421 ROOT file and print the flux point
// vlaues with proper provenance (flux values and instrument names)
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TGraphAsymmErrors>

TString namesList[25] = {"SMA", "VLBA_core(BP143)", "VLBA(BP143)", "VLBA(BK150)", "Metsahovi",
                      "Noto", "VLBA_core(MOJAVE)", "VLBA(MOJAVE)", "OVRO", "RATAN", "Medicina",
                      "Effelsberg", "Swift/UVOT", "ROVOR", "NewMexicoSkies", "MITSuME", "GRT",
                      "GASP", "WIRO", "OAGH", "Swift/BAT", "RXTE/PCA", "Swift/XRT", "Fermi", "MAGIC"};


int extractMrk421SED(){

    TFile *file = TFile::Open("SED_Mrk421_2010_11_24.root");
    TCanvas *canvas = (TCanvas*) file->Get("c");

    // print the flux points to a .txt file
    ofstream outFile("Mrk421_2011.txt");
    outFile << "# nu / Hz, nuFnu / (erg cm-2 s-1), nuFnu_err_low / (erg cm-2 s-1), nuFnu_err_high / (erg cm-2 s-1)\n";

    for (Int_t i = 0; i < 25; ++i){
        TString instrument = namesList[i];
        TGraphAsymmErrors* graph = (TGraphAsymmErrors*) canvas->GetPrimitive(instrument);

        Double_t* xValues = graph->GetX();
        Double_t* yValues = graph->GetY();
        for (Int_t j = 0; j < graph->GetN(); ++j) {
            outFile << xValues[j] << " " << yValues[j] << " " << graph->GetErrorYlow(j) << " " << graph->GetErrorYhigh(j) << " " << instrument << "\n";
        }
    }

    outFile.close();

}