////////////////////////////////////////////////////////////////////////////////
// test the particle and Compton class
////////////////////////////////////////////////////////////////////////////////
#include "../radiativeProcesses/Particles.cpp"
#include "../radiativeProcesses/Compton.cpp"
#include <TF1.h>
#include <TCanvas.h>

using namespace std;

void testCompton(){

  Particles electrons(1e2, 1e5, 2.2, 1);
  cout << "electrons gammma min and max" << endl;
  cout << electrons.gammaMin << endl;

  Compton compton(electrons);
  cout << "gamma min of imported base electron distribution" << endl;
  cout << compton.baseElectrons.gammaMin << endl;

  reproduceFigure_6_6();

}
