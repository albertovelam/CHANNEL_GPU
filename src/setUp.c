#include"channel.h"

void setUp(void){

	//Set up

	fftSetup();
	setDerivatives_HO();
	setHemholzt();
	setImplicit();
	setRK3();

	setDerivativesDouble();

	cudaCheck(cudaMalloc(&LDIAG,SIZE_AUX),"malloc");
	cudaCheck(cudaMalloc(&CDIAG,SIZE_AUX),"malloc");
	cudaCheck(cudaMalloc(&UDIAG,SIZE_AUX),"malloc");
	cudaCheck(cudaMalloc(&AUX,SIZE_AUX),"malloc");


	return;

}
