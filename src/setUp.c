#include"channel.h"

void setUp(void){

	//Set up

	fftSetup();
	setDerivatives_HO();
	//setHemholzt();
	//setImplicit();
	setRK3();

	setTransposeCudaMpi();
	
	setDerivativesDouble();
	setHemholztDouble();
	setImplicitDouble();

	cudaCheck(cudaMalloc(&LDIAG,SIZE_AUX),"malloc");
	cudaCheck(cudaMalloc(&CDIAG,SIZE_AUX),"malloc");
	cudaCheck(cudaMalloc(&UDIAG,SIZE_AUX),"malloc");

	//AUX FOR TRANPOSE
	cudaCheck(cudaMalloc(&AUX,SIZE),"malloc");


	return;

}
