#include"channel.h"

void setUp(domain_t domain){

	//Set up

	fftSetup(domain);
	setDerivatives_HO(domain);
	//setHemholzt();
	//setImplicit();
	setRK3(domain);

	setTransposeCudaMpi(domain);
	
	setDerivativesDouble(domain);
	setHemholztDouble(domain);
	setImplicitDouble(domain);

	cudaCheck(cudaMalloc(&LDIAG,SIZE_AUX),domain,"malloc");
	cudaCheck(cudaMalloc(&CDIAG,SIZE_AUX),domain,"malloc");
	cudaCheck(cudaMalloc(&UDIAG,SIZE_AUX),domain,"malloc");

	//AUX FOR TRANPOSE
	cudaCheck(cudaMalloc(&AUX,SIZE),domain,"malloc");


	return;

}
