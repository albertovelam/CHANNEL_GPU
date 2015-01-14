#include "channel.h"

void convolution(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz){


	// Derivadas respeto de y
	cudaCheck(cudaMemcpy(wz,ux,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
	deriv_Y_HO_double(wz);
	
	cudaCheck(cudaMemcpy(wx,uz,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
	deriv_Y_HO_double(wx);	

	//Read U_tau
	if(RANK==0){
	readUtau(wz);
	}	
	
	// Calcular omega en fourier
	
	calcOmega(wx,wy,wz,ux,uy,uz);
	
	// Transform to real	
	
	fftBackwardTranspose(ux);	
	fftBackwardTranspose(uy);	
	fftBackwardTranspose(uz);	
	
	
	fftBackwardTranspose(wx);	
	fftBackwardTranspose(wy);	
	fftBackwardTranspose(wz);	
	
	// Calculate rotor
	
	calcRotor(wx,wy,wz,ux,uy,uz);

	// Transform back to Fourier space		
	
	fftForwardTranspose(wx);	
	fftForwardTranspose(wy);	
	fftForwardTranspose(wz);	
	
	//Dealias	

	dealias(wx);
	dealias(wy);
	dealias(wz);
	

	return;

			
}




