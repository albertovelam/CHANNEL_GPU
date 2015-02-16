#include "channel.h"

void convolution(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz, domain_t domain){


	// Derivadas respeto de y
  cudaCheck(cudaMemcpy(wz,ux,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");
  deriv_Y_HO_double(wz,domain);
  
  cudaCheck(cudaMemcpy(wx,uz,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");
  deriv_Y_HO_double(wx,domain);	

  //Read U_tau
  if(domain.rank==0){
    readUtau(wz,domain);
  }	
  
  // Calcular omega en fourier
  
  calcOmega(wx,wy,wz,ux,uy,uz,domain);
  
  // Transform to real	
  
  fftBackwardTranspose(ux, domain);	
  fftBackwardTranspose(uy, domain);	
  fftBackwardTranspose(uz, domain);	
  
  
  fftBackwardTranspose(wx, domain);	
  fftBackwardTranspose(wy, domain);	
  fftBackwardTranspose(wz, domain);	
  
  // Calculate rotor
  
  calcRotor(wx,wy,wz,ux,uy,uz,domain);
  
  // Transform back to Fourier space		
  
  fftForwardTranspose(wx, domain);	
  fftForwardTranspose(wy, domain);	
  fftForwardTranspose(wz, domain);	
  
  //Dealias	
  
  dealias(wx,domain);
  dealias(wy,domain);
  dealias(wz,domain);
  
  
  return;
  
			
}




