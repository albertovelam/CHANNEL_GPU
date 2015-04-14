#include "channel.h"



//Completely dealiased convolution with zero padding 2/3ds rule.
//Uses auxiliary buffers AUX of size NYSIZE/NSTEPS_CONV*NX*NZ

void convolutionZeroPadding(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz,
			float2* AUX_1,float2* AUX_2,float2* AUX_3,float2* AUX_4,float2* AUX_5,float2* AUX_6,domain_t domain){



	// Derivadas respeto de y
	CHECK_CUDART(cudaMemcpy(wz,ux,SIZE,cudaMemcpyDeviceToDevice));
	deriv_Y_HO_double(wz,domain);
	
	CHECK_CUDART(cudaMemcpy(wx,uz,SIZE,cudaMemcpyDeviceToDevice));
	deriv_Y_HO_double(wx,domain);	

	//Read U_tau
	if(RANK==0){
	readUtau(wz,domain);
	}	
	
	// Calcular omega en fourier
	
	calcOmega(wx,wy,wz,ux,uy,uz,domain);
	
	// Transform to real	
	
	backwardTranspose(ux,domain);
	backwardTranspose(uy,domain);
	backwardTranspose(uz,domain);
	
	backwardTranspose(wx,domain);
	//backwardTranspose(wy,domain);
	backwardTranspose(wz,domain);

	calcWy(wy,ux,uz,domain); //saves one transpose

	for(int i=0;i<NSTEPS_CONV;i++){

	//AUX_# buffers of size NYSIZE/NSTEPS_CONV*NXP*NZP

	padForward(AUX_1,ux+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);
	padForward(AUX_2,uy+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);
	padForward(AUX_3,uz+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);

	padForward(AUX_4,wx+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);
	padForward(AUX_5,wy+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);
	padForward(AUX_6,wz+i*NYSIZE/NSTEPS_CONV*NX*NZ,domain);

	fftBackwardPadded(AUX_1,domain);
	fftBackwardPadded(AUX_2,domain);
	fftBackwardPadded(AUX_3,domain);

	fftBackwardPadded(AUX_4,domain);
	fftBackwardPadded(AUX_5,domain);
	fftBackwardPadded(AUX_6,domain);

	//INCLUDE ROUTINE TO CALCULATE MAX VELOCITIES FROM AUX_1,AUX_2 AND AUX_3
	// TO EVALUATE CFL CONDITION FOR Dt 

	calcRotorZeroPadding(AUX_4,AUX_5,AUX_6,AUX_1,AUX_2,AUX_3,domain);

	fftForwardPadded(AUX_4,domain);
	fftForwardPadded(AUX_5,domain);
	fftForwardPadded(AUX_6,domain);

	padBackward(wx+i*NYSIZE/NSTEPS_CONV*NX*NZ,AUX_4,domain);
	padBackward(wy+i*NYSIZE/NSTEPS_CONV*NX*NZ,AUX_5,domain);
	padBackward(wz+i*NYSIZE/NSTEPS_CONV*NX*NZ,AUX_6,domain);

		
	}


	forwardTranspose(wx,domain);
	forwardTranspose(wy,domain);
	forwardTranspose(wz,domain);	


	return;

			
}







