#include "channel.h"



//Completely dealiased convolution with four shifted grids
//Uses extra buffers of size NXSIZE*NZ*NY

void convolutionPhaseShift(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz,float2* Ax,float2* Ay,
		float2* Az,float2* Bx,float2* Bz,float2* Cx,float2* Cy,float2* Cz,domain_t domain){

	//Shifts
	
	const float Delta1[3]={0.5f,0.0f,0.5f};
	const float Delta3[3]={0.0f,0.5f,0.5f};
	


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
	backwardTranspose(wz,domain);



	//Calc convolution on four shifted grids

	set2zero(Cx,domain);
	set2zero(Cy,domain);
	set2zero(Cz,domain);

	//Calc three shifted grids

	for(int i=0;i<3;i++){	

	calcWy(wy,ux,uz,domain);

	//Forward phase shift

    	phaseShiftBackward(Ax,Ay,Az,ux,uy,uz,Delta1[i],Delta3[i],domain);
    	phaseShiftBackward(Bx,wy,Bz,wx,wy,wz,Delta1[i],Delta3[i],domain);
	

	fftBackward(Ax,domain);
	fftBackward(Ay,domain);
	fftBackward(Az,domain);
	
	fftBackward(Bx,domain);
	fftBackward(wy,domain);
	fftBackward(Bz,domain);	

	// Calculate rotor
	
	calcRotor(Bx,wy,Bz,Ax,Ay,Az,domain);

	// Transform back to Fourier space		

	fftForward(Bx,domain);
	fftForward(wy,domain);
	fftForward(Bz,domain);		

	//Backward phase shift

	
	phaseShiftForward(Cx,Cy,Cz,Bx,wy,Bz,-Delta1[i],-Delta3[i],domain);
        
	}

	calcWy(wy,ux,uz,domain);

	//INCLUDE ROUTINE TO EVALUATE MAX VELOCITIES
	// FOR CFL CONDITION

	fftBackward(ux,domain);
	fftBackward(uy,domain);
	fftBackward(uz,domain);

	fftBackward(wx,domain);
	fftBackward(wy,domain);
	fftBackward(wz,domain);	

	// Calculate rotor
	
	calcRotor(wx,wy,wz,ux,uy,uz,domain);

	// Transform back to Fourier space		

	fftForward(wx,domain);
	fftForward(wy,domain);
	fftForward(wz,domain);	
	
	sumCon(wx,wy,wz,Cx,Cy,Cz,domain);

	forwardTranspose(wx,domain);
	forwardTranspose(wy,domain);
	forwardTranspose(wz,domain);	


	return;

			
}







