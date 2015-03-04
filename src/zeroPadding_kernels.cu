#include"channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

///////////////////KERNELS////////////////////////

static __global__ void padForward_kernel(float2* v,float2* u,domain_t domain)
{

	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZP;
	i=(i-k)/NZP;

	// [i,k,j][NX,NZ,NY]	

	float2 aux;

	float factor=(float)(NXP*2*(NZP-1))/(float)(NX*2*(NZ-1));
	

	if (j<NYSIZE/NSTEPS_CONV && i<NXP && k<NZP)
	{*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZP;
	int j = h/(NXP*NZP);
	int i = (h-j*NXP*NZP)/NXP;
  	if(h<NYSIZE/NSTEPS_CONV*NXP*NZP)
    	{
	
	float factor=(float)(NXP*2*(NZP-1))/(float)(NX*2*(NZ-1));

	int ip;
	int kp;
	
	float2 aux;
	aux.x=0.0;
	aux.y=0.0;

	if((i<NX/2 || i>NX-1) && k<NZ){

	// X indices		
	int ip=i<NX/2 ? i : i-NX/2;
	
	// Z indices
	int kp=k;


	int h =j*NX*NZ+ip*NZ+kp;
	
	
	aux=u[h];
	
	aux.x*=factor;
	aux.y*=factor;

	}	
	
	int hp=j*NXP*NZP+i*NZP+k;
	v[hp]=aux;
		
	}


	

}

static __global__ void padBackward_kernel(float2* u,float2* v,domain_t domain)
{

	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZP;
	i=(i-k)/NZP;

	// [i,k,j][NX,NZ,NY]	

	float2 aux;

	float factor=(float)(NX*2*(NZ-1))/(float)(NXP*2*(NZP-1));

	if (j<NYSIZE/NSTEPS_CONV && i<NXP && k<NZP)
	{
	*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZP;
	int j = h/(NXP*NZP);
	int i = (h-j*NXP*NZP)/NXP;
  	if(h<NYSIZE/NSTEPS_CONV*NXP*NZP)
    	{

	float factor=(float)(NX*2*(NZ-1))/(float)(NXP*2*(NZP-1));

	int ip;
	int kp;
	
	float2 aux;
	aux.x=0.0;
	aux.y=0.0;

	if((i<NX/2 || i>NX-1) && k<NZ){

	// X indices		
	int ip=i<NX/2 ? i : i-NX/2;
	
	// Z indices
	int kp=k;

	int hp=j*NXP*NZP+i*NZP+k;
	int h =j*NX*NZ+ip*NZ+kp;
	
	aux=v[hp];
	aux.x*=factor;
	aux.y*=factor;
	u[h]=aux;

	}	
	

	}


	


}

static __global__ void rotorZP_kernel(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz,domain_t domain)
{

	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZP;
	i=(i-k)/NZP;

		// [i,k,j][NX,NZ,NY]	

	// [j,i,k][NY,NX,NZ]	

	int h=j*NXP*NZP+i*NZP+k;


	if ( j<NYSIZE/NSTEPS_CONV && i<NXP && k<NZP)
	{
	*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZP;
	int j = h/(NXP*NZP);
	int i = (h-j*NXP*NZP)/NXP;
  	if(h<NYSIZE/NSTEPS_CONV*NXP*NZP)
    	{
	float2 m1;
	float2 m2;
	float2 m3;
	
	
	//Normalisation

	float N2=(float)NXP*(2*NZP-2);

	// Read velocity and vorticity	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	float2 w1=wx[h];
	float2 w2=wy[h];
	float2 w3=wz[h];
	
	// Normalize velocity and vorticity
	
	u1.x=u1.x/N2;
	u2.x=u2.x/N2;
	u3.x=u3.x/N2;

	u1.y=u1.y/N2;
	u2.y=u2.y/N2;
	u3.y=u3.y/N2;

	w1.x=w1.x/N2;
	w2.x=w2.x/N2;
	w3.x=w3.x/N2;
		
	w1.y=w1.y/N2;
	w2.y=w2.y/N2;
	w3.y=w3.y/N2;
	
	// Calculate the convolution rotor

	m1.x=u2.x*w3.x-u3.x*w2.x;
	m2.x=u3.x*w1.x-u1.x*w3.x;
	m3.x=u1.x*w2.x-u2.x*w1.x;

	m1.y=u2.y*w3.y-u3.y*w2.y;
	m2.y=u3.y*w1.y-u1.y*w3.y;
	m3.y=u1.y*w2.y-u2.y*w1.y;

	// Output must be normalized with N^3	
	
	wx[h].x=m1.x;
	wx[h].y=m1.y;

	wy[h].x=m2.x;
	wy[h].y=m2.y;

	wz[h].x=m3.x;
	wz[h].y=m3.y;	


	}
	
	
}

extern void padForward(float2* aux,float2* u,domain_t domain)
{

	//Size of aux = NYSIZE/NSTEPS_CONV*(3*NX)*((3*(2*NZ-2)/4+1))
	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/NSTEPS_CONV/threadsPerBlock.y;
	blocksPerGrid.x=NXP*NZP/threadsPerBlock.x;
	*/

	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE/NSTEPS_CONV*NXP*NZP + block.x - 1)/block.x;

	padForward_kernel<<<grid,block>>>(aux,u,domain);
	kernelCheck(RET,domain,"dealias");
	
	return;

}

extern void padBackward(float2* u,float2* aux,domain_t domain)
{

	//Size of aux = NYSIZE/NSTEPS_CONV*(3/2*NX)*((3*(2*NZ-2)/4+1))
	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/NSTEPS_CONV/threadsPerBlock.y;
	blocksPerGrid.x=NXP*NZP/threadsPerBlock.x;
	*/

	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE/NSTEPS_CONV*NXP*NZP + block.x - 1)/block.x;

	padBackward_kernel<<<grid,block>>>(u,aux,domain);
	kernelCheck(RET,domain,"dealias");
	
	return;

}

extern void calcRotorZeroPadding(float2* wx,float2* wy,float2* wz,float2* u,float2* v,float2* w,domain_t domain){


	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/NSTEPS_CONV/threadsPerBlock.y;
	blocksPerGrid.x=NXP*NZP/threadsPerBlock.x;
	*/

	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE/NSTEPS_CONV*NXP*NZP + block.x - 1)/block.x;

	rotorZP_kernel<<<grid,block>>>(wx,wy,wz,u,v,w,domain);
	kernelCheck(RET,domain,"dealias");


}



