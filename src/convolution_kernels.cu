#include "channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


static __global__ void calcOmegakernel(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz,int IGLOBAL)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;


	if (i<NXSIZE && j<NY && k<NZ)
	{
	
	float k1;
	float k3;

	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
	
	// Z indices
	k3=(float)k;	

	//Set to LX and LZ
	//Fraction
	k1=(PI2/LX)*k1;
	k3=(PI2/LZ)*k3;	

	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];

	float2 w1=wx[h];
	float2 w3=wz[h];
	float2 w2;

	w1.x=w1.x -(-k3*u2.y);
	w1.y=w1.y -k3*u2.x ;
	
	w2.x=-(k3*u1.y-k1*u3.y);
	w2.y=  k3*u1.x-k1*u3.x ;
	
	w3.x=-w3.x-(k1*u2.y);
	w3.y=-w3.y+ k1*u2.x ;		
		
	//Write
	
	wx[h]=w1;
	wy[h]=w2;
	wz[h]=w3;
	
	
	
	}
	
	
}


static __global__ void rotorkernel(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz,int IGLOBAL)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;


	if (i<NXSIZE && j<NY && k<NZ)
	{

	float2 m1;
	float2 m2;
	float2 m3;
	
	
	//Normalisation

	int N2=NX*(2*NZ-2);

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

///////////////////FUNCTIONS///////////////////////

extern void calcOmega(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain){

	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;


	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;


	calcOmegakernel<<<blocksPerGrid,threadsPerBlock>>>(wx,wy,wz,ux,uy,uz,domain.iglobal);
	kernelCheck(RET,domain,"Boundary");


}

extern void calcRotor(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain){

	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;


	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	rotorkernel<<<blocksPerGrid,threadsPerBlock>>>(wx,wy,wz,ux,uy,uz,domain.iglobal);
	kernelCheck(RET,domain,"Boundary");


}



