
#include "channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

///////////////////KERNELS////////////////////////

static __global__ void dealiaskernel(float2* u,domain_t domain)
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

      float kk;

      // X indices		
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
	
      // Z indices
      k3=(float)k;	
	
      //Dealias
      if( abs(k1)>floorf(NX/3) || abs(k3)>floorf((2*NZ-2)/3)){
	u[h].x=0.0f;
	u[h].y=0.0f;}
	
	
    }
	
	
}

static __global__ void zerokernel(float2* u1, domain_t domain)
{
	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
    {
		
      u1[h].x=0.0f;
      u1[h].y=0.0f;

	
    }
	
	
}

static __global__ void normalizekernel(float2* u1, domain_t domain)
{
	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
    {
	
      int N2=NX*(2*NZ-2);		

      u1[h].x/=N2;
      u1[h].y/=N2;

	
    }
	
	
}

static __global__ void scalekernel(float2* u,float S,domain_t domain)
{
	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
    {
	
      u[h].x*=S;
      u[h].y*=S;
	
	
    }
	
	
}

extern void dealias(float2* u, domain_t domain){

		
  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  dealiaskernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary");
	


  return;

}





extern void set2zero(float2* u, domain_t domain){


  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  zerokernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary");	
	
  return;
}

extern void normalize(float2* u, domain_t domain){


  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  normalizekernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary"); 
	
  return;
}

extern void scale(float2* u,float S,domain_t domain){


  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  scalekernel<<<blocksPerGrid,threadsPerBlock>>>(u,S,domain);
  kernelCheck(RET,domain,"Boundary");	
	
  return;
}



