
#include "channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

///////////////////KERNELS////////////////////////

static __global__ void dealiaskernel(float2* u,domain_t domain)
{
/*	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
*/
  int h = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = h%NY;
  int i = h/(NY*NZ);
  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)
    {
	
      float k1;
      float k3;

//      float kk;

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
/*	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
*/
  int h = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = h%NY;
//  int i = h/(NY*NZ);
//  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)
    {
		
      u1[h].x=0.0f;
      u1[h].y=0.0f;

	
    }
	
	
}

static __global__ void normalizekernel(float2* u1, domain_t domain)
{
/*	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
*/
  int h = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = h%NY;
//  int i = h/(NY*NZ);
//  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)
    {
	
      int N2=NX*(2*NZ-2);		

      u1[h].x/=N2;
      u1[h].y/=N2;

	
    }
	
	
}

static __global__ void scalekernel(float2* u,float S,domain_t domain)
{
/*	
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
*/
  int h = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = h%NY;
//  int i = h/(NY*NZ);
//  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)
    {
      u[h].x*=S;
      u[h].y*=S;
    }
	
}

extern void dealias(float2* u, domain_t domain){
START_RANGE("dealias",12)
/*		
  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  dealiaskernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary");
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  dealiaskernel<<<grid,block>>>(u,domain);	

END_RANGE
  return;

}





extern void set2zero(float2* u, domain_t domain){
START_RANGE("set2zero",13)
/*
  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  zerokernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary");	
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  zerokernel<<<grid,block>>>(u,domain);

END_RANGE	
  return;
}

extern void normalize(float2* u, domain_t domain){

START_RANGE("normalize",14)
/*
  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  normalizekernel<<<blocksPerGrid,threadsPerBlock>>>(u,domain);
  kernelCheck(RET,domain,"Boundary"); 
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  normalizekernel<<<grid,block>>>(u,domain);

END_RANGE	
  return;
}

extern void scale(float2* u,float S,domain_t domain){
START_RANGE("scale",15)
/*
  threadsPerBlock.x=THREADSPERBLOCK_IN;
  threadsPerBlock.y=THREADSPERBLOCK_IN;


  blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	
  scalekernel<<<blocksPerGrid,threadsPerBlock>>>(u,S,domain);
  kernelCheck(RET,domain,"Boundary");	
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  scalekernel<<<grid,block>>>(u,S,domain);

END_RANGE	
  return;
}



