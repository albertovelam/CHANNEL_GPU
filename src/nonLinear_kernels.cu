#include "channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

///////////////////KERNELS////////////////////////

static __global__ void calcUWkernel(float2* ux,float2* uz, float2* f,
				    float2* g,domain_t domain)
{
/*	
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

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

      float kk;
	
	
      // X indices		
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
	
      // Z indices
      k3=(float)k;

      //Fraction
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;	

	
      kk=k1*k1+k3*k3;
      if(i+IGLOBAL==0 & k==0)kk=1.0f;

      // Read {u1,u2,u3}	
	
      float2 fl=f[h];
      float2 gl=g[h];
	
      float2 u1;
      float2 u3;

      //f=d_y v

      fl.x=-fl.x;
      fl.y=-fl.y;

      //ux=-1/kk(ikx*f+ikz*g)	

      u1.x=-(k1*fl.y+k3*gl.y);
      u1.y=  k1*fl.x+k3*gl.x ;

      u1.x=-u1.x/kk;
      u1.y=-u1.y/kk;

      //ux=-1/kk(ikz*f-ikx*g)	

      u3.x=-(k3*fl.y-k1*gl.y);
      u3.y=  k3*fl.x-k1*gl.x ;

      u3.x=-u3.x/kk;
      u3.y=-u3.y/kk;

      //turn the mean to zero	

      if(i+IGLOBAL==0 & k==0){
	u1.x=0.0f;
	u3.x=0.0f;
	
	u1.y=0.0f;
	u3.y=0.0f;
      }	
	

      ux[h]=u1;
      uz[h]=u3;

	
    }
	
	
}

static __global__ void calcHvgkernel(float2* hx,float2* hz,domain_t domain)
{
/*	
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;
	
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

      // X indices		
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
	
      // Z indices
      k3=(float)k;

      //Fraction
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;	

      // Read {hx,hz}	
	
      float2 h1=hx[h];
      float2 h3=hz[h];
	
      float2 hg_t;
      float2 hv_t;

      hv_t.x=-(-k1*h1.y-k3*h3.y);
      hv_t.y=  -k1*h1.x-k3*h3.x ;
	
      hg_t.x=-(k3*h1.y-k1*h3.y);
      hg_t.y=  k3*h1.x-k1*h3.x ;

	
      hx[h]=hv_t;
      hz[h]=hg_t;

	
    }
	
	
}

static __global__ void calcHvvkernel(float2* hx,float2* hy,domain_t domain)
{
/*
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;
	
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

      float kk;

      // X indices		
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
	
      // Z indices
      k3=(float)k;

      //Fraction
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;		

      kk=k1*k1+k3*k3;

      // Read {u1,u2,u3}	
	
      float2 h1=hx[h];
      float2 h2=hy[h];
	
      h1.x=h1.x-kk*h2.x;
      h1.y=h1.y-kk*h2.y;

      // Write	
      hx[h]=h1;

	
    }
	
	
}

///////////////////FUNCTIONS////////////////////////

extern void calcUW(float2* ux,float2* uz, float2* f,float2* g,domain_t domain){
START_RANGE("calcUW",22)
/*
  threadsPerBlock.x= THREADSPERBLOCK_IN;
  threadsPerBlock.y= THREADSPERBLOCK_IN;

  blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.x=NZ*NY/threadsPerBlock.y;

  //Calcs ux and uz out of dd_v and w_y	

  calcUWkernel<<<blocksPerGrid,threadsPerBlock>>>(ux,uz,f,g,domain);
  kernelCheck(RET,domain,"W_kernel");
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  calcUWkernel<<<grid,block>>>(ux,uz,f,g,domain);

END_RANGE
  return;

}


extern void calcHvg(float2* nl_x,float2* nl_y,float2* nl_z, domain_t domain){
START_RANGE("calcHvg",23)
  //Calcs h_g and h_v out of nonlinear terms in x,y and z
/*	
  threadsPerBlock.x= THREADSPERBLOCK_IN;
  threadsPerBlock.y= THREADSPERBLOCK_IN;

  blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.x=NZ*NY/threadsPerBlock.y;

  calcHvgkernel<<<blocksPerGrid,threadsPerBlock>>>(nl_x,nl_z,domain);
  kernelCheck(RET,domain,"W_kernel");
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;

  calcHvgkernel<<<grid,block>>>(nl_x,nl_z,domain);
  deriv_Y_HO_double(nl_x, domain);
  calcHvvkernel<<<grid,block>>>(nl_x,nl_y,domain);
//  kernelCheck(RET,domain,"W_kernel");

  return;

}

