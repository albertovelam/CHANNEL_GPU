#include"channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

static void __global__ rk_step_1_orig(float2* ux,float2* uy,float2* u_wx,float2* u_wy,float2* rx, float2* ry,float dt,int nc_next,domain_t domain)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j=k%NY;
  k=(k-j)/NY;
  // [i,k,j][NX,NZ,NY]  
  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE && j<NY && k<NZ)
    {
      const float alpha[]={ 29.0f/96.0f, -3.0f/40.0f, 1.0f/6.0f};
      const float dseda[]={ 0.0f, -17.0f/60.0f, -5.0f/12.0f};

      float2 u1;
      float2 u2;
      float2 r1;
      float2 r2;
      float2 u_w1;
      float2 u_w2;
      float k1;
      float k3;
      float kk;
      // X indices              
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX;
      // Z indices
      k3=(float)k;
      //Set to LX and LZ
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;

      kk=k1*k1+k3*k3;

      float nu=1.0f/REYNOLDS;

      // Read u and R from global memory
      u1=ux[h];
      u2=uy[h];
      u_w1=u_wx[h];
      u_w2=u_wy[h];
      r1=rx[h];
      r2=ry[h];

      // u=(alpha(0)+betha(0))*Lu+dseda*R

      //BUG RK3 solved  
      u_w1.x=u1.x+dt*(alpha[nc_next]*nu*(u_w1.x-kk*u1.x)+dseda[nc_next]*r1.x);
      u_w1.y=u1.y+dt*(alpha[nc_next]*nu*(u_w1.y-kk*u1.y)+dseda[nc_next]*r1.y);

      u_w2.x=u2.x+dt*(alpha[nc_next]*nu*(u_w2.x-kk*u2.x)+dseda[nc_next]*r2.x);
      u_w2.y=u2.y+dt*(alpha[nc_next]*nu*(u_w2.y-kk*u2.y)+dseda[nc_next]*r2.y);

      //Write to global memory
      u_wx[h]=u_w1;
      u_wy[h]=u_w2;

    }

}

static void __global__ rk_step_1(float2* ux,float2* uy,float2* u_wx,float2* u_wy,float2* rx, float2* ry,float dt,int nc_next,domain_t domain)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int j = h%NY;
  int i = h/(NY*NZ);
  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)
    {
      const float alpha[]={ 29.0f/96.0f, -3.0f/40.0f, 1.0f/6.0f};
      const float dseda[]={ 0.0f, -17.0f/60.0f, -5.0f/12.0f};
	
      float2 u1;
      float2 u2;
      float2 r1;
      float2 r2;
      float2 u_w1;
      float2 u_w2;
      float k1;
      float k3;
      float kk;
      // X indices		
      k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX;
      // Z indices
      k3=(float)k;
      //Set to LX and LZ
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;		
	
      kk=k1*k1+k3*k3;

      float nu=1.0f/REYNOLDS;

      // Read u and R from global memory
      u1=ux[h];
      u2=uy[h];
      u_w1=u_wx[h];
      u_w2=u_wy[h];
      r1=rx[h];
      r2=ry[h];
	
      // u=(alpha(0)+betha(0))*Lu+dseda*R

      //BUG RK3 solved	
      u_w1.x=u1.x+dt*(alpha[nc_next]*nu*(u_w1.x-kk*u1.x)+dseda[nc_next]*r1.x);
      u_w1.y=u1.y+dt*(alpha[nc_next]*nu*(u_w1.y-kk*u1.y)+dseda[nc_next]*r1.y);
	
      u_w2.x=u2.x+dt*(alpha[nc_next]*nu*(u_w2.x-kk*u2.x)+dseda[nc_next]*r2.x);
      u_w2.y=u2.y+dt*(alpha[nc_next]*nu*(u_w2.y-kk*u2.y)+dseda[nc_next]*r2.y);

      //Write to global memory
      u_wx[h]=u_w1;
      u_wy[h]=u_w2;
	
    }
	
}

static void __global__ rk_step_2(float2* u_wx,float2* u_wy,float2* rx, float2* ry,float dt,int nc,domain_t domain)
{
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  int j = h%NY;
  int i = h/(NY*NZ);
  int k = (h-i*NY*NZ)/NY;
  if(h<NXSIZE*NY*NZ)

    {

      const float gammad[]={ 8.0f/15.0f, 5.0f/12.0f, 3.0f/4.0f};
      float2 r1;
      float2 r2;
      float2 u_w1;
      float2 u_w2;
      // Read u and R from global memory
      u_w1=u_wx[h];
      u_w2=u_wy[h];
      r1=rx[h];
      r2=ry[h];
	
      // u=(alpha(0)+betha(0))*Lu+dseda*R
      u_w1.x=u_w1.x+dt*gammad[nc]*r1.x;
      u_w1.y=u_w1.y+dt*gammad[nc]*r1.y;
	
      u_w2.x=u_w2.x+dt*gammad[nc]*r2.x;
      u_w2.y=u_w2.y+dt*gammad[nc]*r2.y;
	

      //Write to global memory
      u_wx[h]=u_w1;
      u_wy[h]=u_w2;
	
    }
	
}

extern void RKstep_1(float2* ddv,float2* g,float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in,domain_t domain)
{
START_RANGE("RKstep_1",31)
  CHECK_CUDART( cudaMemcpy(ddv_w,ddv,SIZE, cudaMemcpyDeviceToDevice) );
  CHECK_CUDART( cudaMemcpy(g_w,g,SIZE, cudaMemcpyDeviceToDevice) );

	
  //Derivada segunda YY stored in ddv_w y g_w
  deriv_YY_HO_double(ddv_w, domain);
  deriv_YY_HO_double(g_w, domain);
/*  
  threadsPerBlock.x= THREADSPERBLOCK_IN;
  threadsPerBlock.y= THREADSPERBLOCK_IN;

  blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.x=NZ*NY/threadsPerBlock.y;
  
  //first step
  rk_step_1<<<blocksPerGrid,threadsPerBlock>>>(ddv,g,ddv_w,g_w,Rddv,Rg,dt,in,domain);
  kernelCheck(RET,domain,"RK");	
*/

  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;
  rk_step_1<<<grid,block>>>(ddv,g,ddv_w,g_w,Rddv,Rg,dt,in,domain);


//END_RANGE_ASYNC 
  return;
}

extern void RKstep_2(float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in,domain_t domain)
{
START_RANGE("RKstep_2",32)
/*
  threadsPerBlock.x= THREADSPERBLOCK_IN;
  threadsPerBlock.y= THREADSPERBLOCK_IN;

  blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
  blocksPerGrid.x=NZ*NY/threadsPerBlock.y;
*/
  dim3 grid,block;
  block.x = 128;
  grid.x = (NXSIZE*NY*NZ + block.x - 1)/block.x;

  // RK substeps
  rk_step_2<<<grid,block>>>(ddv_w,g_w,Rddv,Rg,dt,in,domain);
//  kernelCheck(RET,domain,"RK");
END_RANGE
  return;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


#define UMAX_BLK (128)
#define UMAX_GRD (70)

static __global__ void calcUmaxKernel(const float2 * __restrict ux, const float2 * __restrict uy, const float2 * __restrict uz, float * __restrict temp, const int elements)
{
  __shared__ float scratch[UMAX_BLK];
  int index;
  float2 max={0.f,0.f};
  //const int stride = UMAX_BLK*gridDim.x;
  const float2 *data;
  if(blockIdx.y==0) data = ux;
  if(blockIdx.y==1) data = uy;
  if(blockIdx.y==2) data = uz; // = blockIdx.y*elements;
  for(index=threadIdx.x+blockIdx.x*UMAX_BLK; index<elements; index+=UMAX_BLK*UMAX_GRD)
  {
    float2 val  = data[index];
    if(val.x<0.f) val.x = -val.x;
    if(val.y<0.f) val.y = -val.y;
    if(val.x>max.x) max.x = val.x;
    if(val.y>max.y) max.y = val.y;
  }
  if(max.y > max.x) max.x = max.y;
  scratch[threadIdx.x] = max.x;
  __syncthreads();
  for(int offset=UMAX_BLK/2; offset>0; offset/=2)
  {
    if(threadIdx.x<offset){
      float val=scratch[threadIdx.x+offset];
      if(val>max.x){
        max.x = val;
        scratch[threadIdx.x] = val;
      }
    }
    __syncthreads();
  }
  if(threadIdx.x==0) atomicMax(&temp[blockIdx.y],scratch[0]);
}



void calc_Umax2(float2* ux, float2* uy, float2* uz, float* temp,domain_t domain)
{
  int elements=NXSIZE*NY*NZ;
  blocksPerGrid.x = UMAX_GRD;//SMCOUNT*8;
  blocksPerGrid.y = 3;

  //printf("elements = %d \n",elements);
  calcUmaxKernel<<<blocksPerGrid,UMAX_BLK,0,compute_stream>>>(ux,uy,uz,temp,elements);
}

void calc_Dmax2(float2* ux, float2* uy, float* temp,domain_t domain)
{ 
  int elements=NXSIZE*NY*NZ;
  blocksPerGrid.x = UMAX_GRD;//SMCOUNT*8;
  blocksPerGrid.y = 2;
  
  //printf("elements = %d \n",elements);
  calcUmaxKernel<<<blocksPerGrid,UMAX_BLK,0,compute_stream>>>(ux,uy,NULL,temp,elements);
} 

