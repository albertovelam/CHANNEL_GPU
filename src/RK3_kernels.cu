#include"channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

static void __global__ rk_step_1(float2* ux,float2* uy,float2* u_wx,float2* u_wy,float2* rx, float2* ry,float dt,int nc_next,int IGLOBAL)
{
	

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

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

static void __global__ rk_step_2(float2* u_wx,float2* u_wy,float2* rx, float2* ry,float dt,int nc,int IGLOBAL)
{
	

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;


	if (i<NXSIZE && j<NY && k<NZ)
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

extern void RKstep_1(float2* ddv,float2* g,float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in)
{
	
	cudaCheck(cudaMemcpy(ddv_w,ddv,SIZE, cudaMemcpyDeviceToDevice),"MemInfo_ABCN");
	cudaCheck(cudaMemcpy(g_w,g,SIZE, cudaMemcpyDeviceToDevice),"MemInfo_ABCN");

	
	//Derivada segunda YY stored in ddv_w y g_w
	deriv_YY_HO(ddv_w);
	deriv_YY_HO(g_w);

	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	//first step
	rk_step_1<<<blocksPerGrid,threadsPerBlock>>>(ddv,g,ddv_w,g_w,Rddv,Rg,dt,in,IGLOBAL);
	kernelCheck(RET,"RK");	
	
	return;
}

extern void RKstep_2(float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in)
{

	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;


	// RK substeps
	 rk_step_2<<<blocksPerGrid,threadsPerBlock>>>(ddv_w,g_w,Rddv,Rg,dt,in,IGLOBAL);
	 kernelCheck(RET,"RK");

	return;
}


