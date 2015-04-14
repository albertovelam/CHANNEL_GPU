#include"channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

///////////////////KERNELS////////////////////////

static __global__ void shift_B_kernel(float2* ux,float2* uy,float2* uz,float2* tx,float2* ty,float2* tz,float Delta_1,float Delta_3,domain_t domain)
{
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZ;
	i=(i-k)/NZ;

	// [i,k,j][NX,NZ,NY]	

	int h=j*NX*NZ+i*NZ+k;

	if (j<NYSIZE && i<NX && k<NZ)
	{
	*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZ;
	int j = h/(NX*NZ);
	int i = (h-j*NX*NZ)/NX;
  	if(h<NYSIZE*NX*NZ)
    	{
	float k1;
	float k3;

	
	// X indices		
	k1=(i)<NX/2 ? (float)(i) : (float)(i)-(float)NX ;
	
	// Z indices
	k3=(float)k;

	//Fraction
	k1=(PI2/LX)*k1;
	k3=(PI2/LZ)*k3;	


	float2 t1=tx[h];
	float2 t2=ty[h];
	float2 t3=tz[h];
	
	float aux_x;
	float aux_y;

	// Phase shifting by Delta;

	Delta_1=Delta_1*LX/(float)NX;
	Delta_3=Delta_3*LZ/(float)(2*(NZ-1));	
	
	float sine=sin(k1*Delta_1+k3*Delta_3);
	float cosine=cos(k1*Delta_1+k3*Delta_3);
	
	//t1;

	aux_x=cosine*t1.x-sine*t1.y;
	aux_y=sine*t1.x+cosine*t1.y;

	t1.x=aux_x;
	t1.y=aux_y;
	
	//t2;

	aux_x=cosine*t2.x-sine*t2.y;
	aux_y=sine*t2.x+cosine*t2.y;

	t2.x=aux_x;
	t2.y=aux_y;	

	//t3	
	
	aux_x=cosine*t3.x-sine*t3.y;
	aux_y=sine*t3.x+cosine*t3.y;

	t3.x=aux_x;
	t3.y=aux_y;	

	//Remove oddball wavenumber	
	if(k==NZ-1){
	
	t1.x=0.0f;
	t1.y=0.0f;

	t2.x=0.0f;
	t2.y=0.0f;	
	
	t3.x=0.0f;
	t3.y=0.0f;
	}

	ux[h]=t1;
	uy[h]=t2;
	uz[h]=t3;



	}

}

static __global__ void shift_F_kernel(float2* ux,float2* uy,float2* uz,float2* tx,float2* ty,float2* tz,float Delta_1,float Delta_3,domain_t domain)
{
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZ;
	i=(i-k)/NZ;

	// [i,k,j][NX,NZ,NY]	

	int h=j*NX*NZ+i*NZ+k;

	if (j<NYSIZE && i<NX && k<NZ)
	{*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZ;
	int j = h/(NX*NZ);
	int i = (h-j*NX*NZ)/NX;
  	if(h<NYSIZE*NX*NZ)
    	{
	
	float k1;
	float k3;

	
	// X indices		
	k1=(i)<NX/2 ? (float)(i) : (float)(i)-(float)NX ;
	
	// Z indices
	k3=(float)k;

	//Fraction
	k1=(PI2/LX)*k1;
	k3=(PI2/LZ)*k3;	


	float2 t1=tx[h];
	float2 t2=ty[h];
	float2 t3=tz[h];
	
	float aux_x;
	float aux_y;

	// Phase shifting by Delta;

	Delta_1=Delta_1*LX/(float)NX;
	Delta_3=Delta_3*LZ/(float)(2*(NZ-1));	
	
	float sine=sin(k1*Delta_1+k3*Delta_3);
	float cosine=cos(k1*Delta_1+k3*Delta_3);
	
	//t1;

	aux_x=cosine*t1.x-sine*t1.y;
	aux_y=sine*t1.x+cosine*t1.y;

	t1.x=aux_x;
	t1.y=aux_y;
	
	//t2;

	aux_x=cosine*t2.x-sine*t2.y;
	aux_y=sine*t2.x+cosine*t2.y;

	t2.x=aux_x;
	t2.y=aux_y;	

	//t3	
	
	aux_x=cosine*t3.x-sine*t3.y;
	aux_y=sine*t3.x+cosine*t3.y;

	t3.x=aux_x;
	t3.y=aux_y;	
	
	float2 a1=ux[h];
	float2 a2=uy[h];
	float2 a3=uz[h];

	a1.x+=t1.x;
	a1.y+=t1.y;

	a2.x+=t2.x;
	a2.y+=t2.y;

	a3.x+=t3.x;
	a3.y+=t3.y;	

	//Remove oddball wavenumber
	
	if(k==NZ-1){
	
	a1.x=0.0f;
	a1.y=0.0f;

	a2.x=0.0f;
	a2.y=0.0f;	
	
	a3.x=0.0f;
	a3.y=0.0f;
	}

	ux[h]=a1;
	uy[h]=a2;
	uz[h]=a3;



	}

}


static __global__ void add_kernel(float2* ux,float2* uy,float2* uz,float2* tx,float2* ty,float2* tz,domain_t domain)
{
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int k=i%NZ;
	i=(i-k)/NZ;

	// [i,k,j][NX,NZ,NY]	

	int h=j*NX*NZ+i*NZ+k;

	if (j<NYSIZE && i<NX && k<NZ)
	{*/
  	int h = blockIdx.x * blockDim.x + threadIdx.x;
	int k = h%NZ;
	int j = h/(NX*NZ);
	int i = (h-j*NX*NZ)/NX;
  	if(h<NYSIZE*NX*NZ)
    	{
	
	float k1;
	float k3;

	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i) : (float)(i)-(float)NX ;
	
	// Z indices
	k3=(float)k;

	//Fraction
	k1=(PI2/LX)*k1;
	k3=(PI2/LZ)*k3;	


	float2 t1=tx[h];
	float2 t2=ty[h];
	float2 t3=tz[h];

	
	float2 a1=ux[h];
	float2 a2=uy[h];
	float2 a3=uz[h];

	a1.x=0.25f*(a1.x+t1.x);
	a1.y=0.25f*(a1.y+t1.y);

	a2.x=0.25f*(a2.x+t2.x);
	a2.y=0.25f*(a2.y+t2.y);

	a3.x=0.25f*(a3.x+t3.x);
	a3.y=0.25f*(a3.y+t3.y);	

	ux[h]=a1;
	uy[h]=a2;
	uz[h]=a3;



	}

}
extern void phaseShiftBackward(float2* ux,float2* uy,float2* uz,float2* tx,float2* ty,float2* tz,float Delta1,float Delta3,domain_t domain)
{
	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NX*NZ/threadsPerBlock.y;
	*/
	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE*NX*NZ + block.x - 1)/block.x;

	shift_B_kernel<<<grid,block>>>(ux,uy,uz,tx,ty,tz,Delta1,Delta3,domain);
	kernelCheck(RET,domain,"dealias");
	
	return;

}

extern void phaseShiftForward(float2* ux,float2* uy,float2* uz,float2* tx,float2* ty,float2* tz,float Delta1,float Delta3,domain_t domain)
{
	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NX*NZ/threadsPerBlock.y;
	*/

	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE*NX*NZ + block.x - 1)/block.x;

	shift_F_kernel<<<grid,block>>>(ux,uy,uz,tx,ty,tz,Delta1,Delta3,domain);
	kernelCheck(RET,domain,"dealias");
	
	return;

}

extern void sumCon(float2* ax,float2* ay,float2* az,float2* tx,float2* ty,float2* tz,domain_t domain){
	/*
	threadsPerBlock.x= THREADSPERBLOCK_IN;
	threadsPerBlock.y= THREADSPERBLOCK_IN;

	blocksPerGrid.y=NYSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NX*NZ/threadsPerBlock.y;
	*/

	dim3 grid,block;
	block.x = 128;
	grid.x = (NYSIZE*NX*NZ + block.x - 1)/block.x;

	add_kernel<<<grid,block>>>(ax,ay,az,tx,ty,tz,domain);
	kernelCheck(RET,domain,"dealias");

}


