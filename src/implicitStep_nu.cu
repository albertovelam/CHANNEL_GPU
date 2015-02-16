#include"channel.h"


static __global__ void rhs_A_kernel(float2* u, domain_t domain)
{  

  //Define shared memory


  __shared__ float2 sf[NY+2];

  int k   = blockIdx.x;
  int i   = blockIdx.y;

  int j   = threadIdx.x;


  int h=i*NZ*NY+k*NY+j;

  float2 u_temp;
		
  float2 ap_1;
  float2 ac_1;
  float2 am_1;

  float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
  float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	

  float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
  float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);


  if(i<NXSIZE & k<NZ & j<NY){

    //Read from global so shared

    sf[j+1]=u[h];

    __syncthreads();

    ap_1=sf[j+2];	
    ac_1=sf[j+1];
    am_1=sf[j];
		
		
    u_temp.x=(alpha*ap_1.x+ac_1.x+betha*am_1.x);
    u_temp.y=(alpha*ap_1.y+ac_1.y+betha*am_1.y);


    if(j==0){
      u_temp.x=0.0f;
      u_temp.y=0.0f;		
    }		
	
    if(j==NY-1){
      u_temp.x=0.0f;
      u_temp.y=0.0f;		
    }	

    u[h]=u_temp;
 	
  }

}

static __global__ void setDiagkernel(float2* ldiag,float2* cdiag,float2* udiag,const float betha_RK,const float dt,domain_t domain){  

	
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
	
      //Fraction
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;	

      kk=k1*k1+k3*k3;

      //Coeficient

      float nu=1.0f/REYNOLDS;
      float D=nu*dt*betha_RK;

      //COEFICIENTS OF THE NON_UNIFORM GRID

      float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
	

      float A=-12.0f*b/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
      float B= 12.0f*a/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
      float C=-A-B;
		
      float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
      float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);

		
		

      //veamos
	
      float2 ldiag_h;
      float2 cdiag_h;
      float2 udiag_h;
	
      ldiag_h.x=betha-D*B+D*kk*betha;
      ldiag_h.y=0.0f;			
	
      cdiag_h.x=1.0f-D*C+D*kk*1.0f;
      cdiag_h.y=0.0f;		
	
      udiag_h.x=alpha-D*A+D*kk*alpha;
      udiag_h.y=0.0f;	

      //To be improved 
		
      if(j==0){
	ldiag_h.x=0.0f;
	cdiag_h.x=1.0f;
	udiag_h.x=0.0f;
      }
	
      if(j==1){
	ldiag_h.x=0.0f;
      }

      if(j==NY-1){
	ldiag_h.x=0.0f;
	cdiag_h.x=1.0f;
	udiag_h.x=0.0f;
      }	
	
      if(j==NY-2){
	udiag_h.x=0.0f;
      }

      // Write		

      ldiag[h]=ldiag_h;
      cdiag[h]=cdiag_h;
      udiag[h]=udiag_h;			
	
    }

}


static float2* udiag;
static float2* cdiag;
static float2* ldiag;


static dim3 threadsPerBlock_A;
static dim3 blocksPerGrid_A;

static dim3 threadsPerBlock_B;
static dim3 blocksPerGrid_B;


static cusparseHandle_t implicit_handle;

void setImplicit(domain_t domain){


  cusparseCheck(cusparseCreate(&implicit_handle),domain,"Handle");	

  cudaCheck(cudaMalloc(&udiag,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&ldiag,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&cdiag,SIZE),domain,"C");

  //Set work groups for A matrix

  threadsPerBlock_A.x=NY;
  threadsPerBlock_A.y=1;

  blocksPerGrid_A.x=NZ;
  blocksPerGrid_A.y=NX;

  //Set work groups for B matrix

  threadsPerBlock_B.x=THREADSPERBLOCK_IN;
  threadsPerBlock_B.y=THREADSPERBLOCK_IN;
	
  blocksPerGrid_B.x=NXSIZE/THREADSPERBLOCK_IN;
  blocksPerGrid_B.y=NZ*NY/THREADSPERBLOCK_IN;

  return;


}

static void setDiag(float2* lidag,float2* cdiag,float2* udiag,float betha,float dt,domain_t domain){

  setDiagkernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(ldiag,cdiag,udiag,betha,dt,domain);
  kernelCheck(RET,domain,"deriv_kernel");	

  return;
}


extern void implicitSolver(float2* u,float betha,float dt,domain_t domain){

  rhs_A_kernel<<<blocksPerGrid_A,threadsPerBlock_A>>>(u,domain);
  kernelCheck(RET,domain,"deriv_kernel");	

  setDiag(ldiag,cdiag,udiag,betha,dt,domain);

  cusparseCheck(cusparseCgtsvStridedBatch(implicit_handle,NY,ldiag,cdiag,udiag,u,NXSIZE*NZ,NY),domain,"HEM");

  return;

}






