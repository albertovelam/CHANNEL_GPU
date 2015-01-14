#include"channel.h"

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


//KERNELS

__global__ void deriv_Y_kernel(float2* u)

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



		if(i<NXSIZE & k<NZ & j<NY){


		float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
		float c;

		float A=2.0f*(2.0f*a*b*b-b*b*b)/((a*a-a*b)*(a-b)*(a-b));
		float B=2.0f*(2.0f*b*a*a-a*a*a)/((b*b-a*b)*(a-b)*(a-b));
		float C=-A-B;
		float E;


		//Read from global to shared

		sf[j+1]=u[h];
						
		__syncthreads();


		ap_1=sf[j+2];
		ac_1=sf[j+1];	
		am_1=sf[j];
			
		u_temp.x=A*ap_1.x+C*ac_1.x+B*am_1.x;
		u_temp.y=A*ap_1.y+C*ac_1.y+B*am_1.y;

		//Second part multiplied by -1 to ensure simetry of the derivative
		
		if(j==0){
		
		a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh((j+1)*DELTA_Y-1.0f);

		
		A= -(3.0f*a+2.0f*b)/(a*a+a*b);
		B=  ((a+b)*(2.0f*b-a))/(a*b*b);
		C=  a*a/(b*b*a+b*b*b);

		u_temp.x=A*sf[1].x+B*sf[2].x+C*sf[3].x;
		u_temp.y=A*sf[1].y+B*sf[2].y+C*sf[3].y;			
		
		}		
	
		if(j==NY-1){
		
		a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh((j)*DELTA_Y-1.0f);
		b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh((j-1)*DELTA_Y-1.0f);

		
		A= -(3.0f*a+2.0f*b)/(a*a+a*b);
		B=  ((a+b)*(2.0f*b-a))/(a*b*b);
		C=  a*a/(b*b*a+b*b*b);

		u_temp.x=A*sf[NY].x+B*sf[NY-1].x+C*sf[NY-2].x;
		u_temp.y=A*sf[NY].y+B*sf[NY-1].y+C*sf[NY-2].y;				
					
		}	
				

		u[h]=u_temp;
 	
	  }

}


__global__ void deriv_YY_kernel(float2* u)

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



		if(i<NXSIZE & k<NZ & j<NY){
		
		float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
		float c;

		float A=-12.0f*b/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
		float B= 12.0f*a/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
		float C=-A-B;
		float E;

		//Read from global so shared

		sf[j+1]=u[h];

		__syncthreads();


		ap_1=sf[j+2];
		ac_1=sf[j+1];
		am_1=sf[j];	


		u_temp.x=A*ap_1.x+C*ac_1.x+B*am_1.x;
		u_temp.y=A*ap_1.y+C*ac_1.y+B*am_1.y;
		
		if(j==0){

		a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
		
		A= 6.0f/((a-b)*(2.0f*a-b));
		B= -6.0f*a/((a*b-b*b)*(2.0f*a-b));
	
		E=-A-B;

		u_temp.x=E*sf[1].x+A*sf[2].x+B*sf[3].x;
		u_temp.y=E*sf[1].y+A*sf[2].y+B*sf[3].y;				

		}		
	
		if(j==NY-1){

		a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);

		A= 6.0f/((a-b)*(2.0f*a-b));
		B= -6.0f*a/((a*b-b*b)*(2.0f*a-b));
	
		E=-A-B;

		u_temp.x=E*sf[NY].x+A*sf[NY-1].x+B*sf[NY-2].x;
		u_temp.y=E*sf[NY].y+A*sf[NY-1].y+B*sf[NY-2].y;			

		}	
		
		u[h]=u_temp;
 	
	  }

}



float2* udiag_y;
float2* cdiag_y;
float2* ldiag_y;

float2* udiag_yy;
float2* cdiag_yy;
float2* ldiag_yy;

static cusparseHandle_t derivatives_handle;

extern void setDerivatives_HO(domain_t domain){

  cusparseCheck(cusparseCreate(&derivatives_handle),domain,"Handle");
  
  
  float2* udia_host=(float2*)malloc(SIZE);
  float2* cdia_host=(float2*)malloc(SIZE);		
  float2* ldia_host=(float2*)malloc(SIZE);	
  
  for(int i=0;i<NXSIZE;i++){
    for(int k=0;k<NZ;k++){
      for(int j=0;j<NY;j++){
	
	
	int h=i*NZ*NY+k*NY+j;
	
	float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
	float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
	
	float alpha=b*b/((a-b)*(a-b));
	float betha=a*a/((a-b)*(a-b));
	
	udia_host[h].x=alpha;
	udia_host[h].y=0.0f;
	
	cdia_host[h].x=1.0f;
	cdia_host[h].y=0.0f;
	
	ldia_host[h].x=betha;
	ldia_host[h].y=0.0f;	
      }
		
      int h_0=i*NZ*NY+k*NY;	
      
      ldia_host[h_0+0].x=0.0f;
      udia_host[h_0+NY-1].x=0.0f;	
      
      //Derivatives in boundary condition for dY
      
      int j=0;
      
      float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      float b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh((j+1)*DELTA_Y-1.0f);
      
      
      float alpha=(a+b)/b;
      
      udia_host[h_0].x=alpha;
      cdia_host[h_0].x=1.0f;
      
      j=NY-1;
      
      a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh((j)*DELTA_Y-1.0f);
      b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh((j-1)*DELTA_Y-1.0f);
      
      alpha=(a+b)/b;			
      
      ldia_host[h_0+NY-1].x=alpha;
      cdia_host[h_0+NY-1].x=1.0f;
          
    }
  }
  
  
  //Copy to whatever
  
  cudaCheck(cudaMalloc(&udiag_y,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&ldiag_y,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&cdiag_y,SIZE),domain,"C");
  
  cudaCheck(cudaMemcpy(udiag_y,udia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");
  cudaCheck(cudaMemcpy(cdiag_y,cdia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");
  cudaCheck(cudaMemcpy(ldiag_y,ldia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");

  //Diagonals for dYY

  for(int i=0;i<NXSIZE;i++){
    for(int k=0;k<NZ;k++){
      for(int j=0;j<NY;j++){
	
	int h=i*NZ*NY+k*NY+j;
	
	float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
	float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);			
	
	float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
	float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
	
	udia_host[h].x=alpha;
	udia_host[h].y=0.0f;
	
	cdia_host[h].x=1.0f;
	cdia_host[h].y=0.0f;
	
	ldia_host[h].x=betha;
	ldia_host[h].y=0.0f;
	
      }
      
      
      int h_0=i*NZ*NY+k*NY;	
      
      ldia_host[h_0+0].x=0.0f;
      udia_host[h_0+NY-1].x=0.0f;	
      
      
      int j=0;
      
      float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      float b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      

      float alpha=(a+b)/(2.0f*a-b);
      
      udia_host[h_0].x=alpha;
      cdia_host[h_0].x=1.0f;
      
      j=NY-1;
      
      a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
      
      
      alpha=(a+b)/(2.0f*a-b);			
      
      ldia_host[h_0+NY-1].x=alpha;
      cdia_host[h_0+NY-1].x=1.0f;
      
    }
  }
  
  //Copy to whatever
  
  cudaCheck(cudaMalloc(&udiag_yy,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&ldiag_yy,SIZE),domain,"C");
  cudaCheck(cudaMalloc(&cdiag_yy,SIZE),domain,"C");
  
  cudaCheck(cudaMemcpy(udiag_yy,udia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");
  cudaCheck(cudaMemcpy(cdiag_yy,cdia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");
  cudaCheck(cudaMemcpy(ldiag_yy,ldia_host,SIZE,cudaMemcpyHostToDevice),domain,"C");
  
  //Free
  
  free(udia_host);
  free(cdia_host);
  free(ldia_host);		
  
  //Set work groups
  
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;
  
  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE;					
  
  return;
  
  
}



extern void deriv_Y_HO(float2* u, domain_t domain){
  
  deriv_Y_kernel<<<blocksPerGrid,threadsPerBlock>>>(u);
  kernelCheck(RET,domain,"W_kernel");
  
  //Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))

  cusparseCheck(cusparseCgtsvStridedBatch(derivatives_handle,NY,ldiag_y,cdiag_y,
					  udiag_y,u,NXSIZE*NZ,NY),domain,"HEM");
			
	return;
}



extern void deriv_YY_HO(float2* u, domain_t domain){

  /*
    deriv_YY_kernel<<<blocksPerGrid,threadsPerBlock>>>(u);
    kernelCheck(RET,"Boundary");		
    
    //Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))
    
    cusparseCheck(cusparseCgtsvStridedBatch(derivatives_handle,NY,ldiag_yy,cdiag_yy,udiag_yy,u,NX*NZ,NY),"HEM");
  */	
  
  
  deriv_Y_HO(u, domain);
  deriv_Y_HO(u, domain);
  
  return;

}



