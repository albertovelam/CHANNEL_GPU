#include"channel.h"


static __global__ void rhs_A_kernel(float2* u)

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



static float2* udiag_host;
static float2* cdiag_host;
static float2* ldiag_host;

static float2* udiag;
static float2* cdiag;
static float2* ldiag;


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

static cusparseHandle_t hemholzt_handle;

extern void setHemholzt(void){


		//Use 3 more buffers for speed

		cusparseCheck(cusparseCreate(&hemholzt_handle),"Handle");

		float2* udia_host=(float2*)malloc(SIZE);
		float2* cdia_host=(float2*)malloc(SIZE);		
		float2* ldia_host=(float2*)malloc(SIZE);
	

		for(int i=0;i<NXSIZE;i++){

			float kx=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;
			//kx=0;
			for(int k=0;k<NZ;k++){

				float kz=k;
			//	kz=0;
				
				
					//Fraction
					kx=(PI2/LX)*kx;
					kz=(PI2/LZ)*kz;	
	
					float kk=kz*kz+kx*kx;
				
				for(int j=0;j<NY;j++){

				
				//COEFICIENTES NON_UNIFORM GRID
				// alpha f''_{j+1}+f''_{j}+betha f''_{j-1}=A f_{j+1}+E f_{j}+B f_{j-1}

				float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
				float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
	

				float A=-12.0f*b/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
				float B= 12.0f*a/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
				float C=-A-B;
		
				float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
				float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);

					int h=i*NZ*NY+k*NY+j;
	
					udia_host[h].x=A-kk*alpha;
					udia_host[h].y=0.0f;

					cdia_host[h].x=C-kk;
					cdia_host[h].y=0.0f;

					ldia_host[h].x=B-kk*betha;
					ldia_host[h].y=0.0f;

			
				}
			
					//Boundary conditions set to zero 			

					udia_host[i*NZ*NY+k*NY].x=0.0f;
					udia_host[i*NZ*NY+k*NY+NY-1].x=0.0f;
					udia_host[i*NZ*NY+k*NY+NY-2].x=0.0f;
			
					cdia_host[i*NZ*NY+k*NY].x=1.0f;
					cdia_host[i*NZ*NY+k*NY+NY-1].x=1.0f;
	
					ldia_host[i*NZ*NY+k*NY].x=0.0f;
					ldia_host[i*NZ*NY+k*NY+1].x=0.0f;
					ldia_host[i*NZ*NY+k*NY+NY-1].x=0.0f;
				
				
			}
		}
										

		//Malloc memory: 3 extra buffers

		cudaCheck(cudaMalloc(&udiag,SIZE),"C");
		cudaCheck(cudaMalloc(&ldiag,SIZE),"C");
		cudaCheck(cudaMalloc(&cdiag,SIZE),"C");

		cudaCheck(cudaMemcpy(udiag,udia_host,SIZE,cudaMemcpyHostToDevice),"C");
		cudaCheck(cudaMemcpy(cdiag,cdia_host,SIZE,cudaMemcpyHostToDevice),"C");
		cudaCheck(cudaMemcpy(ldiag,ldia_host,SIZE,cudaMemcpyHostToDevice),"C");

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



extern void hemholztSolver(float2* u){

	rhs_A_kernel<<<blocksPerGrid,threadsPerBlock>>>(u);
	kernelCheck(RET,"hemholz");	

	cusparseCheck(cusparseCgtsvStridedBatch(hemholzt_handle,NY,ldiag,cdiag,udiag,u,NXSIZE*NZ,NY),"HEM");



	return;
}




