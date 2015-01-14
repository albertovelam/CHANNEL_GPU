#include "channel.h"

static cufftHandle fft2_r2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft2_sum; 

static cublasHandle_t cublasHandle;


float2* u_host_1;
float2* u_host_2;

void fftSetup(domain_t domain)
{
  cublasCheck(cublasCreate(&cublasHandle),domain,"Cre_fft");
	
  int n[2]={NX,2*NZ-2};
  int nsum[2]={NXSIZE,2*NZ-2};

  cufftCheck(cufftPlanMany( &fft2_r2c,2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,NYSIZE),domain,"ALLOCATE_FFT3_R2C");
  cufftCheck(cufftPlanMany( &fft2_c2r,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,NYSIZE),domain,"ALLOCATE_FFT3_C2R");

  cufftCheck(cufftPlanMany( &fft2_sum,2,nsum,NULL,1,0,NULL,1,0,CUFFT_R2C,NY),domain,"ALLOCATE_FFT3_C2R");

  u_host_1=(float2*)malloc(SIZE);
  u_host_2=(float2*)malloc(SIZE);

  return;
}

void fftDestroy(void)
{
  cufftDestroy(fft2_r2c);
  cufftDestroy(fft2_c2r);
  
  return;
}

void fftForward(float2* buffer, domain_t domain)
{
  cufftCheck(cufftExecR2C(fft2_r2c,(float*)buffer,(float2*)buffer),domain,"forward transform");
	
  return;
}

void fftBackward(float2* buffer, domain_t domain)
{
  cufftCheck(cufftExecC2R(fft2_c2r,(float2*)buffer,(float*)buffer),domain,"forward transform");
		
  return;	

}

void fftBackwardTranspose(float2* u2, domain_t domain){

	/*
	//Copy to host
	cudaCheck(cudaMemcpy(u_host_1,u2,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
	
	//Tranpose (x,z,y)-->(y,x,z)
	mpiCheck(chyzx2xyz((double*)u_host_1,(double*)u_host_2,NY,NX,NZ,RANK,MPISIZE),"Tr");

	//Copy to device
	cudaCheck(cudaMemcpy(u2,u_host_2,SIZE,cudaMemcpyHostToDevice),"MemInfo1");
	*/

  transposeYZX2XYZ(u2,
		   domain.ny,
		   domain.nx,
		   domain.nz,
		   domain.rank,
		   domain.size, domain);	

  fftBackward(u2,domain);
		
  return;

}

void fftForwardTranspose(float2* u2, domain_t domain){
  
  
  fftForward(u2,domain);	
	
  transposeXYZ2YZX(u2,
		   domain.ny,
		   domain.nx,
		   domain.nz,
		   domain.rank,
		   domain.size,domain);	
  
  /*
  //Copy to host
  cudaCheck(cudaMemcpy(u_host_1,u2,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
  
  //Tranpose (y,x,z)-->(x,z,y)
  mpiCheck(chxyz2yzx((double *)u_host_1,(double *)u_host_2,NY,NX,NZ,RANK,MPISIZE),"Tr");
  
  //Copy to device
  cudaCheck(cudaMemcpy(u2,u_host_2,SIZE,cudaMemcpyHostToDevice),"MemInfo1");
  */
  
  return;
  
}

/*
  void forwardTranspose(float2* u2){
  
  //Traspuesta de COMPLEJA
  
  //cufftCheck(cufftExecR2C(fft2_r2c,(float*)u2,(float2*)aux),"forward transform");
  
  
  //fftForward(u2);	
	
  //Transpuesta de [j,i,k][NY,NX,NZ] a -----> [i,k,j][NX,NZ,NY]
  transpose_B(aux,u2);
  cudaCheck(cudaMemcpy(u2,aux,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
  
  return;
  
  }
  
void backwardTranspose(float2* u2){

//Traspuesta de COMPLEJA

//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
transpose_A(aux,u2);


cudaCheck(cudaMemcpy(u2,aux,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
//fftBackward(u2);

//cufftCheck(cufftExecC2R(fft2_c2r,(float2*)aux,(float*)u2),"forward transform");

return;

}
*/

void calcUmax(float2* u_x,float2* u_y,float2* u_z,float* ux,float* uy,float* uz, domain_t domain)
{


	int size_l=2*NXSIZE*NY*NZ;
	int index;

		
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)u_x,1,&index),domain,"Isa");
	cudaCheck(cudaMemcpy(ux,(float*)u_x+index-1, sizeof(float), cudaMemcpyDeviceToHost),domain,"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)u_y,1,&index),domain,"Isa");
	cudaCheck(cudaMemcpy(uy,(float*)u_y+index-1, sizeof(float), cudaMemcpyDeviceToHost),domain,"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)u_z,1,&index),domain,"Isa");
	cudaCheck(cudaMemcpy(uz,(float*)u_z+index-1, sizeof(float), cudaMemcpyDeviceToHost),domain,"MemInfo_isa");
	
	*ux=fabs(*ux);
	*uy=fabs(*uy);
	*uz=fabs(*uz);
	
	//MPI reduce
	reduceMAX(ux,uy,uz);

	return;

}

void calcDmax(float2* u_x,float2* u_y,float* ux,float* uy,domain_t domain)
{


	int size_l=2*NXSIZE*NY*NZ;
	int index;

		
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)u_x,1,&index),domain,"Isa");
	cudaCheck(cudaMemcpy(ux,(float*)u_x+index-1, sizeof(float), cudaMemcpyDeviceToHost),domain,"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)u_y,1,&index),domain,"Isa");
	cudaCheck(cudaMemcpy(uy,(float*)u_y+index-1, sizeof(float), cudaMemcpyDeviceToHost),domain,"MemInfo_isa");
	
	
	*ux=fabs(*ux);
	*uy=fabs(*uy);


	float* uz=(float*)malloc(sizeof(float));
	*uz=0;
	
	//MPI reduce
	reduceMAX(ux,uy,uz);

	free(uz);		

	return;

}



float sumElementsReal(float2* buffer_1, domain_t domain){

	//destroza lo que haya en el buffer

	float2 sum[NYSIZE];
	float sum_all=0;
	
	//Transformada NYSIZE*[NX*NZ]
	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1),buffer_1),domain,"forward transform");

	

	for(int i=0;i<NYSIZE;i++){

	  cudaCheck(cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NX*NZ,sizeof(float2),cudaMemcpyDeviceToHost),
		    domain,
		    "MemInfo1");

	};
	

	for(int i=1;i<NYSIZE;i++){

	sum[0].x+=sum[i].x;
	}

	//MPI SUM

	reduceSUM((float*)sum,&sum_all);


	return sum_all;

}

void sumElementsComplex(float2* buffer_1,float* out, domain_t domain){

	//destroza lo que haya en el buffer

	float sum_all=0;
	float2 sum[NY];
	float2 sum1[NY];
	float2 sum2[NY];

	//Transpose [NXSIZE,NZ,NY] to [NY,NXSIZE,NZ]	
	transpose((float2*)AUX,(const float2*)buffer_1,NY,NXSIZE*NZ, domain);

	//Transformada NYSIZE*[NX*NZ]
	cufftCheck(cufftExecR2C(fft2_sum,(float*)(AUX),(float2*)AUX),domain,"forward transform");

	//Transpose [NXSIZE,NZ,NY] to [NY,NXSIZE,NZ]	
	transpose((float2*)buffer_1,(const float2*)AUX,NXSIZE*NZ,NY,domain);	

	cudaCheck(cudaMemcpy((float2*)sum,(float2*)buffer_1,NY*sizeof(float2),cudaMemcpyDeviceToHost),domain,"MemInfo1");


 	mpiCheck(MPI_Allreduce((float*)sum,(float*)sum2,2*NY,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),"caca");	

	for(int i=0;i<NY;i++){
	out[i]=sum2[i].x;
	}


	return;

}
