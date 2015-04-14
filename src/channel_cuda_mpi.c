
#include"channel.h"
#include<cublas_v2.h>

static float2 alpha[1];
static cublasHandle_t cublasHandle;

static float2* aux_host1;
static float2* aux_host2;

static size_t size;
static int MPIErr;


void setTransposeCudaMpi(domain_t domain){


  CHECK_CUBLAS( cublasCreate(&cublasHandle) );
  
  alpha[0].x=1.0f;
  alpha[0].y=0.0f;
  
  size=NXSIZE*NY*NZ*sizeof(float2);
  
  aux_host1=(float2*)malloc(size);
  aux_host2=(float2*)malloc(size);	

  CHECK_CUDART( cudaHostRegister(aux_host1,size,0) );
  CHECK_CUDART( cudaHostRegister(aux_host2,size,0) );
  
  return;
}

//Transpose [Ny,Nx] a [Nx,Ny]

void transpose(float2* u_2,const float2* u_1,int Nx,int Ny, domain_t domain){
START_RANGE("transpose",25)
	//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
  
  CHECK_CUBLAS( cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1,Nx,0,0,Nx,(float2*)u_2,Ny) );
  //printf("\n%f,%f",alpha[0].x,alpha[0].y);
END_RANGE
  return;
  

}

//Transpose [Ny,Nx] a [Nx,Ny]

void transposeBatched(float2* u_2,const float2* u_1,int Nx,int Ny,int batch, domain_t domain){
START_RANGE("transposedBatched",26)
  //Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
  for(int nstep=0;nstep<batch;nstep++){
    int stride=nstep*Nx*Ny;
    CHECK_CUBLAS( cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1+stride,Nx,0,0,Nx,(float2*)u_2+stride,Ny) );
  }	
END_RANGE
	//printf("\n%f,%f",alpha[0].x,alpha[0].y);
  return;

  
}

void transposeXYZ2YZX(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain){
START_RANGE("transposeXYZ2YZX",27)  
  int myNx = Nx/sizeMpi;
  int myNy = Ny/sizeMpi;
  
  //Transpose [NXISZE,NY,NZ] ---> [NY,myNx,NZ]
/*  
  transposeBatched((float2*)AUX,(const float2*)u1,Nz,NY,myNx,domain);
  transpose(u1,(const float2*)AUX,NY,myNx*Nz,domain);
  
  //COPY TO HOST
  cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),domain,"copy");
*/
  trans_zxy_to_zyx(u1, (float2*)AUX, 0/*compute_stream*/,domain);
#ifdef USE_GPU_MPI
START_RANGE("MPI",1)
  MPIErr = MPI_Alltoall((float2*)AUX,Nz*myNx*myNy,MPI_DOUBLE,
                        u1,Nz*myNx*myNy,MPI_DOUBLE,
                        MPI_COMM_WORLD);
END_RANGE
  mpiCheck(MPIErr,"transpoze");

#else
  CHECK_CUDART( cudaMemcpy((float2*)aux_host1,(float2*)AUX,size,cudaMemcpyDeviceToHost) );
  
START_RANGE("MPI",1)	
  MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);
END_RANGE  
  mpiCheck(MPIErr,"transpoze");
  
  //COPY TO DEVICE
  CHECK_CUDART( cudaMemcpy((float2*)AUX,(float2*)aux_host2,size,cudaMemcpyHostToDevice) );
#endif
  //Transpose [sizeMpi,myNy,myNx,Nz] ---> [myNy,Nz,sizeMpi,myNx]
/*  
  transposeBatched(u1,(const float2*)AUX,myNx*Nz,myNy,sizeMpi,domain);
  transposeBatched((float2*)AUX,(const float2*)u1,myNy,Nz,sizeMpi*myNx,domain);
  transpose(u1,(const float2*)AUX,myNy*Nz,sizeMpi*myNx,domain);
*/
  trans_zyx_yblock_to_yzx((float2*)AUX, u1, 0/*compute_stream*/,domain);
END_RANGE	
}	
		
void transposeYZX2XYZ(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain){
START_RANGE("transposeYZX2XYZ",28)
  int myNx = Nx/sizeMpi;
  int myNy = Ny/sizeMpi;
  
  
  //Transpose [myNy,Nz,sizeMpi,myNx] ---> [sizeMpi,NYISZE,myNx,Nz]
/*
  transpose((float2*)AUX,(const float2*)u1,sizeMpi*myNx,myNy*Nz,domain);
  transposeBatched(u1,(const float2*)AUX,myNy*Nz,myNx,sizeMpi,domain);
  transposeBatched((float2*)AUX,(const float2*)u1,myNx,Nz,sizeMpi*myNy,domain);
*/
  trans_yzx_to_zyx_yblock(u1, (float2*)AUX, 0/*compute_stream*/, domain);	

  //COPY TO HOST
  CHECK_CUDART( cudaMemcpy((float2*)aux_host1,(float2*)AUX,size,cudaMemcpyDeviceToHost) );
START_RANGE("MPI",1)	
  /* Communications */
  MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);
END_RANGE  
  mpiCheck(MPIErr,"transpoze");	

/*
  //COPY TO DEVICE
  cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),domain,"copy");
	
  
  //Transpose [NY,myNx,Nz]--->[NXISZE,NY,Nz]  
  
  transpose((float2*)AUX,(const float2*)u1,myNx*Nz,NY,domain);
  transposeBatched(u1,(const float2*)AUX,NY,Nz,myNx,domain);
*/
  CHECK_CUDART(cudaMemcpy((float2*)AUX,(float2*)aux_host2,size,cudaMemcpyHostToDevice));
  trans_zyx_to_zxy((float2*)AUX, u1, 0/*compute_stream*/,domain);

END_RANGE  
  return;
	
}




