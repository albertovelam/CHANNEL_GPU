
#include"channel.h"
#include<cublas_v2.h>

static float2 alpha[1];
static cublasHandle_t cublasHandle;

static float2* aux_host1;
static float2* aux_host2;

static size_t size;
static int MPIErr;


void setTransposeCudaMpi(domain_t domain){


  cublasCheck(cublasCreate(&cublasHandle),domain,"Cre_Transpose");
  
  alpha[0].x=1.0f;
  alpha[0].y=0.0f;
  
  size=NXSIZE*NY*NZ*sizeof(float2);
  
  aux_host1=(float2*)malloc(size);
  aux_host2=(float2*)malloc(size);	
  
  return;
}

//Transpose [Ny,Nx] a [Nx,Ny]

void transpose(float2* u_2,const float2* u_1,int Nx,int Ny, domain_t domain){

	//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
  
  cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1,Nx,0,0,Nx,(float2*)u_2,Ny),domain,"Tr_1");
  //printf("\n%f,%f",alpha[0].x,alpha[0].y);
  return;
  

}

//Transpose [Ny,Nx] a [Nx,Ny]

void transposeBatched(float2* u_2,const float2* u_1,int Nx,int Ny,int batch, domain_t domain){

  //Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
  
  for(int nstep=0;nstep<batch;nstep++){
    
    int stride=nstep*Nx*Ny;
    
    cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1+stride,Nx,0,0,Nx,(float2*)u_2+stride,Ny),domain,"Tr_2");
    
  }	
  
	//printf("\n%f,%f",alpha[0].x,alpha[0].y);
  return;

  
}

void transposeXYZ2YZX(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain){
  
  int myNx = Nx/sizeMpi;
  int myNy = Ny/sizeMpi;
  
  //Transpose [NXISZE,NY,NZ] ---> [NY,myNx,NZ]
  
  transposeBatched((float2*)AUX,(const float2*)u1,Nz,NY,myNx,domain);
  transpose(u1,(const float2*)AUX,NY,myNx*Nz,domain);
  
  //COPY TO HOST
  cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),domain,"copy");
  
	
  MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);
  
  mpiCheck(MPIErr,"transpoze");
  
  //COPY TO DEVICE
  cudaCheck(cudaMemcpy((float2*)AUX,(float2*)aux_host2,size,cudaMemcpyHostToDevice),domain,"copy");

  //Transpose [sizeMpi,myNy,myNx,Nz] ---> [myNy,Nz,sizeMpi,myNx]
  
  transposeBatched(u1,(const float2*)AUX,myNx*Nz,myNy,sizeMpi,domain);
  transposeBatched((float2*)AUX,(const float2*)u1,myNy,Nz,sizeMpi*myNx,domain);
  transpose(u1,(const float2*)AUX,myNy*Nz,sizeMpi*myNx,domain);
	
}	
		
void transposeYZX2XYZ(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain){

  int myNx = Nx/sizeMpi;
  int myNy = Ny/sizeMpi;
  
  
  //Transpose [myNy,Nz,sizeMpi,myNx] ---> [sizeMpi,NYISZE,myNx,Nz]

  transpose((float2*)AUX,(const float2*)u1,sizeMpi*myNx,myNy*Nz,domain);
  transposeBatched(u1,(const float2*)AUX,myNy*Nz,myNx,sizeMpi,domain);
  transposeBatched((float2*)AUX,(const float2*)u1,myNx,Nz,sizeMpi*myNy,domain);
	
  //COPY TO HOST
  cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)AUX,size,cudaMemcpyDeviceToHost),domain,"copy");
	
  /* Communications */
  MPIErr = MPI_Alltoall(aux_host1,Nz*myNx*myNy,MPI_DOUBLE,
			aux_host2,Nz*myNx*myNy,MPI_DOUBLE,
			MPI_COMM_WORLD);
  
  mpiCheck(MPIErr,"transpoze");	

  //COPY TO DEVICE
  cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),domain,"copy");
	
  
  //Transpose [NY,myNx,Nz]--->[NXISZE,NY,Nz]  
  
  transpose((float2*)AUX,(const float2*)u1,myNx*Nz,NY,domain);
  transposeBatched(u1,(const float2*)AUX,NY,Nz,myNx,domain);
  
  return;
	
}




