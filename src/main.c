#include "channel.h"

int RANK;
int IGLOBAL;

//Buffer for derivatives

double2* LDIAG;
double2* UDIAG;
double2* CDIAG;
double2* AUX;

int main(int argc, char** argv)
{ 


	MPI_Init(NULL,NULL);	
	H5open();

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
 	MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

	if(size!=MPISIZE){
	printf("Error número de procesos debe ser: %d",MPISIZE);
	exit(1);
	}

	//Local id
	IGLOBAL=NXSIZE*RANK;

	cudaDeviceProp prop; 

	//cudaCheck(cudaGetDeviceProperties(&prop,0),"prop");
	//if(RANK==0)	
	//printf("\nMaxthreadperN=%d",prop.maxThreadsPerBlock);
    

	// Set up cuda device
	cudaCheck(cudaSetDevice(RANK),"Set");		

	//Set the whole damn thing up
	setUp();

	if(RANK==0){
	checkDerivatives();	
	}

	if(RANK==0){
	setRKmean();
	}
	
	

	//Allocate initial memory

	float2* ddv;
	float2* g;

	//Two buffers allocated

	cudaCheck(cudaMalloc(&ddv,SIZE),"malloc");
	cudaCheck(cudaMalloc(&g,SIZE),"malloc");

	//Read data

	readData(ddv,g);

	if(RANK==0){
	readU();}

	RKstep(ddv,g,0);

	//Write data

	writeData(ddv,g);
	
	if(RANK==0){
	writeU();}

	H5close();
	MPI_Finalize();
	

return 0;
}
