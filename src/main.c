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
	
	int Ndevices;
	cudaCheck(cudaGetDeviceCount(&Ndevices),"device_count");
	printf("\nNdevices=%d",Ndevices);

	//Local id
	IGLOBAL=NXSIZE*RANK;
	printf("\n(SIZE,RANK)=(%d,%d)",size,RANK);

	cudaDeviceProp prop; 

	cudaCheck(cudaGetDeviceProperties(&prop,0),"prop");
	if(RANK==0)	
	printf("\nMaxthreadperN=%d",prop.maxThreadsPerBlock);
    

	// Set up cuda device
	cudaCheck(cudaSetDevice(RANK%2),"Set");		

	//Set the whole damn thing up
	setUp();

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
	//scale(ddv,10.0f);scale(g,10.0f);
	
	//genRandData(ddv,g,(float)(NX*NZ));

	if(RANK==0){
	readU();
	}

	/*
	checkDerivatives();
	checkHemholzt();
	checkImplicit();
	*/
	RKstep(ddv,g,1);

	//Write data

	writeData(ddv,g);
	
	if(RANK==0){
	writeU();}

	H5close();
	MPI_Finalize();
	

return 0;
}
