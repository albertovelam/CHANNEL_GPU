#include "channel.h"

void readData(float2* ddv,float2* g){

	float* host_buffer=(float*)malloc(SIZE);

	//read u

	mpiCheck(read_parallel_float("/gpfs/projects/upm79/channel_950/G3.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	
	cudaCheck(cudaMemcpy(g,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");
	
	mpiCheck(read_parallel_float("/gpfs/projects/upm79/channel_950/DDV3.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	
	cudaCheck(cudaMemcpy(ddv,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");

	free(host_buffer);

	return;
}


void writeData(float2* ddv,float2* g){

	float* host_buffer=(float*)malloc(SIZE);

	//read u

		
	cudaCheck(cudaMemcpy(host_buffer,g,SIZE,cudaMemcpyDeviceToHost),"MemInfo_uy");
	mpiCheck(wrte_parallel_float("/gpfs/projects/upm79/channel_950/G3.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");

	cudaCheck(cudaMemcpy(host_buffer,ddv,SIZE,cudaMemcpyDeviceToHost),"MemInfo_uy");
	mpiCheck(wrte_parallel_float("/gpfs/projects/upm79/channel_950/DDV3.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	

	free(host_buffer);

	return;

}

void genRandData(float2* ddv,float2* g,float F){

	int NM=1000;
	
	srand(time(NULL));

	float2* host_buffer=(float2*)malloc(SIZE);
	
	for(int i=0;i<NXSIZE*NZ*NY;i++){
	host_buffer[i].x=0.5f*F*((rand()%NM)/NM-1.0f);
	host_buffer[i].y=0.5f*F*((rand()%NM)/NM-1.0f);
	}
	
	cudaCheck(cudaMemcpy(g,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");
	
	for(int i=0;i<NXSIZE*NZ*NY;i++){
	host_buffer[i].x=0.5f*F*((rand()%NM)/NM-1.0f);
	host_buffer[i].y=0.5f*F*((rand()%NM)/NM-1.0f);
	}
	
	cudaCheck(cudaMemcpy(ddv,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");

	return;

}


