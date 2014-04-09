#include "channel.h"

void readData(float2* ddv,float2* g){

	float* host_buffer=(float*)malloc(SIZE);

	//read u

	mpiCheck(read_parallel_float("./data_inicial/G.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	
	cudaCheck(cudaMemcpy(g,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");
	
	mpiCheck(read_parallel_float("./data_inicial/ddV.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	
	cudaCheck(cudaMemcpy(ddv,host_buffer,SIZE,cudaMemcpyHostToDevice),"MemInfo_uy");

	free(host_buffer);

	return;
}


void writeData(float2* ddv,float2* g){

	float* host_buffer=(float*)malloc(SIZE);

	//read u

		
	cudaCheck(cudaMemcpy(host_buffer,g,SIZE,cudaMemcpyDeviceToHost),"MemInfo_uy");
	mpiCheck(wrte_parallel_float("./data_inicial/G.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");

	cudaCheck(cudaMemcpy(host_buffer,ddv,SIZE,cudaMemcpyDeviceToHost),"MemInfo_uy");
	mpiCheck(wrte_parallel_float("./data_inicial/ddV.h5",(float *)host_buffer,NX,NY,2*NZ,RANK,MPISIZE),"read");	

	free(host_buffer);

	return;

}



