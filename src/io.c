#include "channel.h"

void readData(float2* ddv, float2* g, domain_t domain){

  float* host_buffer=(float*)malloc(domain.size);
  
  //read u
  
  mpiCheck(read_parallel_float("/drive1/guillem/Channel/G.h5",
			       (float *)host_buffer,
			       domain.nx,
			       domain.ny,
			       2*domain.nz,
			       domain.rank,
			       domain.size),"read");	
  cudaCheck(cudaMemcpy(g,host_buffer,SIZE,cudaMemcpyHostToDevice),domain,"MemInfo_uy");
	
  mpiCheck(read_parallel_float("/drive1/guillem/Channel/ddV.h5",
			       (float *)host_buffer,
			       domain.nx,
			       domain.ny,
			       2*domain.nz,
			       domain.rank,
			       domain.size),"read");	
  cudaCheck(cudaMemcpy(ddv,
		       host_buffer,
		       domain.size,
		       cudaMemcpyHostToDevice),domain,"MemInfo_uy");
  
  free(host_buffer);
  
  return;
}


void writeData(float2* ddv,float2* g, domain_t domain){
  
  float* host_buffer=(float*)malloc(domain.size);
  
  //read u
  
		
  cudaCheck(cudaMemcpy(host_buffer,
		       g,
		       domain.size,
		       cudaMemcpyDeviceToHost),domain,"MemInfo_uy");
  mpiCheck(wrte_parallel_float("/drive1/guillem/Channel/G.01.h5",
			       (float *)host_buffer,
			       domain.nx,
			       domain.ny,
			       2*domain.nz,
			       domain.rank,
			       domain.size),"read");

  cudaCheck(cudaMemcpy(host_buffer,
		       ddv,
		       domain.size,
		       cudaMemcpyDeviceToHost),domain,"MemInfo_uy");
  mpiCheck(wrte_parallel_float("/drive1/guillem/Channel/ddV.01.h5",
			       (float *)host_buffer,
			       domain.nx,
			       domain.ny,
			       2*domain.nz,
			       domain.rank,
			       domain.size),"read");	
  
  free(host_buffer);
	
	return;

}

void genRandData(float2* ddv, float2* g, float F, domain_t domain){

	int NM=1000;
	
	srand(time(NULL));

	float2* host_buffer=(float2*)malloc(SIZE);
	
	for(int i=0;i<NXSIZE*NZ*NY;i++){
	host_buffer[i].x=0.5f*F*((rand()%NM)/NM-1.0f);
	host_buffer[i].y=0.5f*F*((rand()%NM)/NM-1.0f);
	}
	
	cudaCheck(cudaMemcpy(g,host_buffer,SIZE,cudaMemcpyHostToDevice),domain,"MemInfo_uy");
	
	for(int i=0;i<NXSIZE*NZ*NY;i++){
	host_buffer[i].x=0.5f*F*((rand()%NM)/NM-1.0f);
	host_buffer[i].y=0.5f*F*((rand()%NM)/NM-1.0f);
	}
	
	cudaCheck(cudaMemcpy(ddv,host_buffer,SIZE,cudaMemcpyHostToDevice),domain,"MemInfo_uy");
	
	return;
	
}


