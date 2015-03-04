#include"channel.h"

char host_name[MPI_MAX_PROCESSOR_NAME];
char mybus[16];
int SMCOUNT;
MPI_Request *send_requests;
MPI_Request *recv_requests;
MPI_Status *send_status;
MPI_Status *recv_status;
extern float2* ddv;
extern float2* g;

#include <unistd.h>
#include <cstring>

int stringCmp( const void *a, const void *b)
{
     return strcmp((const char*)a,(const char*)b);
}

void setUp(domain_t domain){

	//Set up
   int rank=0, local_rank=0, deviceCount;
   char (*host_names)[MPI_MAX_PROCESSOR_NAME];
   MPI_Comm nodeComm;
   int n, namelen, color, nprocs, local_procs;
   size_t bytes;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Get_processor_name(host_name,&namelen);
   bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
   host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
   strcpy(host_names[rank], host_name);
   for (n=0; n<nprocs; n++)
   {
      MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
   }
   qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
   color = 0;
   for (n=0; n<nprocs; n++)
   {
      if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
      if(strcmp(host_name, host_names[n]) == 0) break;
   }
   MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
   MPI_Comm_rank(nodeComm, &local_rank);
   MPI_Comm_size(nodeComm, &local_procs);
   free(host_names);
   CHECK_CUDART( cudaGetDeviceCount(&deviceCount) );
   CHECK_CUDART( cudaSetDevice(local_rank%deviceCount) );
   int dev;
   struct cudaDeviceProp deviceProp;
   CHECK_CUDART( cudaGetDevice(&dev) );
   CHECK_CUDART( cudaGetDeviceProperties(&deviceProp, dev) );
   sprintf(&mybus[0], "0000:%02x:%02x.0", deviceProp.pciBusID, deviceProp.pciDeviceID);

   SMCOUNT = deviceProp.multiProcessorCount;
   MPI_Barrier(MPI_COMM_WORLD);
   if(rank==0){
     printf("\n# Running on '%s'\n", deviceProp.name);
     printf("# SMs = %d\n", deviceProp.multiProcessorCount);
     printf("# clock = %d\n\n", deviceProp.clockRate);
     printf("===================================================================\n");
     printf("host\tdevice\t\trank\tsize\tiglobal\tnx\tny\tnz\n");
     printf("===================================================================\n");
     fflush(stdout);
   }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i=0; i<domain.size; i++){
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == domain.rank){
      printf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n",
             host_name,mybus,domain.rank, domain.size, domain.iglobal,
             domain.nx, domain.ny, domain.nz);
      fflush(stdout);
    }
    usleep(1000);
    MPI_Barrier(MPI_COMM_WORLD);
  }


//   printf("rank: %d host: %s device: %s IGLOBAL: %d\n",RANK,host_name,mybus,IGLOBAL);
//   fflush(stdout);
/*   SMCOUNT = deviceProp.multiProcessorCount;
   MPI_Barrier(MPI_COMM_WORLD);
   if(rank==0){
     printf("\n# Running on '%s'\n", deviceProp.name);
     printf("# SMs = %d\n", deviceProp.multiProcessorCount);
     printf("# clock = %d\n", deviceProp.clockRate);
   }*/

  send_requests = (MPI_Request*)malloc(MPISIZE*sizeof(MPI_Request));
  recv_requests = (MPI_Request*)malloc(MPISIZE*sizeof(MPI_Request));
  send_status = (MPI_Status*)malloc(MPISIZE*sizeof(MPI_Status));
  recv_status = (MPI_Status*)malloc(MPISIZE*sizeof(MPI_Status));

size_t free,tot,used,dused;
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; used=tot-free; 
  printf("start\tfree: %lld MB \ttot: %lld MB \tused: %lld MB\n",free,tot,used);
}
  CHECK_CUDART( cudaMalloc((void**)&ddv, SIZE) );
  CHECK_CUDART( cudaMalloc((void**)&g  , SIZE) );
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("ddv+g\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}

	fftSetup(domain);

if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("fft\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
/*
	setDerivatives_HO(domain);

if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("derH0\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
*/
	//setHemholzt();
	//setImplicit();
	setRK3(domain);
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("RK3\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
    
	setTransposeCudaMpi(domain);
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("Trans\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
    	
	setDerivativesDouble(domain);
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("DerDbl\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
    
	//setHemholztDouble(domain);
	setImplicitDouble(domain);
if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("Impl\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}
    
//#ifdef USE_CUSPARSE
	CHECK_CUDART( cudaMalloc(&LDIAG,SIZE_AUX) );
	CHECK_CUDART( cudaMalloc(&CDIAG,SIZE_AUX) );
	CHECK_CUDART( cudaMalloc(&UDIAG,SIZE_AUX) );
//#endif
	//AUX FOR TRANPOSE
	//cudaCheck(cudaMalloc(&AUX,SIZE_AUX),domain,"malloc");
        AUX = (double2*)aux_dev[4];

if(domain.rank==0){
  CHECK_CUDART( cudaMemGetInfo(&free,&tot) );
  free=free>>20; tot=tot>>20; dused=tot-free-used; used=tot-free;
  printf("AUX\tfree: %lld MB \ttot: %lld MB \tused: %lld MB \tdelta: %11d MB\n",free,tot,used,dused);
}

    MPI_Barrier(MPI_COMM_WORLD);

    
//exit(1);
	return;

}
