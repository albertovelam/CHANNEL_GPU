#include "channel.h"
//Buffer for derivatives

double2* LDIAG;
double2* UDIAG;
double2* CDIAG;
double2* AUX;

int main(int argc, char** argv)
{ 	
  int rank;
  int iglobal;
  int size;
  cudaDeviceProp prop;
  int Ndevices;
  float2* ddv;
  float2* g;
  domain_t domain = {0, 0, 0, 0, 0, 0};
  
  MPI_Init(&argc, &argv);	
  H5open();
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  read_domain_from_config(&domain);
  domain.rank = rank;
  domain.size = size;
  domain.iglobal = domain.nx * domain.rank / domain.size;
  
  if(size != MPISIZE){
    printf("Error. The number of MPI processes is hardcoded in channel.h. Got%d, expecting %d\n",
	   size,
	   MPISIZE);
    exit(1);
  }
  
  cudaCheck(cudaGetDeviceCount(&Ndevices),domain,"device_count");
  printf("Ndevices=%d\n",Ndevices);
  
  //Local id
  iglobal=NXSIZE*rank;
  printf("(SIZE,RANK)=(%d,%d)\n",size,rank);
  
  cudaCheck(cudaGetDeviceProperties(&prop,0),domain,"prop");
  
  if(rank == 0){	
    printf("MaxthreadperN=%d\n",prop.maxThreadsPerBlock);
  }
  
  // Set up cuda device
  cudaCheck(cudaSetDevice(rank%2),domain,"Set");		
  
  //Set the whole damn thing up
  setUp(domain);
  
  if(rank==0){
    setRKmean();
  }

  if(rank == 0){	
    printf("Allocation...\n");
  }

  //Allocate initial memory
  //Two buffers allocated
  cudaCheck(cudaMalloc(&ddv,SIZE),domain,"malloc");
  cudaCheck(cudaMalloc(&g,SIZE),domain,"malloc");
  
  //Read data
  if(rank == 0){	
    printf("Reading...\n");
  }

  readData(ddv,g,domain);
  //scale(ddv,10.0f);scale(g,10.0f);
  //genRandData(ddv,g,(float)(NX*NZ));
  
  if(rank == 0){
    readU();
  }
  
  /*
    checkDerivatives();
    checkHemholzt();
    checkImplicit();
  */

  if(rank == 0){	
    printf("Starting RK iterations...\n");
  }

  RKstep(ddv,g,1,domain);
  
  //Write data
  
  writeData(ddv,g,domain);
  
  if(rank==0){
    writeU();
  }
  
  H5close();
  MPI_Finalize();
  
  return 0;
}


void read_domain_from_config(domain_t *domain){
  config_t config;
  config_init(&config);

  if ( !config_read_file(&config, "run.conf")){
    fprintf(stderr,  "%s:%d - %s\n", config_error_file(&config),
	    config_error_line(&config), config_error_text(&config));
    config_destroy(&config);
    return;
  }
  
  domain->nx = (int) config_setting_get_int(config_lookup(&config, "application.NX"));
  domain->ny = (int) config_setting_get_int(config_lookup(&config, "application.NY"));
  domain->nz = (int) config_setting_get_int(config_lookup(&config, "application.NZ"));

  config_destroy(&config);
  return;
}
