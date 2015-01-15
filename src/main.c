#include "channel.h"

//Buffer for derivatives

double2* LDIAG;
double2* UDIAG;
double2* CDIAG;
double2* AUX;

int main(int argc, char** argv)
{ 	
  int rank;
  int i;
  int iglobal;
  int size;
  cudaDeviceProp prop;
  int Ndevices;
  float2* ddv;
  float2* g;
  domain_t domain = {0, 0, 0, 0, 0, 0};
  char *ginput, *goutput, *ddvinput, *ddvoutput;
  
  MPI_Init(&argc, &argv);	
  H5open();
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  // Initial configuration
  if(rank == 0){
    config_t config = read_config_file("run.conf");
    printf("Reading configuration\n");
    read_domain_from_config(&domain,&config);
    printf("Reading input file names\n");
    ginput = (char *) config_setting_get_string(config_lookup(&config, "application.input.G"));
    printf("Input G file: %s\n",ginput);
    goutput = (char *) config_setting_get_string(config_lookup(&config, "application.output.G"));
    printf("Output G file: %s\n",goutput);
    ddvinput = (char *) config_setting_get_string(config_lookup(&config, "application.input.DDV"));
    printf("Input DDV file: %s\n",ddvinput);
    ddvoutput = (char *) config_setting_get_string(config_lookup(&config, "application.output.DDV"));
    printf("Output DDV file: %s\n\n",ddvoutput);
  }
  
  MPI_Bcast(&(domain.nx), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.ny), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.nz), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  domain.rank = rank;
  domain.size = size;
  domain.iglobal = domain.nx * domain.rank / domain.size;
  
  for (i=0; i<domain.size; i++){
    if (i == domain.rank){
      if (i == 0) printf("rank, size, iglobal, nx, ny ,nz\n");
      if (i == 0) printf("===============================\n");
      MPI_Barrier(MPI_COMM_WORLD);
      printf("%d, %d, %d, %d, %d, %d\n",
	     domain.rank, domain.size, domain.iglobal,
	     domain.nx, domain.ny, domain.nz);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  
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
  if(rank == 0) printf("Reading Data...\n");

  readData(ddv,g,ddvinput,ginput,domain);
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
  
  writeData(ddv,g,ddvoutput,goutput,domain);
  
  if(rank==0){
    writeU();
    //config_destroy(&config);
  }

  H5close();
  MPI_Finalize();
  
  return 0;
}



