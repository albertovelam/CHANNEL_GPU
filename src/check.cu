#include"channel.h"

void kernelCheck( cudaError_t error, domain_t domain, const char* function)
{


	error= cudaGetLastError();			
		if(error !=cudaSuccess)
		{
			const char* error_string= cudaGetErrorString(error);
			printf("\n error  %s : %s domain.rank=%d \n", function, error_string,domain.rank);
			exit(1);
		}

	return;
}

extern void cufftCheck( cufftResult error, domain_t domain,  const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d domain.rank=%d \n", function, error,domain.rank);
		exit(1);
	}
		
	return;
}  
#ifdef USE_CUSPARSE
extern void cusparseCheck( cusparseStatus_t error, domain_t domain,  const char* function )
{
	if(error != CUSPARSE_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d domain.rank=%d \n", function, error,domain.rank);
		exit(1);
	}
		
	return;
}  
#endif
extern void cublasCheck(cublasStatus_t error, domain_t domain, const char* function )
{
	if(error !=  CUBLAS_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d domain.rank=%d \n", function, error,domain.rank);
		exit(1);
	}
		
	return;
}  


extern void cudaCheck( cudaError_t error, domain_t domain, const char* function)
{
	if(error !=cudaSuccess)
	{
		const char* error_string= cudaGetErrorString(error);
		printf("\n error  %s : %s domain.rank=%d \n", function, error_string,domain.rank);
		exit(1);
	}
		

	return;
}



extern void mpiCheck( int error, const char* function)
{
	if(error !=0)
	{
		//printf("\n error_MPI %s \n",(char*)function);
		printf("error_mpi");		
		exit(1);
	}
		
	

	return;
}



