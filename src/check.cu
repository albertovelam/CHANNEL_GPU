#include"channel.h"

void kernelCheck( cudaError_t error, const char* function)
{


	error= cudaGetLastError();			
		if(error !=cudaSuccess)
		{
			const char* error_string= cudaGetErrorString(error);
			printf("\n error  %s : %s RANK=%d \n", function, error_string,RANK);
			exit(1);
		}

	return;
}

extern void cufftCheck( cufftResult error, const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d RANK=%d \n", function, error,RANK);
		exit(1);
	}
		
	return;
}  

extern void cusparseCheck( cusparseStatus_t error, const char* function )
{
	if(error != CUSPARSE_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d RANK=%d \n", function, error,RANK);
		exit(1);
	}
		
	return;
}  

extern void cublasCheck(cublasStatus_t error, const char* function )
{
	if(error !=  CUBLAS_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d RANK=%d \n", function, error,RANK);
		exit(1);
	}
		
	return;
}  


extern void cudaCheck( cudaError_t error, const char* function)
{
	if(error !=cudaSuccess)
	{
		const char* error_string= cudaGetErrorString(error);
		printf("\n error  %s : %s RANK=%d \n", function, error_string,RANK);
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



