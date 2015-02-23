
#include"channel.h"

static cublasHandle_t cublasHandle;
static float2 alpha[1];
static float2 betha[1];
static float2* B;


void setTranspose(domain_t domain){
  cublasCheck(cublasCreate(&cublasHandle),domain,"Cre");
  
  alpha[0].x=1.0f;
  alpha[0].y=0.0f;
  
  return;
}

void transpose_A(float2* u_2,float2* u_1, domain_t domain){
START_RANGE("transpose_A",23)
  //Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
  
  cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NX*NZ,NY,alpha,(const float2*)u_1,NY,0,0,NY,u_2,NX*NZ),domain,"Tr");
  //printf("\n%f,%f",alpha[0].x,alpha[0].y);
END_RANGE
  return;
   
}

void transpose_B(float2* u_2,float2* u_1, domain_t domain){
START_RANGE("transpose_B",24)
  //Transpuesta de [j,i,k][NY,NX,NZ] a -----> [i,k,j][NX,NZ,NY]
  
  cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NY,NX*NZ,alpha,(const float2*)u_1,NX*NZ,0,0,NX*NZ,u_2,NY),domain,"Tr");
  //printf("\n%f,%f",alpha[0].x,alpha[0].y);
END_RANGE
  return;

}
