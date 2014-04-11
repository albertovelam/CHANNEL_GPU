#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <math.h>

#include <stdlib.h>

#include <hdf5.h>
#include <hdf5_hl.h>

//MESHGRID


#define Fmesh(X)  (tanh(2.0*(X))/tanh(2.0))

const int NX=128;
const int NY=128;
const int NZ=128/2+1;

//Dimensions in Y direction 
// h=1.0f channel height 2.0f

const float LY=2.0f;
const float DELTA_Y=2.0f/(NY-1);

const float PI2=2.0f*3.14159265f;


const float LX=1.0f*PI2;
const float LZ=0.5f*PI2;

//Reynolds number and bulk velocity

const float REYNOLDS=5.6e3;
const float QVELOCITY=1.0f;

static cublasHandle_t   CUBLAS_HANDLE; 

static cudaError_t RET;
static const int THREADSPERBLOCK_IN=16;

//MPI number of process

const int MPISIZE=2;

const int NXSIZE=NX/MPISIZE;
const int NYSIZE=NY/MPISIZE;

extern int RANK;
extern int IGLOBAL;

//size local
const int SIZE=NXSIZE*NY*NZ*sizeof(float2);

//BUFFERS FOR DERIVATIVES

extern double2* LDIAG;
extern double2* UDIAG;
extern double2* CDIAG;

extern double2* AUX;


const int NSTEPS=2;
const int SIZE_AUX=2*SIZE/NSTEPS;

//CUSPARSE HANDELS


///////////////////////////C///////////////////////////

//Set up

void setUp(void);

//fft

void fftSetup(void);
void fftDestroy(void);

void fftBackwardTranspose(float2* u2);
void fftForwardTranspose(float2* u2);

void calcUmax(float2* u_x,float2* u_y,float2* u_z,float* ux,float* uy,float* uz);
void calcDmax(float2* u_x,float2* u_y,float* ux,float* uy);

float sumElements(float2* buffer_1);

//Rk

void RKstep(float2* ddv,float2* g,float time);
void setRK3(void);

//Non linear
void calcNL(float2* ddv,float2* g,float2* R_ddv,float2* R_g,float2* u,float2* v,float2* w,float2* dv,int ii);

//Mean velocity profile

void setRKmean(void);

void readNmean(float2* u);
void writeUmean(float2* u);
void readUmean(float2* u);
void readUtau(float2* wz);	

void writeU();
void readU();

void meanURKstep_1(int in);
void meanURKstep_2(float dt, int in);

void writeUmeanT(float2* u_r);

void secondDerivative(float2* u);
void implicitStepMeanU(float2* u,float betha_RK,float dt);

//hit_mpi

void reduceMAX(float* u1,float* u2,float* u3);
void reduceSUM(float* sum,float* sum_all);

int chyzx2xyz(double *y, double *x, int Nx, int Ny, int Nz,
	      int rank, int size);
int chxyz2yzx(double *x, double *y, int Nx, int Ny, int Nz,
	      int rank, int size);
int read_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size);
int wrte_parallel_float(char *filename, float *x, int NX, int NY, int NZ,
			 int rank, int size);
int read_parallel_double(char *filename, double *x, int NX, int NY, int NZ,
			 int rank, int size);
int wrte_parallel_double(char *filename, double *x, int NX, int NY, int NZ,
			 int rank, int size);

//Convolution

void convolution(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz);

//io

void readData(float2* ddv,float2* g);
void writeData(float2* ddv,float2* g);


//CUDA FUNCTIONS

//Hemholzt

extern void setHemholzt(void);
extern void setHemholztDouble(void);

extern void hemholztSolver(float2* u);
extern void hemholztSolver_double(float2* u);

//RK

extern void RKstep_1(float2* ddv,float2* g,float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in);
extern void RKstep_2(float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in);

//Non linear

extern void calcUW(float2* ux,float2* uz, float2* f,float2* g);
extern void calcHvg(float2* nl_x,float2* nl_y,float2* nl_z);

//Implicit step

extern void setImplicit(void);
extern void setImplicitDouble(void);

extern void implicitSolver(float2* u,float betha,float dt);
extern void implicitSolver_double(float2* u,float betha,float dt);

//Derivatives

extern void setDerivatives_HO(void);
extern void deriv_Y_HO(float2* u);
extern void deriv_YY_HO(float2* u);

void setDerivativesDouble(void);
extern void deriv_Y_HO_double(float2* u);
extern void deriv_YY_HO_double(float2* u);

//Dealias
extern void dealias(float2* u);
extern void set2zero(float2* u);

//Convolution kernels

extern void calcOmega(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz);
extern void calcRotor(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz);

//Check

extern void kernelCheck( cudaError_t error, const char* function);
extern void cufftCheck( cufftResult error, const char* function );
extern void cusparseCheck( cusparseStatus_t error, const char* function );
extern void cublasCheck(cublasStatus_t error, const char* function);
extern void cudaCheck( cudaError_t error, const char* function);
extern void mpiCheck( int error, const char* function);

//Bilaplacian

extern void bilaplaSolver(float2* ddv,float2* v,float2* dv,float betha,float dt);

//Routine check
void checkDerivatives(void);
void checkHemholzt(void);
void checkImplicit(void);



