#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//#include <cusparse_v2.h> 
#include <cublas_v2.h>
#include <cufft.h>
#include <math.h>

#include <stdlib.h>
#include <string.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <libconfig.h>

//MESHGRID

#define th2 (0.964027580075816903)
#define Fmesh(X)  (tanh(2.0*(X))/th2)

typedef struct domain_t{
  int nx;
  int ny;
  int nz;
  int rank;
  int size;
  int iglobal;
  float lx;
  float lz;
  float reynolds;
  float massflux;
} domain_t;

typedef struct paths_t{
  char ginput[100];
  char goutput[100];
  char ddvinput[100];
  char ddvoutput[100];
  char umeaninput[100];
  char umeanoutput[100];
  char path[100];
  int freq_stats;
  int nsteps;
} paths_t;

#if !defined(NX) || !defined(NY) || !defined(NZ)
#error "Sizes have to be defined at compile time"
#endif

//Dimensions in Y direction 
// h=1.0f channel height 2.0f

// TODO: make that configurable too.
#define LY 2.0f
#define PI 3.14159265f
#define PI2 (2.0f*PI)

#define DELTA_Y  (2.0f/(NY-1))
#define LX domain.lx
#define LZ domain.lz

//Reynolds number and bulk velocity
#define REYNOLDS domain.reynolds
#define QVELOCITY domain.massflux

//MPI number of process
#define MPISIZE domain.size
#define NXSIZE NX/MPISIZE
#define NYSIZE NY/MPISIZE

#define RANK domain.rank
#define IGLOBAL domain.iglobal

//size local
#define SIZE NXSIZE*NY*NZ*sizeof(float2)

static cublasHandle_t   CUBLAS_HANDLE; 
static cudaError_t RET;
static const int THREADSPERBLOCK_IN=16;

extern char host_name[MPI_MAX_PROCESSOR_NAME];
extern char mybus[16];
extern int SMCOUNT;
extern MPI_Request *send_requests;
extern MPI_Request *recv_requests;
extern MPI_Status *send_status;
extern MPI_Status *recv_status;

extern float2* aux_dev[6];
extern float2* aux_host_1[6];
extern float2* aux_host_2[6];

extern cudaStream_t compute_stream;
extern cudaStream_t h2d_stream;
extern cudaStream_t d2h_stream;
extern cudaEvent_t events[1000];


#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "Rank: %d Node: %s GPU: %s CUDART: %s = %d (%s) at (%s:%d)\n", RANK, host_name, mybus, #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUBLAS(x) do { \
  cublasStatus_t cublasStatus = (x); \
  if(cublasStatus != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "Rank: %d Node: %s GPU: %s CUBLAS: %s = %d at (%s:%d)\n",RANK, host_name, mybus, #x, cublasStatus,__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

#ifdef USE_CUSPARSE
#define CHECK_CUSPARSE(x) do { \
  cusparseStatus_t cusparseStatus = (x); \
  if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "Rank: %d Node: %s GPU: %s CUSPARSE: %s = %d at (%s:%d)\n",RANK, host_name, mybus, #x, cusparseStatus,__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)
#endif

//#define USE_NVTX
#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors4[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors4 = sizeof(colors4)/sizeof(uint32_t);

#define START_RANGE_ASYNC(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors4;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors4[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define END_RANGE_ASYNC { \
        nvtxRangePop(); \
}


#define START_RANGE(name,cid) { \
        cudaDeviceSynchronize(); \
        int color_id = cid; \
        color_id = color_id%num_colors4;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors4[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define END_RANGE { \
        cudaDeviceSynchronize(); \
        nvtxRangePop(); \
}
#else
#define START_RANGE(name,cid)
#define END_RANGE
#define START_RANGE_ASYNC(name,cid)
#define END_RANGE_ASYNC
#endif


//BUFFERS FOR DERIVATIVES

extern double2* LDIAG;
extern double2* UDIAG;
extern double2* CDIAG;

extern double2* AUX;


#define NSTEPS 1
#define SIZE_AUX 2*SIZE/NSTEPS

//CUSPARSE HANDELS

/////CUDA////
extern void trans_zyx_to_yzx(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_yzx_to_zyx(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_yzx_to_zyx_yblock(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_zxy_to_yzx(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_zxy_to_zyx(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_zyx_to_zxy(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_zyx_yblock_to_yzx(float2* input, float2* output,cudaStream_t stream, domain_t domain);
extern void trans_yzx_to_zxy(float2* input, float2* output,cudaStream_t stream,domain_t domain);

extern void calc_Umax2(float2* ux, float2* uy, float2* uz, float* temp,domain_t domain);
extern void calc_Dmax2(float2* ux, float2* uy, float* temp,domain_t domain);

extern MPI_Request *send_requests;
extern MPI_Request *recv_requests;
extern MPI_Status *send_status;
extern MPI_Status *recv_status;

void fftBack1T_A(float2* u1,int id, domain_t domain);
void fftBack1T_B(float2* u1,int id, domain_t domain);

void fftForw1T_A(float2* u1,int id, domain_t domain);
void fftForw1T_B(float2* u1,int id, domain_t domain);


///////////////////////////C///////////////////////////

//Set up

config_t read_config_file(char* name);
void read_domain_from_config(domain_t*, config_t*);
void read_filenames_from_config(paths_t*, config_t*);
void setUp(domain_t domain);

//fft

void fftSetup(domain_t domain);
void fftDestroy(void);

void fftBackward(float2* buffer, domain_t domain);
void fftForward(float2* buffer, domain_t domain);

void fftBackwardTranspose(float2* u2, domain_t domain);
void fftForwardTranspose(float2* u2, domain_t domain);

void calcUmax(float2* u_x,float2* u_y,float2* u_z,float* ux,float* uy,float* uz, domain_t domain);
void calcDmax(float2* u_x,float2* u_y,float* ux,float* uy, domain_t domain);

float sumElementsReal(float2* buffer_1, domain_t domain);
void sumElementsComplex(float2* buffer_1,float* out, domain_t domain);

//Rk

void RKstep(float2* ddv,float2* g,float time, domain_t domain, paths_t path);
void setRK3(domain_t domain);

//Non linear
void calcNL(float2* ddv,float2* g,float2* R_ddv,float2* R_g,
	    float2* u,float2* v,float2* w,
	    float2* dv,int ii,int counter, domain_t domain,
	    paths_t path);

//Mean velocity profile

void setRKmean(void);

void readNmean(float2* u, domain_t domain);
void writeUmean(float2* u, domain_t domain);
void readUmean(float2* u);
void readUtau(float2* wz, domain_t domain);	

void writeU(char*);
void readU(char*);

void meanURKstep_1(int in, domain_t domain);
void meanURKstep_2(float dt, int in, domain_t domain, paths_t path);

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
int read_parallel_float(char *filename, float *x, int Nx, int Ny, int Nz,
			 int rank, int size);
int wrte_parallel_float(char *filename, float *x, int Nx, int Ny, int Nz,
			 int rank, int size);
int read_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);
int wrte_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);

//ImposeSymetry
void imposeSymetry(float2* u,float2* v, domain_t domain);

//Convolution

//void convolution(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz, domain_t domain);
void convolution_max(int ii,float2* ddv, float2* g, float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz, domain_t domain);
//io

void readData(float2* ddv,float2* g, paths_t path, domain_t domain);
void writeData(float2* ddv,float2* g, paths_t path, domain_t domain);
void genRandData(float2* ddv,float2* g,float F, domain_t domain);

//CUDA local transpose

void setTransposeCudaMpi(domain_t domain);
void transposeXYZ2YZX(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain);
void transposeYZX2XYZ(float2* u1,int Nx,int Ny,int Nz,int rank,int sizeMpi, domain_t domain);
void transposeBatched(float2* u_2,const float2* u_1,int Nx,int Ny,int batch, domain_t domain);
void transpose(float2* u_2,const float2* u_1,int Nx,int Ny, domain_t domain);

//Statistics
void calcSt(float2* dv,float2* u,float2* v,float2* w, domain_t domain, paths_t path);


//Routine check
void checkDerivatives(void);
void checkHemholzt(void);
void checkImplicit(void);

//CUDA FUNCTIONS

//Hemholzt

extern void setHemholzt(domain_t domain);
extern void setHemholztDouble(domain_t domain);

extern void hemholztSolver(float2* u, domain_t domain);
extern void hemholztSolver_double(float2* u, domain_t domain);

//RK

extern void RKstep_1(float2* ddv,float2* g,float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in, domain_t domain);
extern void RKstep_2(float2* ddv_w,float2* g_w,float2* Rddv,float2* Rg,float dt,int in, domain_t domain);

//Non linear

extern void calcUW(float2* ux,float2* uz, float2* f,float2* g, domain_t domain);
extern void calcHvg(float2* nl_x,float2* nl_y,float2* nl_z, domain_t domain);

//Implicit step

extern void setImplicit(domain_t domain);
extern void setImplicitDouble(domain_t domain);

extern void implicitSolver(float2* u,float betha,float dt, domain_t domain);
extern void implicitSolver_double(float2* u,float betha,float dt, domain_t domain);

//Derivatives

extern void setDerivatives_HO(domain_t domain);
extern void deriv_Y_HO(float2* u, domain_t domain);
extern void deriv_YY_HO(float2* u, domain_t domain);

void setDerivativesDouble(domain_t domain);
extern void deriv_Y_HO_double(float2* u, domain_t domain);
extern void deriv_YY_HO_double(float2* u, domain_t domain);

//Dealias
extern void dealias(float2* u, domain_t domain);
extern void set2zero(float2* u, domain_t domain);
extern void normalize(float2* u, domain_t domain);
extern void scale(float2* u,float S, domain_t domain);

//Convolution kernels

extern void calcOmega(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain);
extern void calcRotor(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain);
extern void calcRotor3(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain );
extern void calcRotor12(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, domain_t domain );


//Check

extern void kernelCheck( cudaError_t error, domain_t domain, const char* function);
extern void cufftCheck( cufftResult error, domain_t domain, const char* function );
//extern void cusparseCheck( cusparseStatus_t error, domain_t domain, const char* function );
extern void cublasCheck(cublasStatus_t error, domain_t domain, const char* function);
extern void cudaCheck( cudaError_t error, domain_t domain, const char* function);
extern void mpiCheck( int error, const char* function);

//Bilaplacian

extern void bilaplaSolver(float2* ddv,float2* v,float2* dv,float betha,float dt, domain_t domain);
extern void bilaplaSolver_double(float2* ddv, float2* v, float2* dv, float betha,float dt, domain_t domain);


//Phase shift 
extern void phaseShift(float2* tx,float2* ty,float2* tz,float Delta1,float Delta3);
extern void sumCon(float2* ax,float2* ay,float2* az,float2* tx,float2* ty,float2* tz);


