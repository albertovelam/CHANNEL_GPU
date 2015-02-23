#include"channel.h"

//extern float2* aux_dev[6];

static __global__ void cast_kernel(float2* u,double2* v,domain_t domain)
{

  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  int h=i*NY*NZ+k*NY+j;

  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){

    float2 ud;
    double2 vd;

    vd=v[h];

    ud.x=(float)(vd.x);
    ud.y=(float)(vd.y);


    u[h]=ud;

  }

}

static __global__ void deriv_Y_WABC_kernel(double* Ac, double* Bc, double* Cc, domain_t domain)
{
  __shared__ double Fm[NY];
  int j   = threadIdx.x;
//  if(j<NY){
    Fm[j] = Fmesh(j*DELTA_Y-1.0);
    __syncthreads();
    double a,b;
    if(j==NY-1) a=Fm[j-2] - Fm[j-1];
    else        a=Fm[j+1] - Fm[j  ];//Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    if(j==0)    b=Fm[j+2] - Fm[j+1];
    else        b=Fm[j-1] - Fm[j  ];//Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double A,B,C;
    //int Ai,Bi,Ci;
    if(j==NY-1){
    //  Ai=NY-1; Bi=NY-2; Ci=NY-3;
      A= -(3.0*b+2.0*a)/(b*b+b*a);
      B=  ((b+a)*(2.0*a-b))/(b*a*a);
      C=  b*b/(a*a*b+a*a*a);
    }else if(j==0){
    //  Ai=0; Bi=1; Ci=2;
      A= -(3.0*a+2.0*b)/(a*a+a*b);
      B=  ((a+b)*(2.0*b-a))/(a*b*b);
      C=  a*a/(b*b*a+b*b*b);
    }else{
    //  Ai=j+1; Bi=j-1; Ci=j;
      A=2.0*(2.0*a*b*b-b*b*b)/((a*a-a*b)*(a-b)*(a-b));
      B=2.0*(2.0*b*a*a-a*a*a)/((b*b-a*b)*(a-b)*(a-b));
      C=-A-B;
    }
    //u_temp.x = A*(double)sf[Ai].x + B*(double)sf[Bi].x + C*(double)sf[Ci].x;
    //u_temp.y = A*(double)sf[Ai].y + B*(double)sf[Bi].y + C*(double)sf[Ci].y;
    //v[h]=u_temp;
    Ac[j] = A;
    Bc[j] = B;
    Cc[j] = C;
//  }
}

static __global__ void deriv_Y_RABC_kernel(double2* v,float2* u, double* Ac, double* Bc, double* Cc, domain_t domain)
{
  __shared__ float2 sf[NY];
  int k   = blockIdx.x;
  int i   = blockIdx.y;
  int j   = threadIdx.x;
  int h=i*NZ*NY+k*NY+j;
  double2 u_temp;
  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){
    sf[j]=u[h];
    int Ai,Bi,Ci;
    if(j==NY-1){
      Ai=NY-1; Bi=NY-2; Ci=NY-3;
    }else if(j==0){
      Ai=0; Bi=1; Ci=2;
    }else{
      Ai=j+1; Bi=j-1; Ci=j;
    }
    __syncthreads();
    double A = Ac[j];
    double B = Bc[j];
    double C = Cc[j];
    u_temp.x = A*(double)sf[Ai].x + B*(double)sf[Bi].x + C*(double)sf[Ci].x;
    u_temp.y = A*(double)sf[Ai].y + B*(double)sf[Bi].y + C*(double)sf[Ci].y;
    v[h]=u_temp;
  }
}


static __global__ void deriv_Y_ABC_kernel(double2* v,float2* u, domain_t domain)
{
  __shared__ float2 sf[NY];
  __shared__ double Fm[NY];
  int k   = blockIdx.x;
  int i   = blockIdx.y;
  int j   = threadIdx.x;
  int h=i*NZ*NY+k*NY+j;
  double2 u_temp;
  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){
    sf[j]=u[h];
    Fm[j] = Fmesh(j*DELTA_Y-1.0);
    __syncthreads();
    double a,b;
    if(j==NY-1) a=Fm[j-2] - Fm[j-1];
    else        a=Fm[j+1] - Fm[j  ];//Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    if(j==0)    b=Fm[j+2] - Fm[j+1];
    else        b=Fm[j-1] - Fm[j  ];//Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double A,B,C;
    int Ai,Bi,Ci;
    if(j==NY-1){
      Ai=NY-1; Bi=NY-2; Ci=NY-3;
      A= -(3.0*b+2.0*a)/(b*b+b*a);
      B=  ((b+a)*(2.0*a-b))/(b*a*a);
      C=  b*b/(a*a*b+a*a*a);
    }else if(j==0){
      Ai=0; Bi=1; Ci=2;
      A= -(3.0*a+2.0*b)/(a*a+a*b);
      B=  ((a+b)*(2.0*b-a))/(a*b*b);
      C=  a*a/(b*b*a+b*b*b);
    }else{
      Ai=j+1; Bi=j-1; Ci=j;
      A=2.0*(2.0*a*b*b-b*b*b)/((a*a-a*b)*(a-b)*(a-b));
      B=2.0*(2.0*b*a*a-a*a*a)/((b*b-a*b)*(a-b)*(a-b));
      C=-A-B;
    }
    u_temp.x = A*(double)sf[Ai].x + B*(double)sf[Bi].x + C*(double)sf[Ci].x;
    u_temp.y = A*(double)sf[Ai].y + B*(double)sf[Bi].y + C*(double)sf[Ci].y; 
    v[h]=u_temp;
  }
}


static __global__ void deriv_Y_kernel(double2* v,float2* u, domain_t domain)

{  
	
	
  //Define shared memory
	

  __shared__ double2 sf[NY+2];

  int k   = blockIdx.x;
  int i   = blockIdx.y;

  int j   = threadIdx.x;


  int h=i*NZ*NY+k*NY+j;

  double2 u_temp;
  double2 ap_1;
  double2 ac_1;
  double2 am_1;



  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){

    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	

    double A=2.0*(2.0*a*b*b-b*b*b)/((a*a-a*b)*(a-b)*(a-b));
    double B=2.0*(2.0*b*a*a-a*a*a)/((b*b-a*b)*(a-b)*(a-b));
    double C=-A-B;
	

    //Read from global to shared

    sf[j+1].x=(double)u[h].x;
    sf[j+1].y=(double)u[h].y;
						
    __syncthreads();


    ap_1=sf[j+2];
    ac_1=sf[j+1];	
    am_1=sf[j];
			
    u_temp.x=A*ap_1.x+C*ac_1.x+B*am_1.x;
    u_temp.y=A*ap_1.y+C*ac_1.y+B*am_1.y;

    //Second part multiplied by -1 to ensure simetry of the derivative
		
    if(j==0){
		
      a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh((j+1)*DELTA_Y-1.0);

		
      A= -(3.0*a+2.0*b)/(a*a+a*b);
      B=  ((a+b)*(2.0*b-a))/(a*b*b);
      C=  a*a/(b*b*a+b*b*b);

      u_temp.x=A*sf[1].x+B*sf[2].x+C*sf[3].x;
      u_temp.y=A*sf[1].y+B*sf[2].y+C*sf[3].y;			
		
    }		
	
    if(j==NY-1){
		
      a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh((j)*DELTA_Y-1.0);
      b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh((j-1)*DELTA_Y-1.0);

		
      A= -(3.0*a+2.0*b)/(a*a+a*b);
      B=  ((a+b)*(2.0*b-a))/(a*b*b);
      C=  a*a/(b*b*a+b*b*b);

      u_temp.x=A*sf[NY].x+B*sf[NY-1].x+C*sf[NY-2].x;
      u_temp.y=A*sf[NY].y+B*sf[NY-1].y+C*sf[NY-2].y;				
					
    }	
				

    v[h]=u_temp;
 	
  }

}


__global__ void deriv_YY_kernel(double2* v,float2* u,domain_t domain)

{  
	
  //Define shared memory
		
  __shared__ double2 sf[NY+2];

  int k   = blockIdx.x;
  int i   = blockIdx.y;

  int j   = threadIdx.x;

  int h=i*NZ*NY+k*NY+j;

  double2 u_temp;

  double2 ap_1;
  double2 ac_1;	
  double2 am_1;



  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){
		
    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	


    double A=-12.0*b/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double B= 12.0*a/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double C=-A-B;
    double E;

    //Read from global so shared
	
    sf[j+1].x=(double)u[h].x;
    sf[j+1].y=(double)u[h].y;
	
    __syncthreads();


    ap_1=sf[j+2];
    ac_1=sf[j+1];
    am_1=sf[j];	


    u_temp.x=A*ap_1.x+C*ac_1.x+B*am_1.x;
    u_temp.y=A*ap_1.y+C*ac_1.y+B*am_1.y;
		
    if(j==0){

      a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
		
      A= 6.0/((a-b)*(2.0*a-b));
      B= -6.0*a/((a*b-b*b)*(2.0*a-b));
	
      E=-A-B;

      u_temp.x=E*sf[1].x+A*sf[2].x+B*sf[3].x;
      u_temp.y=E*sf[1].y+A*sf[2].y+B*sf[3].y;				

    }		
	
    if(j==NY-1){

      a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);

      A= 6.0/((a-b)*(2.0*a-b));
      B= -6.0*a/((a*b-b*b)*(2.0*a-b));
	
      E=-A-B;

      u_temp.x=E*sf[NY].x+A*sf[NY-1].x+B*sf[NY-2].x;
      u_temp.y=E*sf[NY].y+A*sf[NY-1].y+B*sf[NY-2].y;			

    }	
		
    v[h]=u_temp;
 	
  }

}

static __global__ void setDiagkernel_Y(double2* ldiag,double2* cdiag,double2* udiag, domain_t domain){  

	
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE/NSTEPS && j<NY && k<NZ)
    {

      //COEFICIENTS OF THE NON_UNIFORM GRID

      double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
	
				
      double alpha=b*b/((a-b)*(a-b));
      double betha=a*a/((a-b)*(a-b));
		

      //veamos
	
      double2 ldiag_h;
      double2 cdiag_h;
      double2 udiag_h;
	
      udiag_h.x=alpha;
      udiag_h.y=0.0;

      cdiag_h.x=1.0;
      cdiag_h.y=0.0;

      ldiag_h.x=betha;
      ldiag_h.y=0.0;

      if(j==0){

	a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
	b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh((j+1)*DELTA_Y-1.0);
	

	alpha=(a+b)/b;

	udiag_h.x=alpha;
	cdiag_h.x=1.0;
	ldiag_h.x=0.0;
      }
	
      if(j==NY-1){
			
	a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh((j)*DELTA_Y-1.0);
	b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh((j-1)*DELTA_Y-1.0);
			

	alpha=(a+b)/b;			
			
	ldiag_h.x=alpha;
	cdiag_h.x=1.0;
	udiag_h.x=0.0;		
      }	
	

      // Write		

      ldiag[h]=ldiag_h;
      cdiag[h]=cdiag_h;
      udiag[h]=udiag_h;			
	
    }

}

static __global__ void setDiagkernel_YY(double2* ldiag,double2* cdiag,double2* udiag, domain_t domain){  

	
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  // [i,k,j][NX,NZ,NY]	

  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE/NSTEPS && j<NY && k<NZ)
    {

      //COEFICIENTS OF THE NON_UNIFORM GRID

      double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
	
      double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
      double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
		

      //veamos
	
      double2 ldiag_h;
      double2 cdiag_h;
      double2 udiag_h;
	
      udiag_h.x=alpha;
      udiag_h.y=0.0;

      cdiag_h.x=1.0;
      cdiag_h.y=0.0;

      ldiag_h.x=betha;
      ldiag_h.y=0.0;

      if(j==0){

	a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
	b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
	

	alpha=(a+b)/(2.0*a-b);
	udiag_h.x=alpha;
	cdiag_h.x=1.0;

      }
	
      if(j==NY-1){
			
	a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
	b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
			

	alpha=(a+b)/(2.0*a-b);			
			
	ldiag_h.x=alpha;
	cdiag_h.x=1.0;
	udiag_h.x=0.0;		
      }	
	

      // Write		

      ldiag[h]=ldiag_h;
      cdiag[h]=cdiag_h;
      udiag[h]=udiag_h;			
	
    }

}

static dim3 threadsPerBlock_B;
static dim3 blocksPerGrid_B;

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;
#ifdef USE_CUSPARSE
static cusparseHandle_t cusparse_handle;
#endif
static double *a_d, *b_d, *c_d;
static double *Ac,*Bc,*Cc;

void setDerivativesDouble(domain_t domain){
START_RANGE("setDerivativesDouble",16)
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE/NSTEPS;	

  threadsPerBlock_B.x= THREADSPERBLOCK_IN;
  threadsPerBlock_B.y= THREADSPERBLOCK_IN;

  blocksPerGrid_B.y=NXSIZE/NSTEPS/threadsPerBlock_B.x;
  blocksPerGrid_B.x=NZ*NY/threadsPerBlock_B.y;
#ifdef USE_CUSPARSE
  cusparseCheck(cusparseCreate(&cusparse_handle),domain,"Handle");
#else

  double *a_h,*b_h,*c_h;
  a_h = (double*)malloc(NY*sizeof(double));
  b_h = (double*)malloc(NY*sizeof(double));
  c_h = (double*)malloc(NY*sizeof(double));

  for(int j=0; j<NY; j++){
 
      double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      double alpha=b*b/((a-b)*(a-b));
      double betha=a*a/((a-b)*(a-b));

      c_h[j]=alpha;
      b_h[j]=1.0;
      a_h[j]=betha;

      if(j==0){

        a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
        b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh((j+1)*DELTA_Y-1.0);
        alpha=(a+b)/b;

        c_h[j]=alpha;
        b_h[j]=1.0;
        a_h[j]=0.0;
      }

      if(j==NY-1){

        a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh((j)*DELTA_Y-1.0);
        b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh((j-1)*DELTA_Y-1.0);
        alpha=(a+b)/b;

        a_h[j]=alpha;
        b_h[j]=1.0;
        c_h[j]=0.0;
      }
  }
  c_h[0] = c_h[0]/b_h[0];
  for(int j=1; j<NY; j++){
    double m = 1.0/(b_h[j] - a_h[j]*c_h[j-1]);
    c_h[j] = c_h[j]*m;
  }

  CHECK_CUDART( cudaMalloc((void**)&a_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&b_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&c_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMemcpy(a_d,a_h,NY*sizeof(double),cudaMemcpyHostToDevice) );
  CHECK_CUDART( cudaMemcpy(b_d,b_h,NY*sizeof(double),cudaMemcpyHostToDevice) );
  CHECK_CUDART( cudaMemcpy(c_d,c_h,NY*sizeof(double),cudaMemcpyHostToDevice) );

  free(a_h); free(b_h); free(c_h);

  CHECK_CUDART( cudaMalloc((void**)&Ac, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&Bc, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&Cc, NY*sizeof(double)) );
  deriv_Y_WABC_kernel<<<1,NY>>>(Ac,Bc,Cc,domain);

#endif
END_RANGE
}

__global__ void solve_ep(double2* __restrict x, const double* __restrict a, const double* __restrict b, const double* __restrict c, double2* __restrict d, domain_t domain){

  int k = threadIdx.x + blockDim.x*blockIdx.x;

  if(k>=NXSIZE/NSTEPS*NZ) return;

  x += k*NY;
  d += k*NY;

  d[0].x = x[0].x/b[0];
  d[0].y = x[0].y/b[0];

  for(int j=1; j<NY; j++){
    double m = 1.0/(b[j] - a[j]*c[j-1]);
    d[j].x = (x[j].x - a[j]*d[(j-1)].x)*m;
    d[j].y = (x[j].y - a[j]*d[(j-1)].y)*m;
  }
  x[NY-1].x = d[NY-1].x;
  x[NY-1].y = d[NY-1].y;
  for(int j=NY-2; j>=0; j--){
    x[j].x = d[j].x - c[j]*x[(j+1)].x;
    x[j].y = d[j].y - c[j]*x[(j+1)].y;
  }
}


__global__ void solve_ep_t(float2* __restrict x, const double* __restrict a, const double* __restrict b, const double* __restrict c, 
                           double2* __restrict d, const double* __restrict A, const double* __restrict B, const double* __restrict C, domain_t domain){

  int k = threadIdx.x + blockDim.x*blockIdx.x;

  if(k>=NXSIZE/NSTEPS*NZ) return;

  x += k;
  d += k; 
  //double2 d[NY];
  double2 xA, xB, xC;
  xA.x = (double)x[0*NXSIZE/NSTEPS*NZ].x;
  xA.y = (double)x[0*NXSIZE/NSTEPS*NZ].y;
  xB.x = (double)x[1*NXSIZE/NSTEPS*NZ].x;
  xB.y = (double)x[1*NXSIZE/NSTEPS*NZ].y;
  xC.x = (double)x[2*NXSIZE/NSTEPS*NZ].x;
  xC.y = (double)x[2*NXSIZE/NSTEPS*NZ].y;

  double2 xD;
  xD.x = A[0]*xA.x + B[0]*xB.x + C[0]*xC.x;
  xD.y = A[0]*xA.y + B[0]*xB.y + C[0]*xC.y;

  double2 dM;
  dM.x = xD.x;///b[0];
  dM.y = xD.y;///b[0];
  d[0].x = dM.x;
  d[0].y = dM.y;
  double cm1 = c[0];

  for(int j=1; j<NY-1; j++){
    double m = 1.0/(/*b[j]*/1.0 - a[j]*cm1);
    cm1 = c[j];
    xD.x = A[j]*xC.x + B[j]*xA.x + /*C[j]*/(-B[j]-A[j])*xB.x;
    xD.y = A[j]*xC.y + B[j]*xA.y + /*C[j]*/(-B[j]-A[j])*xB.y;
    dM.x = (xD.x - a[j]*dM.x)*m;
    dM.y = (xD.y - a[j]*dM.y)*m;
    d[j*NXSIZE/NSTEPS*NZ] = dM;
    if(j<NY-2){
      xA = xB;
      xB = xC;
      float2 next_x = x[(j+2)*NXSIZE/NSTEPS*NZ];
      xC.x = (double)next_x.x;
      xC.y = (double)next_x.y;
    }
  }
/*
  double m1 = 1.0/(b[NY-2] - a[NY-2]*c[NY-3]);
  xD.x = A[NY-2]*xC.x + B[NY-2]*xA.x + C[NY-2]*xB.x;
  xD.y = A[NY-2]*xC.y + B[NY-2]*xA.y + C[NY-2]*xB.y;
  d[(NY-2)*NXSIZE/NSTEPS*NZ].x = (xD.x - a[NY-2]*d[(NY-3)*NXSIZE/NSTEPS*NZ].x)*m1;
  d[(NY-2)*NXSIZE/NSTEPS*NZ].y = (xD.y - a[NY-2]*d[(NY-3)*NXSIZE/NSTEPS*NZ].y)*m1;
*/
  double m2 = 1.0/(/*b[NY-1]*/1.0 - a[NY-1]*cm1);
  xD.x = A[NY-1]*xC.x + B[NY-1]*xB.x + C[NY-1]*xA.x;
  xD.y = A[NY-1]*xC.y + B[NY-1]*xB.y + C[NY-1]*xA.y;
  xD.x = (xD.x - a[NY-1]*dM.x)*m2;
  xD.y = (xD.y - a[NY-1]*dM.y)*m2;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].y = (float)xD.y;

  for(int j=NY-2; j>=0; j--){
    xD.x = d[j*NXSIZE/NSTEPS*NZ].x - c[j]*xD.x;
    xD.y = d[j*NXSIZE/NSTEPS*NZ].y - c[j]*xD.y;
    x[j*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
    x[j*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
    //x[j*NXSIZE/NSTEPS*NZ].x = (float)(d[j*NXSIZE/NSTEPS*NZ].x - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].x);
    //x[j*NXSIZE/NSTEPS*NZ].y = (float)(d[j*NXSIZE/NSTEPS*NZ].y - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].y);
  }

}

extern void deriv_Y_HO_double(float2* u, domain_t domain){
START_RANGE("deriv_Y_H0_double",17)

#ifndef USE_CUSPARSE
  trans_yzx_to_zxy(u/*+i*NXSIZE/NSTEPS*NZ*/, aux_dev[0], 0, domain);
  dim3 grid,block;
  block.x=128;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;
  solve_ep_t<<<grid,block>>>(aux_dev[0],a_d,b_d,c_d,AUX,Ac,Bc,Cc,domain);
  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);
#else

  //SIZE OF LDIAG CDIAG UDIAG AND AUX
  //2*SIZE/NSTEPS


  for(int i=0;i<NSTEPS;i++){
    setDiagkernel_Y<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG,domain);
    deriv_Y_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY);
    kernelCheck(RET,"W_kernel");
    cusparseCheck(cusparseZgtsvStridedBatch(cusparse_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),
                  domain,
                  "HEM");
    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX,domain);
  }
#endif
END_RANGE
  return;
}

#if 0
  for(int i=0;i<NSTEPS;i++){
//if(first_time){	
    setDiagkernel_Y<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG,domain);
//}
//    deriv_Y_ABC_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY,domain);
//    kernelCheck(RET,domain,"W_kernel");	

static double *a,*b,*c,*cs;
static double *a_d, *b_d, *c_d;
static double2 *x2,*a2,*b2,*c2;
static double *Ac,*Bc,*Cc;
if(first_time){
  CHECK_CUDART( cudaMalloc((void**)&Ac, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&Bc, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&Cc, NY*sizeof(double)) );
  deriv_Y_WABC_kernel<<<1,NY>>>(Ac,Bc,Cc,domain);
  CHECK_CUDART( cudaMallocHost((void**)&a, NY*sizeof(double)) );
  CHECK_CUDART( cudaMallocHost((void**)&b, NY*sizeof(double)) );
  CHECK_CUDART( cudaMallocHost((void**)&c, NY*sizeof(double)) );
  CHECK_CUDART( cudaMallocHost((void**)&cs, NY*sizeof(double)) );
  CHECK_CUDART( cudaMallocHost((void**)&a2, NY*sizeof(double2)) );
  CHECK_CUDART( cudaMallocHost((void**)&b2, NY*sizeof(double2)) );
  CHECK_CUDART( cudaMallocHost((void**)&c2, NY*sizeof(double2)) );
  CHECK_CUDART( cudaMallocHost((void**)&x2, NXSIZE/NSTEPS*NZ*NY*sizeof(double2)) );
  CHECK_CUDART( cudaMalloc((void**)&a_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&b_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMalloc((void**)&c_d, NY*sizeof(double)) );
  CHECK_CUDART( cudaMemcpy(a2,LDIAG,NY*sizeof(double2),cudaMemcpyDeviceToHost) );
  CHECK_CUDART( cudaMemcpy(b2,CDIAG,NY*sizeof(double2),cudaMemcpyDeviceToHost) );
  CHECK_CUDART( cudaMemcpy(c2,UDIAG,NY*sizeof(double2),cudaMemcpyDeviceToHost) );
  for(int j=0; j<NY; j++){
    a[j] = a2[j].x;
    b[j] = b2[j].x;
    c[j] = c2[j].x;
  }
  cs[0] = c[0]/b[0];
  for(int j=1; j<NY; j++){
    double m = 1.0/(b[j] - a[j]*cs[j-1]);
    cs[j] = c[j]*m;
  }
  CHECK_CUDART( cudaMemcpy(a_d,a,NY*sizeof(double),cudaMemcpyHostToDevice) );
  CHECK_CUDART( cudaMemcpy(b_d,b,NY*sizeof(double),cudaMemcpyHostToDevice) );
  CHECK_CUDART( cudaMemcpy(c_d,cs,NY*sizeof(double),cudaMemcpyHostToDevice) );
  first_time=0;
}

  trans_yzx_to_zxy(u/*+i*NXSIZE/NSTEPS*NZ*/, aux_dev[0], 0, domain);
//  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);

//  deriv_Y_RABC_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY,Ac,Bc,Cc,domain);
/*
__global__ void solve_ep_t(float2* __restrict x, const double* __restrict a, const double* __restrict b, const double* __restrict c,
                           double2* __restrict d, const double* __restrict A, const double* __restrict B, const double* __restrict C, domain_t domain){
*/
///*
  dim3 grid,block;
  block.x=128;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;
  solve_ep_t<<<grid,block>>>(aux_dev[0],a_d,b_d,c_d,AUX,Ac,Bc,Cc,domain);  
//*/
/*
  dim3 grid,block;
  block.x=32;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;
  solve_ep<<<grid,block>>>(AUX,a_d,b_d,c_d,x2,domain);
*/

  

    //Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))
/*
    cusparseCheck(cusparseZgtsvStridedBatch(cusparse_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),
		  domain,
		  "HEM");
*/
//    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX,domain);

  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);


  }
END_RANGE
  return;
}

#endif

extern void deriv_YY_HO_double(float2* u, domain_t domain){
START_RANGE("deriv_YY_HO_double",18)

  //SIZE OF LDIAG CDIAG UDIAG AND AUX
  //2*SIZE/NSTEPS

  /*
    for(int i=0;i<NSTEPS;i++){
	
    setDiagkernel_YY<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG);

    deriv_YY_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY);
    kernelCheck(RET,"W_kernel");	

    //Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))

    cusparseCheck(cusparseZgtsvStridedBatch(cusparse_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),"HEM");

    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX);

    }
  */

  // Second derivative has to be checked before production.
  // This is crap and I know it.
  
  deriv_Y_HO_double(u,domain);
  deriv_Y_HO_double(u,domain);
	
END_RANGE
  return;

}




