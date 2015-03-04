#include"channel.h"
//extern float2* aux_dev[6];
//extern float2* aux_host_1[6];

static __global__ void cast_kernel(float2* u,double2* v,domain_t domain)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int j=k%NY;
  k=(k-j)/NY;

  int h=i*NY*NZ+k*NY+j;

  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){

    float2 ud;
    double2 vd;

    vd=v[h];

    /*
      ud.x=__double2float_rn(vd.x);
      ud.y=__double2float_rn(vd.y);
    */

    ud.x=(float)(vd.x);
    ud.y=(float)(vd.y);

    u[h]=ud;

  }

}


static __global__ void rhs_A_kernel(double2* v,float2* u,domain_t domain)
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

  double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
  double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	

  double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
  double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);


  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){

    //Read from global so shared

    sf[j+1].x=(double)u[h].x;
    sf[j+1].y=(double)u[h].y;

    __syncthreads();

    ap_1=sf[j+2];	
    ac_1=sf[j+1];
    am_1=sf[j];
		
		
    u_temp.x=(alpha*ap_1.x+ac_1.x+betha*am_1.x);
    u_temp.y=(alpha*ap_1.y+ac_1.y+betha*am_1.y);

    if(j==0){
      u_temp.x=0.0;
      u_temp.y=0.0;		
    }		
	
    if(j==NY-1){
      u_temp.x=0.0;
      u_temp.y=0.0;		
    }	

    v[h]=u_temp;
 	
  }
}

static __global__ void rhs_A_kernel_bilaplacian(double2* v,domain_t domain)
{  

  //Define shared memory



  int k   = blockIdx.x;
  int i   = blockIdx.y;

  int j   = threadIdx.x;


  int h=i*NZ*NY+k*NY+j;

  double2 u_temp;


  if(i<NXSIZE/NSTEPS & k<NZ & j<NY){

    u_temp.x=0.0;
    u_temp.y=0.0;	

    if(j==0){
      u_temp.x=1.0;
      u_temp.y=1.0;		
    }	

    v[h]=u_temp;
 	
  }

}

static __global__ void setABalphabetha(double2* AB, /*double* B,*/ double2* alphabetha,/* double* betha,*/ domain_t domain){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j<NY){
    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    AB[j].x = -12.0*b/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    AB[j].y =  12.0*a/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    alphabetha[j].x = -(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
    alphabetha[j].y = -( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
  }
}

static __global__ void setDiagkernel(double2* ldiag,double2* cdiag,double2* udiag,float bethaDt,int nstep,domain_t domain){  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j=k%NY;
  k=(k-j)/NY;
  // [i,k,j][NX,NZ,NY]	
  int h=i*NY*NZ+k*NY+j;

  if (i<NXSIZE/NSTEPS && j<NY && k<NZ)
    {

      double k1;
      double k3;
	
      double kk;
	
      int stride=nstep*NXSIZE/NSTEPS;	

      // X indices		
      k1=(i+IGLOBAL+stride)<NX/2 ? (double)(i+IGLOBAL+stride) : (double)(i+IGLOBAL+stride)-(double)NX ;
		
      // Z indices
      k3=(double)k;
	
      //Fraction
      k1=(PI2/LX)*k1;
      k3=(PI2/LZ)*k3;	

      kk=k1*k1+k3*k3;

      double nu=1.0/REYNOLDS;
      double D=nu*bethaDt;

      //COEFICIENTS OF THE NON_UNIFORM GRID

      double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
      double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
	

      double A=-12.0*b/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
      double B= 12.0*a/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
      double C=-A-B;
		
      double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
      double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);

      //veamos
	
      double2 ldiag_h;
      double2 cdiag_h;
      double2 udiag_h;
	
      ldiag_h.x=betha-D*B+D*kk*betha;
      ldiag_h.y=0.0;			
	
      cdiag_h.x=1.0-D*C+D*kk*1.0;
      cdiag_h.y=0.0;		
	
      udiag_h.x=alpha-D*A+D*kk*alpha;
      udiag_h.y=0.0;	

      //To be improved 
		
      if(j==0){
	ldiag_h.x=0.0;
	cdiag_h.x=1.0;
	udiag_h.x=0.0;
      }
      /*		
      if(j==1){
	ldiag_h.x=0.0;
      }
      */
      if(j==NY-1){
	ldiag_h.x=0.0;
	cdiag_h.x=1.0;
	udiag_h.x=0.0;
      }	
      	
      if(j==NY-2){
	udiag_h.x=0.0;
      }
      
      // Write		

      ldiag[h]=ldiag_h;
      cdiag[h]=cdiag_h;
      udiag[h]=udiag_h;			
	
    }

}



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


static dim3 threadsPerBlock_B;
static dim3 blocksPerGrid_B;
#ifdef USE_CUSPARSE
//static cusparseHandle_t implicit_handle;
#endif
static cusparseHandle_t implicit_handle;
static double2 *AB_c,*alphabetha_c;
static double *cstar;


extern void setImplicitDouble(domain_t domain){
START_RANGE("setImplicitDouble",19)
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE/NSTEPS;	

  threadsPerBlock_B.x= THREADSPERBLOCK_IN;
  threadsPerBlock_B.y= THREADSPERBLOCK_IN;

  blocksPerGrid_B.x=NXSIZE/NSTEPS/threadsPerBlock_B.x;
  blocksPerGrid_B.y=NZ*NY/threadsPerBlock_B.y;
  
#ifdef USE_CUSPARSE
  cusparseCheck(cusparseCreate(&implicit_handle),domain,"Handle");
#else
  CHECK_CUDART( cudaMalloc((void**)&AB_c    , NY*sizeof(double2)) );
  CHECK_CUDART( cudaMalloc((void**)&alphabetha_c, NY*sizeof(double2)) );
  //CHECK_CUDART( cudaMalloc((void**)&cstar  , NXSIZE/NSTEPS*NZ*NY*sizeof(double)) );
  setABalphabetha<<<(NY+31)/32,32>>>(AB_c,alphabetha_c,domain);
#endif
END_RANGE
}

__global__ void hsolve_ep_t(float2* __restrict x, const double2* __restrict AB, /*const double* __restrict B,*/ const double2* __restrict alphabetha, /*const double* __restrict betha,*/
                           double2* __restrict d, double* __restrict cs, const int nstep, domain_t domain){

  int ik = threadIdx.x + blockDim.x*blockIdx.x;

  if(ik>=NXSIZE/NSTEPS*NZ) return;

  x  += ik;
  d  += ik;
  cs += ik;

  int i = ik/NZ;
  int k = ik%NZ;

  int stride=nstep*NXSIZE/NSTEPS;
  double k1=(i+IGLOBAL+stride)<NX/2 ? (double)(i+IGLOBAL+stride) : (double)(i+IGLOBAL+stride)-(double)NX ;
  double k3=(double)k;
  k1=(PI2/LX)*k1;
  k3=(PI2/LZ)*k3;
  double kk=k1*k1+k3*k3;

  double2 xA, xB, xC;
  xA.x = (double)x[0*NXSIZE/NSTEPS*NZ].x;
  xA.y = (double)x[0*NXSIZE/NSTEPS*NZ].y;
  xB.x = (double)x[1*NXSIZE/NSTEPS*NZ].x;
  xB.y = (double)x[1*NXSIZE/NSTEPS*NZ].y;
  xC.x = (double)x[2*NXSIZE/NSTEPS*NZ].x;
  xC.y = (double)x[2*NXSIZE/NSTEPS*NZ].y;

  double2 xD;
  //xD.x = A[0]*xA.x + B[0]*xB.x + C[0]*xC.x;
  //xD.y = A[0]*xA.y + B[0]*xB.y + C[0]*xC.y;

  double2 dM;
  dM.x = 0.0;//xD.x/b[0];
  dM.y = 0.0;//xD.y/b[0];
  d[0] = dM;
  double cm1 = 0.0;
  cs[0] = 0.0;
  for(int j=1; j<NY-1; j++){
    double C = -AB[j].x-AB[j].y;
    double a=AB[j].y-kk*alphabetha[j].y;
    double b=C      -kk;
    double c=AB[j].x-kk*alphabetha[j].x;
    if(j==1   ) a=0.0;
    if(j==NY-2) c=0.0;
    xD.x = alphabetha[j].x*xC.x + alphabetha[j].y*xA.x + xB.x;
    xD.y = alphabetha[j].x*xC.y + alphabetha[j].y*xA.y + xB.y;
    double m = 1.0/(b - a*cm1);
    cm1 = c*m;
    cs[j*NXSIZE/NSTEPS*NZ] = c*m;
    dM.x = (xD.x - a*dM.x)*m;
    dM.y = (xD.y - a*dM.y)*m;
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
  xD.x = 0.0;
  xD.y = 0.0;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
  for(int j=NY-2; j>=0; j--){
    xD.x = d[j*NXSIZE/NSTEPS*NZ].x - cs[j*NXSIZE/NSTEPS*NZ]*xD.x;
    xD.y = d[j*NXSIZE/NSTEPS*NZ].y - cs[j*NXSIZE/NSTEPS*NZ]*xD.y;
    x[j*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
    x[j*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
    //x[j*NXSIZE/NSTEPS*NZ].x = (float)(d[j*NXSIZE/NSTEPS*NZ].x - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].x);
    //x[j*NXSIZE/NSTEPS*NZ].y = (float)(d[j*NXSIZE/NSTEPS*NZ].y - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].y);
  }

}

__global__ void isolve_ep_t(float2* __restrict x, const double2* __restrict AB, /*const double* __restrict B,*/ const double2* __restrict alphabetha, /*const double* __restrict betha,*/
                           double2* __restrict d, double* __restrict cs, const double D, const int nstep, domain_t domain){

  int ik = threadIdx.x + blockDim.x*blockIdx.x;

  if(ik>=NXSIZE/NSTEPS*NZ) return;

  x  += ik;
  d  += ik;
  cs += ik;

  int i = ik/NZ;
  int k = ik%NZ;
  
  int stride=nstep*NXSIZE/NSTEPS;
  double k1=(i+IGLOBAL+stride)<NX/2 ? (double)(i+IGLOBAL+stride) : (double)(i+IGLOBAL+stride)-(double)NX ;
  double k3=(double)k;
  k1=(PI2/LX)*k1;
  k3=(PI2/LZ)*k3;
  double kk=k1*k1+k3*k3;

  double2 xA, xB, xC;
  xA.x = (double)x[0*NXSIZE/NSTEPS*NZ].x;
  xA.y = (double)x[0*NXSIZE/NSTEPS*NZ].y;
  xB.x = (double)x[1*NXSIZE/NSTEPS*NZ].x;
  xB.y = (double)x[1*NXSIZE/NSTEPS*NZ].y;
  xC.x = (double)x[2*NXSIZE/NSTEPS*NZ].x;
  xC.y = (double)x[2*NXSIZE/NSTEPS*NZ].y;

  double2 xD;
  //xD.x = A[0]*xA.x + B[0]*xB.x + C[0]*xC.x;
  //xD.y = A[0]*xA.y + B[0]*xB.y + C[0]*xC.y;

  double2 dM;
  dM.x = 0.0;//xD.x/b[0];
  dM.y = 0.0;//xD.y/b[0];
  d[0] = dM;
  double cm1 = 0.0;
  cs[0] = 0.0;

  for(int j=1; j<NY-1; j++){
    double C = -AB[j].x-AB[j].y;
    double a=alphabetha[j].y-D*AB[j].y+D*kk*alphabetha[j].y;
    double b=1.0     -D*C   +D*kk;
    double c=alphabetha[j].x-D*AB[j].x+D*kk*alphabetha[j].x;
    if(j==1   ) a=0.0;
    if(j==NY-2) c=0.0;
    xD.x = alphabetha[j].x*xC.x + alphabetha[j].y*xA.x + xB.x;
    xD.y = alphabetha[j].x*xC.y + alphabetha[j].y*xA.y + xB.y;
    double m = 1.0/(b - a*cm1);
    cm1 = c*m;
    cs[j*NXSIZE/NSTEPS*NZ] = c*m;
    dM.x = (xD.x - a*dM.x)*m;
    dM.y = (xD.y - a*dM.y)*m;
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
  xD.x = 0.0;
  xD.y = 0.0;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
  for(int j=NY-2; j>=0; j--){
    xD.x = d[j*NXSIZE/NSTEPS*NZ].x - cs[j*NXSIZE/NSTEPS*NZ]*xD.x;
    xD.y = d[j*NXSIZE/NSTEPS*NZ].y - cs[j*NXSIZE/NSTEPS*NZ]*xD.y;
    x[j*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
    x[j*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
    //x[j*NXSIZE/NSTEPS*NZ].x = (float)(d[j*NXSIZE/NSTEPS*NZ].x - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].x);
    //x[j*NXSIZE/NSTEPS*NZ].y = (float)(d[j*NXSIZE/NSTEPS*NZ].y - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].y);
  }

}

__global__ void isolve_bilap_ep_t(float2* __restrict x, const double2* __restrict AB, /*const double* __restrict B,*/ const double2* __restrict alphabetha, /*const double* __restrict betha,*/
                           double2* __restrict d, double* __restrict cs, const double D, const int nstep, domain_t domain){

  int ik = threadIdx.x + blockDim.x*blockIdx.x;

  if(ik>=NXSIZE/NSTEPS*NZ) return;

  x  += ik;
  d  += ik;
  cs += ik;

  int i = ik/NZ;
  int k = ik%NZ;
  
  int stride=nstep*NXSIZE/NSTEPS;
  double k1=(i+IGLOBAL+stride)<NX/2 ? (double)(i+IGLOBAL+stride) : (double)(i+IGLOBAL+stride)-(double)NX ;
  double k3=(double)k;
  k1=(PI2/LX)*k1;
  k3=(PI2/LZ)*k3;
  double kk=k1*k1+k3*k3;

  double2 xA, xB, xC;/*
  xA.x = (double)x[0*NXSIZE/NSTEPS*NZ].x;
  xA.y = (double)x[0*NXSIZE/NSTEPS*NZ].y;
  xB.x = (double)x[1*NXSIZE/NSTEPS*NZ].x;
  xB.y = (double)x[1*NXSIZE/NSTEPS*NZ].y;
  xC.x = (double)x[2*NXSIZE/NSTEPS*NZ].x;
  xC.y = (double)x[2*NXSIZE/NSTEPS*NZ].y;
  */

	xA.x = 0.0;
	xA.y = 0.0;
	xB.x = 0.0;      
	xB.y = 0.0;
	xC.x = 0.0;
	xC.y = 0.0;


  double2 xD;
  //xD.x = A[0]*xA.x + B[0]*xB.x + C[0]*xC.x;
  //xD.y = A[0]*xA.y + B[0]*xB.y + C[0]*xC.y;

  double2 dM;
  dM.x = 1.0;//xD.x/b[0];
  dM.y = 1.0;//xD.y/b[0];
  d[0] = dM;
  double cm1 = 0.0;
  cs[0] = 0.0;


  for(int j=1; j<NY-1; j++){
    double C = -AB[j].x-AB[j].y;
    double a=alphabetha[j].y-D*AB[j].y+D*kk*alphabetha[j].y;
    double b=               -D*C      +D*kk;
    double c=alphabetha[j].x-D*AB[j].x+D*kk*alphabetha[j].x;
    //if(j==1   ) a=0.0;
    //if(j==NY-2) c=0.0;
    xD.x =0.0;// alphabetha[j].x*xC.x + alphabetha[j].y*xA.x + xB.x;
    xD.y =0.0;// alphabetha[j].x*xC.y + alphabetha[j].y*xA.y + xB.y;
    double m = 1.0/(b - a*cm1);
    cm1 = c*m;
    cs[j*NXSIZE/NSTEPS*NZ] = c*m;
    dM.x = (xD.x - a*dM.x)*m;
    dM.y = (xD.y - a*dM.y)*m;
    d[j*NXSIZE/NSTEPS*NZ] = dM;
    /*if(j<NY-2){
			xA.x = 0.0;//xB;
			xA.y = 0.0;
			xB.x = 0.0;      
			xB.y = 0.0;//xC;
      //float2 next_x = x[(j+2)*NXSIZE/NSTEPS*NZ];
      xC.x = 0.0;//(double)next_x.x;
      xC.y = 0.0;//(double)next_x.y;
     }*/
  }
/*
  double m1 = 1.0/(b[NY-2] - a[NY-2]*c[NY-3]);
  xD.x = A[NY-2]*xC.x + B[NY-2]*xA.x + C[NY-2]*xB.x;
  xD.y = A[NY-2]*xC.y + B[NY-2]*xA.y + C[NY-2]*xB.y;
  d[(NY-2)*NXSIZE/NSTEPS*NZ].x = (xD.x - a[NY-2]*d[(NY-3)*NXSIZE/NSTEPS*NZ].x)*m1;
  d[(NY-2)*NXSIZE/NSTEPS*NZ].y = (xD.y - a[NY-2]*d[(NY-3)*NXSIZE/NSTEPS*NZ].y)*m1;
*/
  xD.x = 0.0;
  xD.y = 0.0;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
  x[(NY-1)*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
  for(int j=NY-2; j>=0; j--){
    xD.x = d[j*NXSIZE/NSTEPS*NZ].x - cs[j*NXSIZE/NSTEPS*NZ]*xD.x;
    xD.y = d[j*NXSIZE/NSTEPS*NZ].y - cs[j*NXSIZE/NSTEPS*NZ]*xD.y;
    x[j*NXSIZE/NSTEPS*NZ].x = (float)xD.x;
    x[j*NXSIZE/NSTEPS*NZ].y = (float)xD.y;
    //x[j*NXSIZE/NSTEPS*NZ].x = (float)(d[j*NXSIZE/NSTEPS*NZ].x - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].x);
    //x[j*NXSIZE/NSTEPS*NZ].y = (float)(d[j*NXSIZE/NSTEPS*NZ].y - c[j]*(double)x[(j+1)*NXSIZE/NSTEPS*NZ].y);
  }

}




extern void implicitSolver_double(float2* u,float betha,float dt, domain_t domain){
START_RANGE("implicitSolver_double",20)
#ifndef USE_CUSPARSE
  trans_yzx_to_zxy(u/*+i*NXSIZE/NSTEPS*NZ*/, aux_dev[0], 0, domain);

  dim3 grid,block;
  block.x=64;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;

  isolve_ep_t<<<grid,block>>>(aux_dev[0],AB_c,/*B_c,*/alphabetha_c,/*betha_c,*/AUX,(double*)aux_dev[1],(1.0/REYNOLDS)*(betha*dt),0,domain);

  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);
#else
  //SIZE OF LDIAG CDIAG UDIAG AND AUX
  //2*SIZE/NSTEPS

  for(int i=0;i<NSTEPS;i++){
    setDiagkernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG,dt*betha,i,domain);

    rhs_A_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY,domain);
    kernelCheck(RET,domain,"hemholz");	

    cusparseCheck(cusparseZgtsvStridedBatch(implicit_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),domain,"HEM");

    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX,domain);
  }
/*
  CHECK_CUDART( cudaMemcpy(aux_host_1[0],u         ,NXSIZE*NZ*NY*sizeof(float2), cudaMemcpyDeviceToHost) );
  CHECK_CUDART( cudaMemcpy(aux_host_1[1],aux_dev[1],NXSIZE*NZ*NY*sizeof(float2), cudaMemcpyDeviceToHost) );
  int err_count=0;
  for(int i=0; i<NXSIZE; i++)
  for(int k=0; k<NZ    ; k++)
  for(int j=0; j<NY    ; j++){
    int h = i*NY*NZ + k*NY + j;
    float2 gold = aux_host_1[0][h];
    float2 mine = aux_host_1[1][h];
    if(gold.x != mine.x || gold.y != mine.y){
      err_count++;
      if(err_count<64&&RANK==0) printf("mis-match u[%d][%d][%d] = %g + %g , mine = %g + %g \n",i,k,j,gold.x,gold.y,mine.x,mine.y);
    }
  }
  if(RANK==0) printf("%d of %d points have errors %d match\n",err_count,NXSIZE*NZ*NY,NXSIZE*NZ*NY-err_count); 
  exit(1);
*/
#endif
END_RANGE
  return;

}


extern void hemholztSolver_double(float2* u, domain_t domain){
START_RANGE("hemholztSolver_double",5)
#ifndef USE_CUSPARSE
  trans_yzx_to_zxy(u/*+i*NXSIZE/NSTEPS*NZ*/, aux_dev[0], 0, domain);
  dim3 grid,block;
  block.x=64;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;
  hsolve_ep_t<<<grid,block>>>(aux_dev[0],AB_c,/*B_c,*/alphabetha_c,/*betha_c,*/AUX,(double*)aux_dev[1],0,domain);
  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);
#else
  //SIZE OF LDIAG CDIAG UDIAG AND AUX
  //2*SIZE/NSTEPS

  for(int i=0;i<NSTEPS;i++){

    setDiagkernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG,i,domain);

    rhs_A_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY,domain);
    kernelCheck(RET,domain,"hemholz");

    cusparseCheck(cusparseZgtsvStridedBatch(hemholzt_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),
                  domain,
                  "HEM");

    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX,domain);

  }
#endif
END_RANGE
  return;
}

extern void implicitSolver_double_bilaplacian(float2* u,float betha,float dt, domain_t domain){

  //Solves the implicit step with boundary conditions [1,0] at the wall and RHS equal to zero	

  //SIZE OF LDIAG CDIAG UDIAG AND AUX
  //2*SIZE/NSTEPS
#ifndef USE_CUSPARSE
  //trans_yzx_to_zxy(u/*+i*NXSIZE/NSTEPS*NZ*/, aux_dev[0], 0, domain);

  dim3 grid,block;
  block.x=64;
  grid.x=(NXSIZE/NSTEPS*NZ + block.x - 1)/block.x;

  //Solves (1-behta*dt/REYNOLDS*LAP)Phi=0 with boundary conditions [1,0]
  isolve_bilap_ep_t<<<grid,block>>>(aux_dev[0],AB_c,/*B_c,*/alphabetha_c,/*betha_c,*/AUX,(double*)aux_dev[1],(1.0/REYNOLDS)*(betha*dt),0,domain);

  trans_zxy_to_yzx(aux_dev[0], u, 0, domain);

#else
  for(int i=0;i<NSTEPS;i++){

    setDiagkernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG,dt*betha,i,domain);

    rhs_A_kernel_bilaplacian<<<blocksPerGrid,threadsPerBlock>>>(AUX,domain);
    kernelCheck(RET,domain,"hemholz");	

    cusparseZgtsvStridedBatch(implicit_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY);

    cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX,domain);

  }

#endif
  return;

}





