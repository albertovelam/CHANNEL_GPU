#include "channel.h"

//FUNCTIONS


static __global__ void setBoundaryCond(float2* ddu, float2* u,float2* du,double betha,double dt,domain_t domain)

{  

  __shared__ double2 C1_S;
  __shared__ double2 C2_S;

  int k   = blockIdx.x;
  int i   = blockIdx.y;
  int j   = threadIdx.x;
  int h=i*NZ*NY+k*NY+j;

  if(i<NXSIZE & k<NZ & j<NY){

    double k1;
    double k3;
    double kk;		
    // X indices		
    k1=(i+IGLOBAL)<NX/2 ? (double)(i+IGLOBAL) : (double)(i+IGLOBAL)-(double)NX ;
    // Z indices
    k3=(double)k;
    //Fraction
    k1=(PI2/LX)*k1;
    k3=(PI2/LZ)*k3;	

    kk=k1*k1+k3*k3;

    double l1=sqrt(kk+REYNOLDS/(dt*betha));	
    double l2=sqrt(kk);
    double D=(l1*l1-l2*l2);
    D=1.0/D;

    //FOR THE MESH
    double y=Fmesh(j*DELTA_Y-1.0);
    
    double e2l1 = exp(-2.0*l1);
    double e2l2 = exp(-2.0*l2);
    double o1me2l1 = 1.0/(1.0-e2l1);
    double o1pe2l1 = 1.0/(1.0+e2l1);
    double o1me2l2 = 1.0/(1.0-e2l2);
    double o1pe2l2 = 1.0/(1.0+e2l2);
    double e1my = exp(-l1*(1.0-y));
    double e2my = exp(-l2*(1.0-y));
    double e1py = exp(-l1*(1.0+y));
    double e2py = exp(-l2*(1.0+y));

    double v1   = 0.5*D*(     (e1my + e1py)*o1pe2l1 - (e1my - e1py)*o1me2l1        - (e2my + e2py)*o1pe2l2 + (e2my - e2py)*o1me2l2   );
    double v2   = 0.5*D*(     (e1my + e1py)*o1pe2l1 + (e1my - e1py)*o1me2l1        - (e2my + e2py)*o1pe2l2 - (e2my - e2py)*o1me2l2   );
    double dv1  = 0.5*D*( l1*((e1my - e1py)*o1pe2l1 - (e1my + e1py)*o1me2l1) + l2*(- (e2my - e2py)*o1pe2l2 + (e2my + e2py)*o1me2l2 ) );
    double dv2  = 0.5*D*( l1*((e1my - e1py)*o1pe2l1 + (e1my + e1py)*o1me2l1) + l2*(- (e2my - e2py)*o1pe2l2 - (e2my + e2py)*o1me2l2 ) );
    double ddv1 = 0.5*  (     (e1my + e1py)*o1pe2l1 - (e1my - e1py)*o1me2l1                                                                );
    double ddv2 = 0.5*  (     (e1my + e1py)*o1pe2l1 + (e1my - e1py)*o1me2l1                                                                );
    float2 ddu_kf=ddu[h];
    float2 du_kf=du[h];
    float2 u_kf=u[h];

    double2 ddu_k;
    double2 du_k;
    double2 u_k;

    ddu_k.x=(double)(ddu_kf.x);
    ddu_k.y=(double)(ddu_kf.y);
    du_k.x=(double)(du_kf.x);
    du_k.y=(double)(du_kf.y);
    u_k.x=(double)(u_kf.x);
    u_k.y=(double)(u_kf.y);

  __shared__ double dv1_p;
  __shared__ double dv2_p;
  __shared__ double2 dv_p;

  //y==1 boundary
  if(j==NY-1){
    dv1_p = dv1;
    dv2_p = dv2;
    dv_p  = du_k;
  }

  __syncthreads();

  if(j==0){
    double det=dv1*dv2_p-dv1_p*dv2;
    double odet = 1.0/det;
    double2 c1;
    double2 c2;
    c1.x=dv2_p*du_k.x-dv2*dv_p.x;
    c1.y=dv2_p*du_k.y-dv2*dv_p.y;
    c1.x=-c1.x*odet;///det;
    c1.y=-c1.y*odet;///det;
    c2.x=-dv1_p*du_k.x+dv1*dv_p.x;
    c2.y=-dv1_p*du_k.y+dv1*dv_p.y;
    c2.x=-c2.x*odet;///det;
    c2.y=-c2.y*odet;///det;
    C1_S=c1;
    C2_S=c2;
  }  

  __syncthreads();

    ddu_k.x=ddu_k.x+C1_S.x*ddv1+C2_S.x*ddv2;
    ddu_k.y=ddu_k.y+C1_S.y*ddv1+C2_S.y*ddv2;	

    du_k.x=du_k.x+C1_S.x*dv1+C2_S.x*dv2;
    du_k.y=du_k.y+C1_S.y*dv1+C2_S.y*dv2;

    u_k.x=u_k.x+C1_S.x*v1+C2_S.x*v2;
    u_k.y=u_k.y+C1_S.y*v1+C2_S.y*v2;

    ddu_kf.x=(float)(ddu_k.x);
    ddu_kf.y=(float)(ddu_k.y);

    du_kf.x=(float)(du_k.x);
    du_kf.y=(float)(du_k.y);

    u_kf.x=(float)(u_k.x);
    u_kf.y=(float)(u_k.y);

	
    if(i+IGLOBAL==0 && k==0){
      u_kf.x=0.f;	
      u_kf.y=0.f;
      du_kf.x=0.f;
      du_kf.y=0.f;	
      ddu_kf.x=0.f;
      ddu_kf.y=0.f;
    }		

	
    //Write

    //it does no give v or dv

    /*
      ddu_kf.x=__double2float_rn(ddu_k.x);
      ddu_kf.y=__double2float_rn(ddu_k.y);

      du_kf.x=__double2float_rn(du_k.x);
      du_kf.y=__double2float_rn(du_k.y);
	
      u_kf.x=__double2float_rn(u_k.x);
      u_kf.y=__double2float_rn(u_k.y);
    */


    ddu[h]=ddu_kf;
    du[h]=du_kf;
    u[h]=u_kf;
	
 	
  }

}



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


static void setBoundaries(float2* ddv,float2* v,float2* dv,float betha,float dt, domain_t domain){
START_RANGE("setBoundaries",37)		
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE;
	
  //Boundary conditions may have problems if dt too small

  if(dt>1e-8)
    setBoundaryCond<<<blocksPerGrid,threadsPerBlock>>>(ddv,v,dv,betha,dt,domain);
  kernelCheck(RET,domain,"Boundary");
END_RANGE
}

extern void bilaplaSolver_double(float2* ddv, float2* v, float2* dv, float betha,float dt, domain_t domain){
START_RANGE("bilaplaSolver_double",38)

  //Implicit time step
  //Solves (1-0.5*dt*LAP)ddv_w=rhs with ddv(+-1)=0
  //rhs stored in ddv_w;

  implicitSolver_double(ddv,betha,dt,domain);
	
  //Copy ddv--->v
  CHECK_CUDART( cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice) );

  //Solves LAP*v=ddv_w;
  //Solver hemholzt

  hemholztSolver_double(v,domain);
	
  //Copy ddv--->v
  CHECK_CUDART( cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice) );
  deriv_Y_HO_double(dv,domain);

  //Impose boundary conditions 
	
  setBoundaries(ddv,v,dv,betha,dt,domain);
	
END_RANGE
  return;

}


