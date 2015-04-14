#include "channel.h"

//FUNCTIONS


#define BNDBLK (64)

static __global__ void setBoundaryCond(float2* ddu, float2* u,float2* du,double betha,double dt,domain_t domain)

{  

  __shared__ double2 C1_S;
  __shared__ double2 C2_S;

  int k   = blockIdx.x;
  int i   = blockIdx.y;
  int j   = threadIdx.x-1;
  if(threadIdx.x==0) j=NY-1;

  int h=i*NZ*NY+k*NY;//+j;

//  if(i<NXSIZE & k<NZ & j<NY){

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

    double e2l1 = exp(-2.0*l1);
    double e2l2 = exp(-2.0*l2);
    double o1me2l1 = 1.0/(1.0-e2l1);
    double o1pe2l1 = 1.0/(1.0+e2l1);
    double o1me2l2 = 1.0/(1.0-e2l2);
    double o1pe2l2 = 1.0/(1.0+e2l2);

  __shared__ double dv1_p;
  __shared__ double dv2_p;
  __shared__ double2 dv_p;

  for(int jcount=0; jcount<((NY+BNDBLK-1)/BNDBLK); jcount++){

    float2 ddu_kf;
    float2 du_kf;
    float2 u_kf;

    if(i+IGLOBAL==0 && k==0){
      u_kf.x=0.f;
      u_kf.y=0.f;
      du_kf.x=0.f;
      du_kf.y=0.f;
      ddu_kf.x=0.f;
      ddu_kf.y=0.f;
    }else{

      //FOR THE MESH
      double y=Fmesh(j*DELTA_Y-1.0);
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

      ddu_kf=ddu[h+j];
      du_kf=du[h+j];
      u_kf=u[h+j];

      double2 ddu_k;
      double2 du_k;
      double2 u_k;

      ddu_k.x=(double)(ddu_kf.x);
      ddu_k.y=(double)(ddu_kf.y);
      du_k.x=(double)(du_kf.x);
      du_k.y=(double)(du_kf.y);
      u_k.x=(double)(u_kf.x);
      u_k.y=(double)(u_kf.y);

      if(jcount==0){
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

      }

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
	
    }//if(i+IGLOBAL==0 && k==0)		
	
    //Write
    //it does no give v or dv
    ddu[h+j]=ddu_kf;
    du[h+j]=du_kf;
    u[h+j]=u_kf;
   
    j=(j+BNDBLK)%NY;	

  }

}

static __global__ void setBoundaryCond_numerical(float2* ddu, float2* u,float2* du,float2* ddu_w, float2* u_w,float2* du_w,domain_t domain)
{  

  //Define shared memory

  //__shared__ double2 sf[NY];
  __shared__ double2 C1_S;
  __shared__ double2 C2_S;

  int k   = blockIdx.x;
  int i   = blockIdx.y;

  int j   = threadIdx.x;


  int h=i*NZ*NY+k*NY+j;


  if(i<NXSIZE & k<NZ & j<NY){

    //Read from global to shared

    //Calc constants

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

    //Calc constants
	
    double2 det;

    double2 c1;
    double2 c2;

   

    //ONly first thread computes derivatives and constants

    if(j==0){

      float2 dv_mf;
      float2 dv_pf;

      float2 dv1_pf;
      float2 dv1_mf;

      float2 dv2_pf;
      float2 dv2_mf;

      double2 dv_m;
      double2 dv_p;
	
      double2 dv1_p;
      double2 dv1_m;
	
      double2 dv2_p;
      double2 dv2_m;

      //Calc derivatives at boundaries and set C1 and C2
	
      dv_mf=du[i*NZ*NY+k*NY];	//Derivative at y=-1	
      dv_pf=du[i*NZ*NY+k*NY+NY-1]; //Derivative at y=1

      dv_m.x=(double)(dv_mf.x);
      dv_m.y=(double)(dv_mf.y);

      dv_p.x=(double)(dv_pf.x);
      dv_p.y=(double)(dv_pf.y);

      //Read derivatives at wall for homogeneous solution 1: Phi(y=1)=0 Phi(y=-1)=1 

      dv1_mf=du_w[i*NZ*NY+k*NY];	        //Derivative at y=-1	
      dv1_pf=du_w[i*NZ*NY+k*NY+NY-1];    //Derivative at y=1

      dv1_m.x=(double)(dv1_mf.x);
      dv1_m.y=(double)(dv1_mf.y);

      dv1_p.x=(double)(dv1_pf.x);      
      dv1_p.y=(double)(dv1_pf.y); 

      //Read derivatives at wall for homogeneous solution 2: Phi(y=1)=1 Phi(y=-1)=0     

      dv2_m.x=-dv1_p.x;
      dv2_m.y=-dv1_p.y;

      dv2_p.x=-dv1_m.x;     
      dv2_p.y=-dv1_m.y; 	
	
		
      //Solves: 
      // 0=dv_m+c1*dv1_m+c2*dv2_m;
      // 0=dv_p+c1*dv1_p+c2*dv2_p;	
      // c1 and c2 so that derivative at the wall is zero	 
		
      det.x=dv1_m.x*dv2_p.x-dv1_p.x*dv2_m.x;
      det.y=dv1_m.y*dv2_p.y-dv1_p.y*dv2_m.y;	
	
      //C1
	
      c1.x=dv2_p.x*dv_m.x-dv2_m.x*dv_p.x;
      c1.y=dv2_p.y*dv_m.y-dv2_m.y*dv_p.y;
	
      c1.x=-c1.x/(det.x);
      c1.y=-c1.y/(det.y);

      //C2	
	
      c2.x=-dv1_p.x*dv_m.x+dv1_m.x*dv_p.x;
      c2.y=-dv1_p.y*dv_m.y+dv1_m.y*dv_p.y;
	
      c2.x=-c2.x/(det.x);
      c2.y=-c2.y/(det.y);

      //write to shared memory	
	

      C1_S=c1;	
      C2_S=c2;
		
    }
			

    __syncthreads();

	
   __shared__ double2 ddv_w[NY];
   __shared__ double2 dv_w[NY];
   __shared__ double2 v_w[NY];


   //READ FROM GLOBAL TO SHARED
   //Homogeneous solutions 


    ddv_w[j].x=(double)ddu_w[h].x;
    ddv_w[j].y=(double)ddu_w[h].y;

    dv_w[j].x=(double)du_w[h].x;
    dv_w[j].y=(double)du_w[h].y;

    v_w[j].x=(double)u_w[h].x;
    v_w[j].y=(double)u_w[h].y;

 	__syncthreads();

    //Read v ddv and dv
    double2 ddu_k;
    double2 du_k;
    double2 u_k;

    float2 ddu_kf;
    float2 du_kf;
    float2 u_kf;

    ddu_kf=ddu[h];	
    du_kf=du[h];	
    u_kf=u[h];	

    u_k.x=(double)u_kf.x;
    u_k.y=(double)u_kf.y;

    du_k.x=(double)du_kf.x;
    du_k.y=(double)du_kf.y;
    
    ddu_k.x=(double)ddu_kf.x;
    ddu_k.y=(double)ddu_kf.y;


   //Impose solution that has derivative zero at the wall
   
    ddu_k.x=ddu_k.x+C1_S.x*ddv_w[j].x+C2_S.x*ddv_w[NY-1-j].x;
    ddu_k.y=ddu_k.y+C1_S.y*ddv_w[j].y+C2_S.y*ddv_w[NY-1-j].y;	

    du_k.x=du_k.x+C1_S.x*dv_w[j].x-C2_S.x*dv_w[NY-1-j].x;
    du_k.y=du_k.y+C1_S.y*dv_w[j].y-C2_S.y*dv_w[NY-1-j].y;

    u_k.x=u_k.x+C1_S.x*v_w[j].x+C2_S.x*v_w[NY-1-j].x;
    u_k.y=u_k.y+C1_S.y*v_w[j].y+C2_S.y*v_w[NY-1-j].y;
	
    if(i+IGLOBAL==0 && k==0){

      u_k.x=0.0;	
      u_k.y=0.0;
	
      du_k.x=0.0;
      du_k.y=0.0;	

      ddu_k.x=0.0;
      ddu_k.y=0.0;
    }		

    //Write to disk	

    ddu_kf.x=(float)(ddu_k.x);
    ddu_kf.y=(float)(ddu_k.y);

    du_kf.x=(float)(du_k.x);
    du_kf.y=(float)(du_k.y);
	
    u_kf.x=(float)(u_k.x);
    u_kf.y=(float)(u_k.y);

    ddu[h]=ddu_kf;
    du[h]=du_kf;
    u[h]=u_kf;
	
 	
  }

}




//FUNCTIONS



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

static dim3 threadsPerBlock_B;
static dim3 blocksPerGrid_B;


static void setBoundaries_analytic(float2* ddv,float2* v,float2* dv,float betha,float dt, domain_t domain){
START_RANGE("setBoundaries",37)		
  threadsPerBlock.x=BNDBLK;//NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE;
	
  //Boundary conditions may have problems if dt too small

  if(dt>1e-8)
    setBoundaryCond<<<blocksPerGrid,threadsPerBlock>>>(ddv,v,dv,betha,dt,domain);
  kernelCheck(RET,domain,"Boundary");
END_RANGE
}
/*
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
*/

static void setBoundaries_numerical(float2* ddv,float2* v,float2* dv,float2* ddv_w,float2* v_w,float2* dv_w, domain_t domain){
		
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE;
	
  //Boundary conditions may have problems if dt too small

  
  setBoundaryCond_numerical<<<blocksPerGrid,threadsPerBlock>>>(ddv,v,dv,ddv_w,v_w,dv_w,domain);


}

extern void bilaplaSolver_double(float2* ddv, float2* v, float2* dv, float2* ddv_w,float2* v_w,float2* dv_w,float betha,float dt, domain_t domain){


  //Implicit time step
  //Solves (1-0.5*dt*LAP)ddv_w=rhs with ddv(+-1)=0
  //rhs stored in ddv_w;

  implicitSolver_double(ddv,betha,dt,domain);
	
  //Copy ddv--->v
  CHECK_CUDART(cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice));

  //Solves LAP*v=ddv_w;
  //Solver hemholzt

  hemholztSolver_double(v,domain);
	
  //Copy ddv--->v
  CHECK_CUDART(cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice));
  deriv_Y_HO_double(dv,domain);

  //SOLVE SECOND BILAPLACIAN BOUNDARY CONDITION ddv(+1)=1 ddv(-1)=0

  implicitSolver_double_bilaplacian(ddv_w,betha,dt,domain);
	
  //Copy ddv--->v
  CHECK_CUDART(cudaMemcpy(v_w,ddv_w,SIZE,cudaMemcpyDeviceToDevice));

  //Solves LAP*v=ddv_w;
  //Solver hemholzt

  hemholztSolver_double(v_w,domain);
	
  //Copy ddv--->v
  CHECK_CUDART(cudaMemcpy(dv_w,v_w,SIZE,cudaMemcpyDeviceToDevice));
  deriv_Y_HO_double(dv_w,domain);

  //Impose boundary conditions 
	
  setBoundaries_numerical(ddv,v,dv,ddv_w,v_w,dv_w,domain);
  //setBoundaries_analytic(ddv,v,dv,betha,dt,domain);
  	

  return;

}




