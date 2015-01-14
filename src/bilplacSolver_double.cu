#include "channel.h"

//FUNCTIONS


static __global__ void setBoundaryCond(float2* ddu, float2* u,float2* du,double betha,double dt,int IGLOBAL)

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
	

    double l1;
    double l2;

    l1=sqrt(kk+REYNOLDS/(dt*betha));	
    l2=sqrt(kk);
		
    double D=(l1*l1-l2*l2);
    D=1.0/D;

    double y;

    //ONly first thread computes derivatives and constants

    if(j==0){

      float2 dv_mf;
      float2 dv_pf;

      double2 dv_m;
      double2 dv_p;
	
      double dv1_p;
      double dv1_m;
	
      double dv2_p;
      double dv2_m;

      //Calc derivatives at boundaries and set C1 and C2
	
      dv_mf=du[i*NZ*NY+k*NY];		
      dv_pf=du[i*NZ*NY+k*NY+NY-1];

      dv_m.x=(double)(dv_mf.x);
      dv_m.y=(double)(dv_mf.y);

      dv_p.x=(double)(dv_pf.x);
      dv_p.y=(double)(dv_pf.y);

      //Calc analitic solution	
		
      //////////DERIVATIVES AT y=1.0///////////////	
	
      y=1.0;
		
		
      dv1_p =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))-(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
      dv1_p+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))+(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));
			
      dv2_p =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))+(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
      dv2_p+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))-(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));
		

      /*
	dv1_p =0.5*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));
	dv2_p =0.5*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));
      */

      //////////DERIVATIVES AT y=-1.0///////////////
      y=-1.0;

		
      dv1_m =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))-(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
      dv1_m+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))+(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));

      dv2_m =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))+(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
      dv2_m+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))-(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));	
		
	
      /*
	dv1_m =0.5*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));
	dv2_m =0.5*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));		
      */

      //Solución 
		
      det.x=dv1_m*dv2_p-dv1_p*dv2_m;
      det.y=dv1_m*dv2_p-dv1_p*dv2_m;	
	
      //C1
	
      c1.x=dv2_p*dv_m.x-dv2_m*dv_p.x;
      c1.y=dv2_p*dv_m.y-dv2_m*dv_p.y;
	
      c1.x=-c1.x/(det.x);
      c1.y=-c1.y/(det.y);

      //C2	
	
      c2.x=-dv1_p*dv_m.x+dv1_m*dv_p.x;
      c2.y=-dv1_p*dv_m.y+dv1_m*dv_p.y;
	
      c2.x=-c2.x/(det.x);
      c2.y=-c2.y/(det.y);

      //write to shared memory	
	
      C1_S=c1;	
      C2_S=c2;
		
    }
			

    __syncthreads();

		
    double v1,v2;	
    double dv1,dv2;
    double ddv1,ddv2;

    float2 ddu_kf;
    float2 du_kf;	
    float2 u_kf;

    double2 ddu_k;
    double2 du_k;	
    double2 u_k;
	
			
    //FOR THE MESH
    y=Fmesh(j*DELTA_Y-1.0);

    //V1

		
    v1 =0.5*D*( (expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))-(expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
    v1+=0.5*D*(-(expf(-l2*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))+(expf(-l2*(1.0-y))-exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));
		
	
    //v1=0.5*D*(CC(y,l1)-SS(y,l1)-CC(y,l2)+SS(y,l2));
	
    //V2		

		
    v2 =0.5*D*( (expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))+(expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
    v2+=0.5*D*(-(expf(-l2*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))-(expf(-l2*(1.0-y))-exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));		
		

    //v2=0.5*D*(CC(y,l1)+SS(y,l1)-CC(y,l2)-SS(y,l2));

    //DV1

		
    dv1 =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))-(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
    dv1+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))+(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));
		

    //v1=0.5*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));	

    //DV2		

		
    dv2 =0.5*D*l1*( (expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))+(expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
    dv2+=0.5*D*l2*(-(expf(-l2*(1.0-y))-exp(-l1*(1.0+y)))/(1.0+exp(-l2*2.0))-(expf(-l2*(1.0-y))+exp(-l2*(1.0+y)))/(1.0-exp(-l2*2.0)));	
		
	
    //v2=0.5*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));	
		

    //DDV1 and DDV2
		
		
    ddv1=0.5*( (expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))-(expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
    ddv2=0.5*( (expf(-l1*(1.0-y))+exp(-l1*(1.0+y)))/(1.0+exp(-l1*2.0))+(expf(-l1*(1.0-y))-exp(-l1*(1.0+y)))/(1.0-exp(-l1*2.0)));
		
		
    //ddv1=0.5*(CC(y,l1)-SS(y,l1));
    //ddv2=0.5*(CC(y,l1)+SS(y,l1));

    //Read		
    //Not reading boundary conditions
		

    ddu_kf=ddu[h];
    du_kf=du[h];
    u_kf=u[h];
		
    ddu_k.x=(double)(ddu_kf.x);
    ddu_k.y=(double)(ddu_kf.y);
	
    du_k.x=(double)(du_kf.x);
    du_k.y=(double)(du_kf.y);
		
    u_k.x=(double)(u_kf.x);
    u_k.y=(double)(u_kf.y);
		 
    //ddu_k.x=ddu_k.x+C1_S.x*(ddv1-kk*v1)+C2_S.x*(ddv2-kk*v2);
    //ddu_k.y=ddu_k.y+C1_S.y*(ddv1-kk*v1)+C2_S.y*(ddv2-kk*v2);

    ddu_k.x=ddu_k.x+C1_S.x*ddv1+C2_S.x*ddv2;
    ddu_k.y=ddu_k.y+C1_S.y*ddv1+C2_S.y*ddv2;	

    du_k.x=du_k.x+C1_S.x*dv1+C2_S.x*dv2;
    du_k.y=du_k.y+C1_S.y*dv1+C2_S.y*dv2;

    u_k.x=u_k.x+C1_S.x*v1+C2_S.x*v2;
    u_k.y=u_k.y+C1_S.y*v1+C2_S.y*v2;
	
    if(i+IGLOBAL==0 & k==0){

      u_k.x=0.0;	
      u_k.y=0.0;
	
      du_k.x=0.0;
      du_k.y=0.0;	

      ddu_k.x=0.0;
      ddu_k.y=0.0;
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



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


static void setBoundaries(float2* ddv,float2* v,float2* dv,float betha,float dt, domain_t domain){
		
  threadsPerBlock.x=NY;
  threadsPerBlock.y=1;

  blocksPerGrid.x=NZ;
  blocksPerGrid.y=NXSIZE;
	
  //Boundary conditions may have problems if dt too small

  if(dt>1e-8)
    setBoundaryCond<<<blocksPerGrid,threadsPerBlock>>>(ddv,v,dv,betha,dt,domain.iglobal);
  kernelCheck(RET,domain,"Boundary");

}

extern void bilaplaSolver_double(float2* ddv, float2* v, float2* dv, float betha,float dt, domain_t domain){


  //Implicit time step
  //Solves (1-0.5*dt*LAP)ddv_w=rhs with ddv(+-1)=0
  //rhs stored in ddv_w;

  implicitSolver_double(ddv,betha,dt,domain);
	
  //Copy ddv--->v
  cudaCheck(cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");

  //Solves LAP*v=ddv_w;
  //Solver hemholzt

  hemholztSolver_double(v,domain);
	
  //Copy ddv--->v
  cudaCheck(cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");
  deriv_Y_HO_double(dv,domain);

  //Impose boundary conditions 
	
  setBoundaries(ddv,v,dv,betha,dt,domain);
	

  return;

}


