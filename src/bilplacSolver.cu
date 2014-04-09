#include "channel.h"

//FUNCTIONS

static __global__ void setBoundaryCond(float2* ddu, float2* u,float2* du,float betha,float dt,int IGLOBAL)

{  

		//Define shared memory

		//__shared__ float2 sf[NY];
		__shared__ float2 C1_S;
		__shared__ float2 C2_S;

		int k   = blockIdx.x;
		int i   = blockIdx.y;

		int j   = threadIdx.x;


		int h=i*NZ*NY+k*NY+j;


		if(i<NXSIZE & k<NZ & j<NY){

		//Read from global to shared

		//Calc constants

		float k1;
		float k3;
		float kk;		
				
		// X indices		
		k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

		// Z indices
		k3=(float)k;

		//Fraction
		k1=(PI2/LX)*k1;
		k3=(PI2/LZ)*k3;	

		kk=k1*k1+k3*k3;

		//Calc constants
	
		float2 det;

		float2 c1;
		float2 c2;
	

		float l1;
		float l2;

		l1=sqrt(kk+REYNOLDS/(dt*betha));	
		l2=sqrt(kk);
		
		float D=(l1*l1-l2*l2);
		D=1.0f/D;

		float y;

		//ONly first thread computes derivatives and constants

		if(j==0){

		float2 dv_m;
		float2 dv_p;
	
		float dv1_p;
		float dv1_m;
	
		float dv2_p;
		float dv2_m;

		//Calc derivatives at boundaries and set C1 and C2
	
		dv_m=du[i*NZ*NY+k*NY];
		dv_p=du[i*NZ*NY+k*NY+NY-1];

		//Calc analitic solution	
		
		//////////DERIVATIVES AT y=1.0f///////////////	
	
		y=1.0f;
		
		
		dv1_p =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))-(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv1_p+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))+(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));
			
		dv2_p =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))+(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv2_p+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))-(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));
		

		/*
		dv1_p =0.5f*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));
		dv2_p =0.5f*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));
		*/

		//////////DERIVATIVES AT y=-1.0f///////////////
		y=-1.0f;

		
		dv1_m =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))-(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv1_m+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))+(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));

		dv2_m =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))+(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv2_m+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))-(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));	
		
	
		/*
		dv1_m =0.5f*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));
		dv2_m =0.5f*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));		
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

		
		float v1,v2;	
		float dv1,dv2;
		float ddv1,ddv2;
		float2 ddu_k;
		float2 du_k;	
		float2 u_k;
	
			
		//FOR THE MESH
		y=Fmesh(j*DELTA_Y-1.0f);

		//V1

		
		v1 =0.5f*D*( (expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))-(expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		v1+=0.5f*D*(-(expf(-l2*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))+(expf(-l2*(1.0f-y))-exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));
		
	
		//v1=0.5f*D*(CC(y,l1)-SS(y,l1)-CC(y,l2)+SS(y,l2));
	
		//V2		

		
		v2 =0.5f*D*( (expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))+(expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		v2+=0.5f*D*(-(expf(-l2*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))-(expf(-l2*(1.0f-y))-exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));		
		

		//v2=0.5f*D*(CC(y,l1)+SS(y,l1)-CC(y,l2)-SS(y,l2));

		//DV1

		
		dv1 =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))-(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv1+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))+(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));
		

		//v1=0.5f*D*( DCC(y,l1)-DSS(y,l1)-DCC(y,l2)+DSS(y,l2));	

		//DV2		

		
		dv2 =0.5f*D*l1*( (expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))+(expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		dv2+=0.5f*D*l2*(-(expf(-l2*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f+exp(-l2*2.0f))-(expf(-l2*(1.0f-y))+exp(-l2*(1.0f+y)))/(1.0f-exp(-l2*2.0f)));	
		
	
		//v2=0.5f*D*( DCC(y,l1)+DSS(y,l1)-DCC(y,l2)-DSS(y,l2));	
		

		//DDV1 and DDV2
		
		
		ddv1=0.5f*( (expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))-(expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		ddv2=0.5f*( (expf(-l1*(1.0f-y))+exp(-l1*(1.0f+y)))/(1.0f+exp(-l1*2.0f))+(expf(-l1*(1.0f-y))-exp(-l1*(1.0f+y)))/(1.0f-exp(-l1*2.0f)));
		
		
		//ddv1=0.5f*(CC(y,l1)-SS(y,l1));
		//ddv2=0.5f*(CC(y,l1)+SS(y,l1));

		//Read		
		//Not reading boundary conditions
		

		ddu_k=ddu[h];
		du_k=du[h];
		u_k=u[h];
		
		
		//ddu_k.x=ddu_k.x+C1_S.x*(ddv1-kk*v1)+C2_S.x*(ddv2-kk*v2);
		//ddu_k.y=ddu_k.y+C1_S.y*(ddv1-kk*v1)+C2_S.y*(ddv2-kk*v2);

		ddu_k.x=ddu_k.x+C1_S.x*ddv1+C2_S.x*ddv2;
		ddu_k.y=ddu_k.y+C1_S.y*ddv1+C2_S.y*ddv2;	

		du_k.x=du_k.x+C1_S.x*dv1+C2_S.x*dv2;
		du_k.y=du_k.y+C1_S.y*dv1+C2_S.y*dv2;

		u_k.x=u_k.x+C1_S.x*v1+C2_S.x*v2;
		u_k.y=u_k.y+C1_S.y*v1+C2_S.y*v2;
	
		if(i+IGLOBAL==0 & k==0){

		u_k.x=0.0f;	
		u_k.y=0.0f;
	
		du_k.x=0.0f;
		du_k.y=0.0f;	

		ddu_k.x=0.0f;
		ddu_k.y=0.0f;
		}		

	
		//Write

		//it does no give v or dv

		ddu[h]=ddu_k;
		du[h]=du_k;
		u[h]=u_k;
	
 	
	  }

}



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


static void setBoundaries(float2* ddv,float2* v,float2* dv,float betha,float dt){
		
	threadsPerBlock.x=NY;
	threadsPerBlock.y=1;

	blocksPerGrid.x=NZ;
	blocksPerGrid.y=NXSIZE;
	
	//Boundary conditions may have problems if dt too small

	if(dt>1e-8)
	setBoundaryCond<<<blocksPerGrid,threadsPerBlock>>>(ddv,v,dv,betha,dt,IGLOBAL);
	kernelCheck(RET,"Boundary");

}

extern void bilaplaSolver(float2* ddv,float2* v,float2* dv,float betha,float dt){


	//Implicit time step
	//Solves (1-0.5f*dt*LAP)ddv_w=rhs with ddv(+-1)=0
	//rhs stored in ddv_w;

	implicitSolver(ddv,betha,dt);
	
	//Copy ddv--->v
	cudaCheck(cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");

	//Solves LAP*v=ddv_w;
	//Solver hemholzt

	hemholztSolver(v);
	
	//Copy ddv--->v
	cudaCheck(cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
	deriv_Y_HO(dv);

	//Impose boundary conditions 
	
	setBoundaries(ddv,v,dv,betha,dt);
	

return;

}


