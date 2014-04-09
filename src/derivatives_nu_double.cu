#include"channel.h"


static __global__ void cast_kernel(float2* u,double2* v)
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

	ud.x=__double2float_rn(vd.x);
	ud.y=__double2float_rn(vd.y);


	u[h]=ud;

	}

}

static __global__ void deriv_Y_kernel(double2* v,float2* u)

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


__global__ void deriv_YY_kernel(double2* v,float2* u)

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

static __global__ void setDiagkernel_Y(double2* ldiag,double2* cdiag,double2* udiag){  

	
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.y * blockDim.y + threadIdx.y;

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

		//To be improved 
		
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

static __global__ void setDiagkernel_YY(double2* ldiag,double2* cdiag,double2* udiag){  

	
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int k = blockIdx.y * blockDim.y + threadIdx.y;

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

		//To be improved 
		
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

static cusparseHandle_t cusparse_handle;

void setDerivativesDouble(void){

	threadsPerBlock.x=NY;
	threadsPerBlock.y=1;

	blocksPerGrid.x=NZ;
	blocksPerGrid.y=NXSIZE/NSTEPS;	

	threadsPerBlock_B.x= THREADSPERBLOCK_IN;
	threadsPerBlock_B.y= THREADSPERBLOCK_IN;

	blocksPerGrid_B.x=NXSIZE/NSTEPS/threadsPerBlock_B.x;
	blocksPerGrid_B.y=NZ*NY/threadsPerBlock_B.y;

	cusparseCheck(cusparseCreate(&cusparse_handle),"Handle");

}


extern void deriv_Y_HO_double(float2* u){



	//SIZE OF LDIAG CDIAG UDIAG AND AUX
	//2*SIZE/NSTEPS

	for(int i=0;i<NSTEPS;i++){
	
	setDiagkernel_Y<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG);

	deriv_Y_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY);
	kernelCheck(RET,"W_kernel");	

	//Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))

	cusparseCheck(cusparseZgtsvStridedBatch(cusparse_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),"HEM");

	cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX);

	}

	return;
}



extern void deriv_YY_HO_double(float2* u){


	//SIZE OF LDIAG CDIAG UDIAG AND AUX
	//2*SIZE/NSTEPS

	for(int i=0;i<NSTEPS;i++){
	
	setDiagkernel_YY<<<blocksPerGrid_B,threadsPerBlock_B>>>(LDIAG,CDIAG,UDIAG);

	deriv_YY_kernel<<<blocksPerGrid,threadsPerBlock>>>(AUX,u+i*NXSIZE/NSTEPS*NZ*NY);
	kernelCheck(RET,"W_kernel");	

	//Requires extra storage size=( 8×(3+NX*NZ)×sizeof(<type>))

	cusparseCheck(cusparseZgtsvStridedBatch(cusparse_handle,NY,LDIAG,CDIAG,UDIAG,AUX,NXSIZE/NSTEPS*NZ,NY),"HEM");

	cast_kernel<<<blocksPerGrid_B,threadsPerBlock_B>>>(u+i*NXSIZE/NSTEPS*NZ*NY,AUX);

	}
	

	return;

}




