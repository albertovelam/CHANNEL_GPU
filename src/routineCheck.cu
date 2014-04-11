
#include"channel.h"


void checkHemholzt(void){
	
	//Check hemholz subroutine

	float2* u_host_1;
	float2* u_host_2;
		
	float2* u;
	
	u_host_1=(float2*)malloc(SIZE);
	u_host_2=(float2*)malloc(SIZE);
	
	
	for(int i=0;i<NXSIZE;i++){
	for(int j=0;j<NY;j++){
	for(int k=0;k<NZ;k++){

	int h=i*NY*NZ+k*NY+j;
	
	
	u_host_1[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_1[h].y=0.0f;
	
	u_host_2[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_2[h].y=0.0f;	
	
	}
	}
	}

	cudaCheck(cudaMalloc(&u,SIZE),"malloc");
	cudaCheck(cudaMemcpy(u,u_host_1,SIZE, cudaMemcpyHostToDevice),"MemInfo1");
	
	//Solve Hemholzt	

	hemholztSolver_double(u);
	//hemholztSolver_2(u);

	cudaCheck(cudaMemcpy(u_host_2,u,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");

	

	//Compare with known resutls

	float error=0.0f;
	float error_1;
	float error_2;
	
	float err=0.0f;

	float kx;	
	float kz;

	for(int i=0;i<NXSIZE;i++){
	for(int k=0;k<NZ;k++){	

		
	kx=(i+IGLOBAL)<NX/2 ? (double)(i+IGLOBAL) : (double)(i+IGLOBAL)-(double)NX ;
	
	kz=(float)k;

	kx=(PI2/LX)*kx;
	kz=(PI2/LZ)*kz;	
	

	for(int j=0;j<NY;j++){
		
	int h=i*NZ*NY+k*NY+j;

	error_1=abs(-(1.0f/((float)pow(acos(-1.0f),2.0f)+kx*kx+kz*kz))*sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f))-u_host_2[h].x);
	error_2=abs(u_host_2[h].y);

	//if(error_1>1e-4 || error_2>1e-4){
	//printf("Error en Hemholzt muy grande"); exit(1);}

	error+=error_1+error_2;
	
	if(error_1>err){
	err=error_1;}

	}	
	}
	}
	
	printf("\n*********HEMHOLZT CHECK********");
	printf("\nError_medio(%d)=%e",error/(2*NY*NZ*NXSIZE),RANK);
	printf("\nError_max(%d)=%e",err,RANK);
	printf("\n\n");

	FILE* fp1;
	fp1=fopen("hemholzt_check.dat","w");
	
	int k=1;
	int i=1;	

	for(int j=0;j<NY;j++){
	
	int h=i*NZ*NY+k*NY+j;

	fprintf(fp1," %f  %f %f ",Fmesh(j*DELTA_Y-1.0f),u_host_2[h].x,u_host_1[h].y);
	fprintf(fp1,"\n");	
	}


	//Eliminate memory	

	cudaCheck(cudaFree(u),"malloc");
	free(u_host_1);
	free(u_host_2);

	return;

}


void checkDerivatives(void){

	float2* u_host_1;
	float2* u_host_2;
		
	float2* u;
	
	u_host_1=(float2*)malloc(SIZE);
	u_host_2=(float2*)malloc(SIZE);
	
	
	for(int i=0;i<NXSIZE;i++){
	for(int j=0;j<NY;j++){
	for(int k=0;k<NZ;k++){

	int h=i*NY*NZ+k*NY+j;
	
	
	u_host_1[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_1[h].y=0.0f;
	
	u_host_2[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_2[h].y=0.0f;	

	}
	}
	}

	cudaCheck(cudaMalloc(&u,SIZE),"malloc");


	//First derivative
	
	cudaCheck(cudaMemcpy(u,u_host_1,SIZE, cudaMemcpyHostToDevice),"MemInfo1");
		
	deriv_Y_HO_double(u);
	//deriv_Y_HO(u);
	
	cudaCheck(cudaMemcpy(u_host_1,u,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");

	//Second derivative
	
	cudaCheck(cudaMemcpy(u,u_host_2,SIZE, cudaMemcpyHostToDevice),"MemInfo1");
		
	deriv_YY_HO_double(u);

	cudaCheck(cudaMemcpy(u_host_2,u,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	
		
	//Compare with known resutls


	float error_1=0.0f;
	float error_2=0.0f;
	float error_y=0.0f;
	float error_yy=0.0f;

	float kx;	
	float kz;

	//Derivada primera
	 float err_y=0.0f;
	 float err_yy=0.0f;

	for(int i=0;i<NXSIZE;i++){
	for(int k=0;k<NZ;k++){	

		
	kx=i<NX/2 ? (float)i : (float)i-(float)NX;
	
	kz=(float)k;

	

	for(int j=0;j<NY;j++){
		
	int h=i*NZ*NY+k*NY+j;

	error_1=abs(acos(-1.0f)*cos(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f))-u_host_1[h].x);
	error_2=abs(u_host_1[h].y);

	if(error_1>err_y){
	err_y=error_1;}

	error_y+=error_1+error_2;


	}	
	}
	}


	
	//Derivada segunda	

	for(int i=0;i<NXSIZE;i++){
	for(int k=0;k<NZ;k++){	

		
	kx=i<NX/2 ? (float)i : (float)i-(float)NX;
	
	kz=(float)k;

	for(int j=0;j<NY;j++){
		
	int h=i*NZ*NY+k*NY+j;

	error_1=abs(-acos(-1.0f)*acos(-1.0f)*sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f))-u_host_2[h].x);
	error_2=abs(u_host_2[h].y);

	if(error_1>err_yy){
	err_yy=error_1;}
	

	error_yy+=error_1+error_2;

	}	
	}
	}
	
	//Derivada segunda	
	
	printf("\n*********DERIVATIVES: first derivative CHECK********");
	printf("\nError_medio(%d)=%e",error_y/(NY*NZ*NXSIZE),RANK);
	printf("\nError_max(%d)=%e",err_y,RANK);
	printf("\n\n");	

	
	//Derivada segunda	
	
	printf("\n*********DERIVATIVES: second derivative CHECK*******");
	printf("\nError_medio(%d)=%e",error_yy/(NY*NZ*NXSIZE),RANK);
	printf("\nError_max(%d)=%e",err_yy,RANK);
	printf("\n\n");

	

	FILE* fp1;
	fp1=fopen("derivatives_check.dat","w");
	
	int k=NZ-1;
	int i=NXSIZE-1;	

	for(int i=0;i<NXSIZE;i++){
	for(int k=0;k<NZ;k++){
	for(int j=0;j<NY;j++){
	
	int h=i*NZ*NY+k*NY+j;

	fprintf(fp1," %f  %f %f ",Fmesh(j*DELTA_Y-1.0f),u_host_1[h].y,u_host_2[h].y);
	fprintf(fp1,"\n");	
	}
	}
	}

	//Eliminate memory	

	cudaCheck(cudaFree(u),"malloc");
	free(u_host_1);
	free(u_host_2);

	return;


}	
	
void checkImplicit(void){

	//Check implicit
	
	float2* u_host_1;
	float2* u_host_2;
		
	float2* u;
	float2* v;
	
	u_host_1=(float2*)malloc(SIZE);
	u_host_2=(float2*)malloc(SIZE);
	
	
	for(int i=0;i<NXSIZE;i++){
	for(int j=0;j<NY;j++){
	for(int k=0;k<NZ;k++){

	int h=i*NY*NZ+k*NY+j;
		
	u_host_1[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_1[h].y=0.0f;
	
	u_host_2[h].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_host_2[h].y=0.0f;	
	
	

	}
	}
	}

	
	cudaCheck(cudaMalloc(&u,SIZE),"malloc");
	cudaCheck(cudaMalloc(&v,SIZE),"malloc");	

	cudaCheck(cudaMemcpy(u,u_host_1,SIZE, cudaMemcpyHostToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(v,u_host_2,SIZE, cudaMemcpyHostToDevice),"MemInfo1");

	/*
	float dt=1e-4;
	int N_steps=100;	
	float betha=1.0f;
	*/

	float dt=REYNOLDS;
	float betha=10.0f;

	/*
	for(int i=0;i<N_steps;i++){

	//Implicit solver
	implicitSolver(u,betha,dt);
	implicitSolver(v,betha,dt);
	}
	*/

	//implicitSolver(u,betha,dt);
	implicitSolver_double(u,betha,dt);
	
	cudaCheck(cudaMemcpy(u_host_1,u,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	cudaCheck(cudaMemcpy(u_host_2,v,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	
	
	//Save data to file		
	
	FILE* fp1,*fp2;

	fp1=fopen("implicit_solution_1.dat","w");
	fp2=fopen("implicit_solution_2.dat","w");

	int k=0;
	int i=0;	

	for(int j=0;j<NY;j++){
	
	int h=i*NZ*NY+k*NY+j;

	fprintf(fp1,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_1[h].x,u_host_1[h].y);
	fprintf(fp1,"\n");
	
	fprintf(fp2,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_2[h].x,u_host_2[h].y);	
	fprintf(fp2,"\n");
	}
	

	//Implicit solver check

	float error_y=0.0f;

	float error_1=0.0f;
	float error_2=0.0f;

	float kx;	
	float kz;

	//float time=dt*N_steps;
	
	float error_max=0.0f;
		
	for(int i=0;i<NXSIZE;i++){
	for(int k=0;k<NZ;k++){	

		
	kx=(i+IGLOBAL)<NX/2 ? (double)(i+IGLOBAL) : (double)(i+IGLOBAL)-(double)NX ;
	kz=(float)k;
	
		
	kx=(PI2/LX)*kx;
	kz=(PI2/LZ)*kz;	

	

	for(int j=0;j<NY;j++){
		
	int h=i*NZ*NY+k*NY+j;

	//float omega_1=(float)pow(acos(-1.0f),2.0f)+kx*kx+kz*kz;

	error_1=abs((1.0f/(1.0f+(dt*betha/REYNOLDS)*((float)pow(acos(-1.0f),2.0f)+kx*kx+kz*kz)))*sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f))-u_host_1[h].x);
	error_2=abs(u_host_1[h].y);

	/*
	if(error_1>1e-3 || error_2>1e-3){
	printf("Error en derivada primera muy grande"); exit(1);}
	*/

	if(error_1>error_max){
	error_max=error_1;}

	error_y+=error_1+error_2;

	

	}	
	}
	}

	//Derivada segunda	
	
	printf("\n*********IMPLICIT: CHECK********");
	printf("\nError_medio(%d)=%e",RANK,error_y/(NY*NZ*NXSIZE));
	printf("\nError_max(%d)=%e",RANK,error_max);
	printf("\n\n");

	//Check data	
	
	cudaCheck(cudaFree(v),"malloc");	
	cudaCheck(cudaFree(u),"malloc");
	free(u_host_1);
	free(u_host_2);


	return;

}


void checkMeanUEvol(void){

	float2* u_mean=(float2*)malloc(NY*sizeof(float2*));

	for(int j=0;j<NY;j++){
	u_mean[j].x=sin(acos(-1.0f)*Fmesh(j*DELTA_Y-1.0f));
	u_mean[j].x=0.0f;	
	u_mean[j].y=0.0f;
	}

	writeUmeanT(u_mean);

	

	float dt=1e-3;
	int Nsteps=50000;
	
	for(int n=0;n<Nsteps;n++){

	for(int i=0;i<3;i++){
	
		meanURKstep_1(i);
		meanURKstep_2(dt,i);
	
	}

	}

	readUmean(u_mean);	

	

	//secondDerivative(u_mean);
	//implicitStepMeanU(u_mean,REYNOLDS,1.0f);

	
	//Save data to file		
	
	FILE* fp1,*fp2;

	fp1=fopen("mean_1.dat","w");
	fp2=fopen("mean_2.dat","w");

	for(int j=0;j<NY;j++){
	

	fprintf(fp1," %f %f",Fmesh(j*DELTA_Y-1.0f),u_mean[j].x);
	fprintf(fp1,"\n");
	
	}




return;

}

void checkBilaplacian(void){

	//Check implicit
	
	float2* u_host_1;
	float2* u_host_2;
	float2* u_host_3;
	float2* u_host_4;			

	float2* v;
	float2* ddv_w;
	float2* ddv;
	float2* dv;
	float2* dv_2;	

	u_host_1=(float2*)malloc(SIZE);
	u_host_2=(float2*)malloc(SIZE);
	u_host_3=(float2*)malloc(SIZE);
	u_host_4=(float2*)malloc(SIZE);
	
	for(int i=0;i<NX;i++){
	for(int j=0;j<NY;j++){
	for(int k=0;k<NZ;k++){

	int h=i*NY*NZ+k*NY+j;
		
	u_host_1[h].x=cos(acos(-1.0f)*DELTA_Y*j/2.0f);
	u_host_1[h].y=0.0f;
	
	

	}
	}
	}

	
	cudaCheck(cudaMalloc(&ddv_w,SIZE),"malloc");
	cudaCheck(cudaMalloc(&v,SIZE),"malloc");	
	cudaCheck(cudaMalloc(&ddv,SIZE),"malloc");	
	cudaCheck(cudaMalloc(&dv,SIZE),"malloc");
	cudaCheck(cudaMalloc(&dv_2,SIZE),"malloc");

	cudaCheck(cudaMemcpy(ddv_w,u_host_1,SIZE, cudaMemcpyHostToDevice),"MemInfo1");

	float dt=1.0f;

	//Implicit solver
	bilaplaSolver(ddv_w,v,dv,0.5f,1e-3);
	hemholztSolver(ddv_w);
	cudaCheck(cudaMemcpy(dv_2,ddv_w,SIZE, cudaMemcpyDeviceToDevice),"MemInfo1");
	deriv_Y_HO(dv_2);	

	cudaCheck(cudaMemcpy(u_host_1,v,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	cudaCheck(cudaMemcpy(u_host_2,ddv_w,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	cudaCheck(cudaMemcpy(u_host_3,dv,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");
	cudaCheck(cudaMemcpy(u_host_4,dv_2,SIZE,cudaMemcpyDeviceToHost),"MemInfo_34");		

	
	//Save data to file		
	
	FILE* fp1,*fp2,*fp3,*fp4;

	fp1=fopen("bilaplacian_1.dat","w");
	fp2=fopen("bilaplacian_2.dat","w");
	fp3=fopen("bilaplacian_3.dat","w");
	fp4=fopen("bilaplacian_4.dat","w");
	
	int k=1;
	int i=1;	

	for(int j=0;j<NY;j++){
	
	int h=i*NZ*NY+k*NY+j;

	fprintf(fp1,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_1[h].x,u_host_1[h].y);
	fprintf(fp1,"\n");
	
	fprintf(fp2,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_2[h].x,u_host_2[h].y);	
	fprintf(fp2,"\n");

		
	fprintf(fp3,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_3[h].x,u_host_3[h].y);	
	fprintf(fp3,"\n");

	fprintf(fp4,"%f %f %f",Fmesh(j*DELTA_Y-1.0f),u_host_4[h].x,u_host_4[h].y);	
	fprintf(fp4,"\n");
	}
	
	/*
	//Implicit solver check

	float error_1=0.0f;
	float error_2=0.0f;

	float kx;	
	float kz;

	for(int i=0;i<NX;i++){
	for(int k=0;k<NZ;k++){	

		
	kx=i<NX/2 ? (float)i : (float)i-(float)NX;
	kz=(float)k;
	
	

	for(int j=0;j<NY;j++){
		
	int h=i*NZ*NY+k*NY+j;

	float omega_1=(float)pow(2.0f*acos(-1.0f),2.0f)+kx*kx+kz*kz;

	error_1=abs(exp(-omega_1*time)*sin(2.0f*acos(-1.0f)*(float)j/(float)(NY-1))-u_host_1[h].x);
	error_2=abs(u_host_1[h].y);

	/*
	if(error_1>1e-3 || error_2>1e-3){
	printf("Error en derivada primera muy grande"); exit(1);}

	error_y+=error_1+error_2;
	*/
	/*
	}	
	}
	}
	*/
	//Check data	
	
	cudaCheck(cudaFree(ddv_w),"malloc");	
	cudaCheck(cudaFree(ddv),"malloc");	
	cudaCheck(cudaFree(v),"malloc");
	free(u_host_1);
	free(u_host_2);


	return;

}



