
#include "channel.h"
/*
///////////////////KERNELS////////////////////////

static __global__ void calcReynolds(float2* dv,float2* ux,float2* uy)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	float N2=NX*(2*NZ-2);

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;

	if (i<NX && j<NY && k<NZ)
	{
	

	float2 u1=ux[h];
	float2 u2=uy[h];

	u2.x=u2.x/N2;
	u2.y=u2.y/N2;
	
	u1.x=u1.x/N2;	
	u1.y=u1.y/N2;

	
	u2.x=-u2.x*u1.x;
	u2.y=-u2.y*u1.y;
	
	if(k!=0){
	u2.x*=2.0f;
	u2.y*=2.0f;
	}
	
		
		
	//Write
	
	dv[h]=u2;
	
	}
	
	
}

static __global__ void calcRMS(float2* dv,float2* ux)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	float N2=NX*(2*NZ-2);

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;

	if (i<NX && j<NY && k<NZ)
	{
	

	float2 u1=ux[h];
	
	u1.x=u1.x/N2;	
	u1.y=u1.y/N2;

	u1.x=u1.x*u1.x;
	u1.y=u1.y*u1.y;
	
	if(k!=0){
	u1.x*=2.0f;
	u1.y*=2.0f;
	}
	
	//Write
	
	dv[h]=u1;
	
	}
	
	
}

static __global__ void calcEnstrophy(float2* wx,float2* wy,float2* wz)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	int j=k%NY;
	k=(k-j)/NY;

	float N2=NX*(2*NZ-2);

	// [i,k,j][NX,NZ,NY]	

	int h=i*NY*NZ+k*NY+j;

	if (i<NX && j<NY && k<NZ)
	{
	
	float2 ens;

	float2 w1=wx[h];
	float2 w2=wy[h];
	float2 w3=wz[h];	

	w1.x=w1.x/N2;	
	w1.y=w1.y/N2;

	w2.x=w2.x/N2;	
	w2.y=w2.y/N2;
	
	w3.x=w3.x/N2;	
	w3.y=w3.y/N2;

	//Veamos
	
	ens.x=w1.x*w1.x+w2.x*w2.x+w3.x*w3.x;
	ens.y=w1.y*w1.y+w2.y*w2.y+w3.y*w3.y;
	
	//Write
	
	wx[h]=ens;
	
	}
	
	
}

/////////////////////Funciones/////////////////////////////////////

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;
static int threadsPerBlock_in=16;
static cudaError_t ret;



void calcSt(float2* dv,float2* u,float2* v,float2* w){
	
	threadsPerBlock.x=threadsPerBlock_in;
	threadsPerBlock.y=threadsPerBlock_in;

	blocksPerGrid.x=NX/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	int N2=NX*(2*NZ-2);

	static FILE* fp1=fopen("STRESSES.dat","w");
	static FILE* fp2=fopen("URMS.dat","w");
	static FILE* fp3=fopen("VRMS.dat","w");
	static FILE* fp4=fopen("WRMS.dat","w");
	
	
	static float2* aux=(float2*)malloc(NY*sizeof(float2));

	//REYNOLD STRESSES	

	calcReynolds<<<blocksPerGrid,threadsPerBlock>>>(dv,u,v);
	cudaCheck(ret,"Wkernel",1);
	
	backwardTranspose(dv);
	fftForwardTranspose(dv);
	
	cudaCheck(cudaMemcpy(aux,dv,NY*sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");
	for(int j=0;j<NY;j++){
	fprintf(fp1," %f",aux[j].x);}
	fprintf(fp1," \n");

	//URMS

	calcRMS<<<blocksPerGrid,threadsPerBlock>>>(dv,u);
	cudaCheck(ret,"Wkernel",1);
	
	backwardTranspose(dv);
	fftForwardTranspose(dv);

	cudaCheck(cudaMemcpy(aux,dv,NY*sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");
	for(int j=0;j<NY;j++){
	fprintf(fp2," %f",sqrt(aux[j].x));}
	fprintf(fp2," \n");
	
	//VRMS

	calcRMS<<<blocksPerGrid,threadsPerBlock>>>(dv,v);
	cudaCheck(ret,"Wkernel",1);

	backwardTranspose(dv);
	fftForwardTranspose(dv);	

	cudaCheck(cudaMemcpy(aux,dv,NY*sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");
	for(int j=0;j<NY;j++){
	fprintf(fp3," %f",sqrt(aux[j].x));}
	fprintf(fp3," \n");

	//WRMS

	calcRMS<<<blocksPerGrid,threadsPerBlock>>>(dv,w);
	cudaCheck(ret,"Wkernel",1);
	
	backwardTranspose(dv);
	fftForwardTranspose(dv);	

	cudaCheck(cudaMemcpy(aux,dv,NY*sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");
	for(int j=0;j<NY;j++){
	fprintf(fp4," %f",sqrt(aux[j].x));}
	fprintf(fp4," \n");
	
		
	return;

}

void calcSpectra(float2* dv,float2* u,float2* v,float2* w){

	static int counter=0;	

	threadsPerBlock.x=threadsPerBlock_in;
	threadsPerBlock.y=threadsPerBlock_in;

	blocksPerGrid.x=NX/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;

	int N2=NX*(2*NZ-2);


	
	static float2* p1u=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p2u=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p3u=(float2*)malloc(NX*NZ*sizeof(float2));
	
	static float2* p1v=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p2v=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p3v=(float2*)malloc(NX*NZ*sizeof(float2));

	static float2* p1w=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p2w=(float2*)malloc(NX*NZ*sizeof(float2));
	static float2* p3w=(float2*)malloc(NX*NZ*sizeof(float2));	

	static float2* aux=(float2*)malloc(SIZE);


	int j=NY/2;

	//URMS
	
	cudaCheck(cudaMemcpy(aux,u,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
	
	FILE* fp1=fopen("Uspec.dat","w");
	
	for(int i=0;i<NX;i++){
	for(int k=0;k<NZ;k++){
	int h=i*NZ*NY+k*NY+j;
	fprintf(fp1," %f",aux[h].x*aux[h].x+aux[h].y*aux[h].y);}
	fprintf(fp1," \n");
	}
	
	fclose(fp1);
	
	//VRMS

	cudaCheck(cudaMemcpy(aux,v,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
	
	static FILE* fp2=fopen("Vspec.dat","w");
	
	for(int i=0;i<NX;i++){
	for(int k=0;k<NZ;k++){
	int h=i*NZ*NY+k*NY+j;
	fprintf(fp2," %f",aux[h].x*aux[h].x+aux[h].y*aux[h].y);}
	fprintf(fp2," \n");
	}

	fclose(fp2);

	//WRMS
	
	cudaCheck(cudaMemcpy(aux,w,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
	
	static FILE* fp3=fopen("Wspec.dat","w");
	
	for(int i=0;i<NX;i++){
	for(int k=0;k<NZ;k++){
	int h=i*NZ*NY+k*NY+j;
	fprintf(fp3," %f",aux[h].x*aux[h].x+aux[h].y*aux[h].y);}
	fprintf(fp3," \n");
	}

	fclose(fp3);

	counter++;		

	return;

}


void calcEnstrophy(float2* ddv,float2* g,float2* v,float2*dv,float2* wx,float2* wz,float2* u,float2* w){

 	threadsPerBlock.x=threadsPerBlock_in;
	threadsPerBlock.y=threadsPerBlock_in;

	blocksPerGrid.x=NX/threadsPerBlock.x;
	blocksPerGrid.y=NZ*NY/threadsPerBlock.y;	
	 
	calcUW(u,w,dv,g);



	calcOmega(wx,u,wz,u,v,w);
	cudaCheck(cudaMemcpy(u,g,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");

	fftBackwardTranspose(wx);	
	fftBackwardTranspose(u);	
	fftBackwardTranspose(wz);

	calcEnstrophy<<<blocksPerGrid,threadsPerBlock>>>(wx,u,wz);
	

return;

}
*/

