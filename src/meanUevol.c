
#include"channel.h"
#include <string.h>

static float2* u_host;
static float2* N_host;
static float2* uw_host;
static float2* aux;
static float2* aux_2;

//Implicit step

static float* a;
static float* c;
static float* b;
static float* d;

static float2* ldiag_host_yy;
static float2* cdiag_host_yy;
static float2* udiag_host_yy;


static float2* cdiag_h;
static float2* ldiag_h;
static float2* udiag_h;

//Forcing coeficient

static FILE* fp;
static FILE* fp1;
static FILE* fp2;
static FILE* fp3;
static FILE* fp4;

const float alpha[]={ 29.0f/96.0f, -3.0f/40.0f, 1.0f/6.0f};
const float betha[]={ 37.0f/160.0f, 5.0f/24.0f, 1.0f/6.0f};
const float dseda[]={ 0.0f, -17.0f/60.0f, -5.0f/12.0f};
const float gammad[]={ 8.0f/15.0f, 5.0f/12.0f, 3.0f/4.0f};

float2 Utau_1;
float2 Utau_2;
	
void setRKmean(){
	
  //Alloc all buffers

  u_host=(float2*)malloc(NY*sizeof(float2));
  N_host=(float2*)malloc(NY*sizeof(float2));
  uw_host=(float2*)malloc(NY*sizeof(float2));
  aux=(float2*)malloc(NY*sizeof(float2));
  aux_2=(float2*)malloc(NY*sizeof(float2));
	
  //Allocate various buffers

  c=(float*)malloc(NY*sizeof(float));
  b=(float*)malloc(NY*sizeof(float));
  a=(float*)malloc(NY*sizeof(float));
  d=(float*)malloc(NY*sizeof(float));	

  ldiag_host_yy=(float2*)malloc(NY*sizeof(float2));
  cdiag_host_yy=(float2*)malloc(NY*sizeof(float2));
  udiag_host_yy=(float2*)malloc(NY*sizeof(float2));

  ldiag_h=(float2*)malloc(NY*sizeof(float2));
  cdiag_h=(float2*)malloc(NY*sizeof(float2));
  udiag_h=(float2*)malloc(NY*sizeof(float2));

  //INITIATE

  for(int j=0;j<NY;j++){
    u_host[j].x=0.0f;
    u_host[j].y=0.0f;
	
    N_host[j].x=0.0f;
    N_host[j].y=0.0f;
	
    uw_host[j].x=0.0f;
    uw_host[j].y=0.0f;
	
    aux[j].x=0.0f;
    aux[j].y=0.0f;	
  }	
	
}

void solve_tridiagonal_in_place_destructive(float2* x, const size_t N, const float2* a, const float2* b, float2* c) {
  /* unsigned integer of same size as pointer */
  size_t in;
   
  /*
    solves Ax = v where A is a tridiagonal matrix consisting of vectors a, b, c
    note that contents of input vector c will be modified, making this a one-time-use function
    x[] - initially contains the input vector v, and returns the solution x. indexed from [0, ..., N - 1]
    N — number of equations
    a[] - subdiagonal (means it is the diagonal below the main diagonal) -- indexed from [1, ..., N - 1]
    b[] - the main diagonal, indexed from [0, ..., N - 1]
    c[] - superdiagonal (means it is the diagonal above the main diagonal) -- indexed from [0, ..., N - 2]
  */
   
  c[0].x = c[0].x / b[0].x;
  x[0].x = x[0].x / b[0].x;
   
  /* loop from 1 to N - 1 inclusive */
  for (in = 1; in < N; in++) {
    float m = 1.0 / (b[in].x - a[in].x * c[in - 1].x);
    c[in].x = c[in].x * m;
    x[in].x = (x[in].x - a[in].x * x[in - 1].x) * m;
  }
   
  /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
  for (in = N - 1; in-- > 0; )
    x[in].x = x[in].x - c[in].x * x[in + 1].x;

}


void writeUmean(float2* u,domain_t domain){

  //char mess[30];
  //sprintf(mess,"MemInfo_Read_mean:%d",RANK) ;
  cudaCheck(cudaMemcpy(u,u_host,NY*sizeof(float2), cudaMemcpyHostToDevice),domain,"MemInfo_Read_mean");
  return;
	

}

void readNmean(float2* u, domain_t domain){

  cudaCheck(cudaMemcpy(N_host,u,NY*sizeof(float2), cudaMemcpyDeviceToHost),domain,"MemInfo_Read_Nmean");
  return;


}

void readUmean(float2* u){

  //cudaCheck(cudaMemcpy(aux,u,NY*sizeof(float2), cudaMemcpyDeviceToHost),"MemInfo1");

  for(int j=0;j<NY;j++){
    u[j].x=u_host[j].x;
  }
	
  return;

}

void writeUmeanT(float2* u_r){
  for(int j=0;j<NY;j++){
    u_host[j].x=u_r[j].x;
  }
}

void writeU(char* fname){

  FILE* fp_w;
  fp_w=fopen(fname,"wb");
  if(fp_w==NULL){printf("\nerror escritura: %s",fname); exit(1);}
  size_t fsize =fwrite( (unsigned char*)u_host,sizeof(float2),NY,fp_w);
  if(fsize!=NY){ printf("\nwriting error: %s",fname); exit(1);}
  fclose(fp_w);

  return;

}

void readU(char* fname){

  FILE* fp_r;
  fp_r=fopen(fname,"rb");
  if(fp_r==NULL){printf("\nerror lectura: %s",fname); exit(1);}
  size_t fsize =fread( (unsigned char*)u_host,sizeof(float2),NY,fp_r);
  if(fsize!=NY){ printf("\nreading error: %s,%d",fname,fsize); exit(1);}
  fclose(fp_r);
  return;

}

void readUtau(float2* wz, domain_t domain){


  cudaCheck(cudaMemcpy(&Utau_1,wz,sizeof(float2),cudaMemcpyDeviceToHost),domain,"MemInfo1");
  cudaCheck(cudaMemcpy(&Utau_2,wz+NY-1,sizeof(float2),cudaMemcpyDeviceToHost),domain,"MemInfo1");

	

  float N2=NX*(2*NZ-2);
	
  Utau_1.x/=N2;
  Utau_2.x/=N2;

  Utau_1.x=sqrt(1.0f/REYNOLDS*fabs(Utau_1.x));
  Utau_2.x=sqrt(1.0f/REYNOLDS*fabs(Utau_2.x));
		
  //printf("\n %f\n",fabs(-1.0f));		
  //printf("\n(tau_1,tau_2)=[%d](%e,%e)",Utau_1.x,Utau_2.x,RANK);
	
  return;

}

static void forcing(float2* u, domain_t domain){

  float N2=NX*(2*NZ-2);

  //Calc caudal
  float Qt=0.0f;

  for(int j=10;j<NY-1;j++){
    float dy=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
    Qt+=0.5f*(u[j+1].x+u[j].x)*dy;
  }	

  //Forcing done in fourier space so Q has to be multiplied by N2	

  for(int i=1;i<NY-1;i++){
    u[i].x=u[i].x+(QVELOCITY*N2-Qt)/LY;
    u[i].y=0.0f;}
	


}

void secondDerivative(float2* u){


  //SET DERIVATIVES 

  for(int j=0;j<NY;j++){
		
    float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
    float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);			

    float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
    float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);

    udiag_host_yy[j].x=alpha;
    udiag_host_yy[j].y=0.0f;
		
    cdiag_host_yy[j].x=1.0f;
    cdiag_host_yy[j].y=0.0f;
	
    ldiag_host_yy[j].x=betha;
    ldiag_host_yy[j].y=0.0f;
		
  }

					
  int h_0=0;	

  ldiag_host_yy[h_0+0].x=0.0f;
  udiag_host_yy[h_0+NY-1].x=0.0f;	
		

  int j=0;

  float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
  float b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
		

  float alpha=(a+b)/(2.0f*a-b);

  udiag_host_yy[h_0].x=alpha;
  cdiag_host_yy[h_0].x=1.0f;

  j=NY-1;

  a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
  b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
			

  alpha=(a+b)/(2.0f*a-b);			
			
  ldiag_host_yy[h_0+NY-1].x=alpha;
  cdiag_host_yy[h_0+NY-1].x=1.0f;


  ///////////////////////////////

  for(int j=1;j<NY-1;j++){

    float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
    float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
    float c;

    float A=-12.0f*b/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
    float B= 12.0f*a/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
    float C=-A-B;
    float E;		

    aux[j].x=A*u[j+1].x+C*u[j].x+B*u[j-1].x;
    aux[j].y=A*u[j+1].y+C*u[j].y+B*u[j-1].y;
		
  }

  float A;
  float B;
  float E;

  //if(j==0){

  j=0;

  a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
  b=Fmesh((j+2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
		
  A= 6.0f/((a-b)*(2.0f*a-b));
  B= -6.0f*a/((a*b-b*b)*(2.0f*a-b));
	
  E=-A-B;

  aux[j].x=E*u[0].x+A*u[1].x+B*u[2].x;
  aux[j].y=E*u[0].y+A*u[1].y+B*u[2].y;				

				
	
  //if(j==NY-1){

  j=NY-1;

  a=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
  b=Fmesh((j-2)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);

  A= 6.0f/((a-b)*(2.0f*a-b));
  B= -6.0f*a/((a*b-b*b)*(2.0f*a-b));
	
  E=-A-B;

  aux[j].x=E*u[NY-1].x+A*u[NY-2].x+B*u[NY-3].x;
  aux[j].y=E*u[NY-1].y+A*u[NY-2].y+B*u[NY-3].y;			

			

		

  for(int j=0;j<NY;j++){
    u[j].x=aux[j].x;}

  solve_tridiagonal_in_place_destructive(u,NY,(const float2*)ldiag_host_yy,(const float2*)cdiag_host_yy,udiag_host_yy);
	


  return;


}

void implicitStepMeanU(float2* u,float betha_RK,float dt, domain_t domain){

  //Seconnd derivative

  for(int j=0;j<NY;j++){

    float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
    float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	
	

    float A=-12.0f*b/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
    float B= 12.0f*a/(a*a*a-b*b*b-4.0f*a*a*b+4.0f*b*b*a);
    float C=-A-B;
		
    float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
    float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);

		
    float nu=1.0f/REYNOLDS;

    //veamos

	
    ldiag_h[j].x=betha-nu*dt*betha_RK*B;
    ldiag_h[j].y=0.0f;			
	
    cdiag_h[j].x=1.0f-nu*dt*betha_RK*C;
    cdiag_h[j].y=0.0f;		
	
    udiag_h[j].x=alpha-nu*dt*betha_RK*A;
    udiag_h[j].y=0.0f;	

    //To be improved 
		
    if(j==0){
      ldiag_h[j].x=0.0f;
      cdiag_h[j].x=1.0f;
      udiag_h[j].x=0.0f;
    }
	
    if(j==1){
      ldiag_h[j].x=0.0f;
    }

    if(j==NY-1){
      ldiag_h[j].x=0.0f;
      cdiag_h[j].x=1.0f;
      udiag_h[j].x=0.0f;
    }	
	
    if(j==NY-2){
      udiag_h[j].x=0.0f;
    }

  }

		

  for(int j=0;j<NY;j++){


    float a=Fmesh((j+1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);
    float b=Fmesh((j-1)*DELTA_Y-1.0f)-Fmesh(j*DELTA_Y-1.0f);	

    float alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);
    float betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0f*a*a*b+4.0f*a*b*b-b*b*b);


    aux[j].x=(alpha*u[j+1].x+u[j].x+betha*u[j-1].x);

    if(j==0){
      aux[j].x=0.0f;
      aux[j].y=0.0f;		
    }		
	
    if(j==NY-1){
      aux[j].x=0.0f;
      aux[j].y=0.0f;		
    }	

  }

  for(int j=0;j<NY;j++){
    u[j].x=aux[j].x;}

  solve_tridiagonal_in_place_destructive(u,NY,(const float2*)ldiag_h,(const float2*)cdiag_h,udiag_h);


  return;

}

void meanURKstep_1(int in, domain_t domain){



  //Calc first step

  //Second derivative with boundary conditions

  for( int j=0;j<NY;j++){
    aux_2[j].x=u_host[j].x;
    aux_2[j].y=u_host[j].y;	
  }

  secondDerivative(aux_2);

  //First step
	
  for( int j=0;j<NY;j++){
	
    uw_host[j].x=(alpha[in]+betha[in])*(1.0f/REYNOLDS)*aux_2[j].x+dseda[in]*N_host[j].x;
	
  }

  return;

}

void meanURKstep_2(float dt, int in, domain_t domain, paths_t path){


  //End

  for(int j=0;j<NY;j++){
    uw_host[j].x=dt*(uw_host[j].x+gammad[in]*N_host[j].x);	
  }
	
  //Calc implicit step: solve tridiagonal 

  implicitStepMeanU(uw_host,betha[in],dt,domain);
	
  for(int j=0;j<NY;j++){
    u_host[j].x=u_host[j].x+uw_host[j].x;	
  }
	
  forcing(u_host, domain);

  //END OF THE RK STEP

  //STATISTICS
	
  if(in==2){

    float Umean=QVELOCITY/LY;
    float nu=1.0f/REYNOLDS;
    float N2=NX*(2*NZ-2);
    float u_tau;
    char thispath[100];

    //Statistics

    printf("\n(tau_1,tau_2)=(%e,%e)",Utau_1.x,Utau_2.x);	

    u_tau=sqrt(0.5f*(pow(Utau_1.x,2.0f)+pow(Utau_2.x,2.0f)));

    printf("\n****MEAN_PROFILE_STATISTICS****");
    printf("\n(RE_t,RE_c,RE_m)=(%e,%e,%e)",u_tau*LY*0.5f/nu,u_host[NY/2].x*0.5f*LY/(nu*N2),1.0f/nu);
    printf("\n(Dx+,Dz+)=(%e,%e)",(3.0f/2.0f)*u_tau*LX/(nu*NX),(3.0f/2.0f)*u_tau*LZ/(nu*(2*NZ-2)));
    printf("\nDy+(max,min)=(%f,%f)",u_tau*(Fmesh((NY/2+1)*DELTA_Y-1.0f)-Fmesh((NY/2)*DELTA_Y-1.0f))/(nu),u_tau*(Fmesh(1*DELTA_Y-1.0f)-Fmesh(0*DELTA_Y-1.0f))/(nu));	
    printf("\nC_f=%e",2.0f*u_tau/(Umean*Umean));
    printf("\n(Um+,Ux+,Um/Uc)=(%f,%f,%f)",Umean/u_tau,u_host[NY/2].x/(u_tau*N2),Umean*N2/u_host[NY/2].x);
    printf("\n");
    
    strcpy(thispath, path.path);
    strcat(thispath,"MEANPROFILE.dat");
    fp =fopen(thispath,"a");
    
    strcpy(thispath, path.path);
    strcat(thispath,"MEANREAYNOLDS.dat");
    fp1=fopen(thispath,"a");

    strcpy(thispath, path.path);
    strcat(thispath,"UTAU.dat");
    fp2=fopen(thispath,"a");

    strcpy(thispath, path.path);
    strcat(thispath,"STATISTICS.dat");
    fp3=fopen(thispath,"a");

    strcpy(thispath, path.path);
    strcat(thispath,"RESOLUTION.dat");
    fp4=fopen(thispath,"a");

    for(int j=0;j<NY;j++){
      fprintf(fp," %f",u_host[j].x);
    }
    fprintf(fp," \n");
    
	
    for(int j=0;j<NY;j++){
      fprintf(fp1," %f",N_host[j].x);
    }
    fprintf(fp1," \n");
	
    fprintf(fp2," %f",u_tau);

    fprintf(fp3,"%e %e %e %e %e %e %e %e ",u_tau*LY*0.5f/nu,u_host[NY/2].x*0.5f*LY/(nu*N2),1.0f/nu,Umean/u_tau,
    	    u_host[NY/2].x/(u_tau*N2),Umean*N2/u_host[NY/2].x,
    	    2.0f*u_tau*u_tau/(Umean*Umean),2.0f*u_tau*u_tau/(u_host[NY/2].x*u_host[NY/2].x));
    fprintf(fp3,"\n");

    fprintf(fp4,"%f %f %f %f",(3.0f/2.0f)*u_tau*LX/(nu*NX),(3.0f/2.0f)*u_tau*LZ/(nu*NZ),
    	    u_tau*(Fmesh((NY/2+1)*DELTA_Y-1.0f)-Fmesh((NY/2)*DELTA_Y-1.0f))/(nu),u_tau*(Fmesh(1*DELTA_Y-1.0f)-Fmesh(0*DELTA_Y-1.0f))/(nu));
    fprintf(fp4,"\n");
	
    fclose(fp);
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
	

  }
	
	
		

  return;

}



