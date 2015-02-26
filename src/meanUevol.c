
#include"channel.h"
#include <string.h>

static double2* u_host;
static double2* w_host;
static double2* N_host;
static double2* NW_host;
static double2* uw_host;
static double2* Ww_host;
static double2* aux;
static double2* aux_2;
static double2* aux_3;

static float2* aux_float;

//Implicit step

static double* a;
static double* c;
static double* b;
static double* d;

static double2* ldiag_host_yy;
static double2* cdiag_host_yy;
static double2* udiag_host_yy;


static double2* cdiag_h;
static double2* ldiag_h;
static double2* udiag_h;

//Forcing coeficient

static FILE* fp;
static FILE* fp1;
static FILE* fp2;
static FILE* fp3;
static FILE* fp4;

const double alpha[]={ 29.0/96.0, -3.0/40.0, 1.0/6.0};
const double betha[]={ 37.0/160.0, 5.0/24.0, 1.0/6.0};
const double dseda[]={ 0.0, -17.0/60.0, -5.0/12.0};
const double gammad[]={ 8.0/15.0, 5.0/12.0, 3.0/4.0};

//Fraction of the Uwall=f*QVELOCITY/LY. Velocity of the wall

const double Uwall=-0.53;

float2 Utau_1;
float2 Utau_2;
	
void setRKmean(){
	
  //Alloc all buffers

	u_host=(double2*)malloc(NY*sizeof(double2));
	w_host=(double2*)malloc(NY*sizeof(double2));

	N_host=(double2*)malloc(NY*sizeof(double2));
	NW_host=(double2*)malloc(NY*sizeof(double2));

	uw_host=(double2*)malloc(NY*sizeof(double2));
	Ww_host=(double2*)malloc(NY*sizeof(double2));

	aux=(double2*)malloc(NY*sizeof(double2));
	aux_2=(double2*)malloc(NY*sizeof(double2));
	aux_3=(double2*)malloc(NY*sizeof(double2));

	aux_float=(float2*)malloc(NY*sizeof(float2));	

  //Allocate various buffers

  c=(double*)malloc(NY*sizeof(double));
  b=(double*)malloc(NY*sizeof(double));
  a=(double*)malloc(NY*sizeof(double));
  d=(double*)malloc(NY*sizeof(double));	

  ldiag_host_yy=(double2*)malloc(NY*sizeof(double2));
  cdiag_host_yy=(double2*)malloc(NY*sizeof(double2));
  udiag_host_yy=(double2*)malloc(NY*sizeof(double2));

  ldiag_h=(double2*)malloc(NY*sizeof(double2));
  cdiag_h=(double2*)malloc(NY*sizeof(double2));
  udiag_h=(double2*)malloc(NY*sizeof(double2));

  //INITIATE

  for(int j=0;j<NY;j++){
    u_host[j].x=0.0;
    u_host[j].y=0.0;

    w_host[j].x=0.0;
    w_host[j].y=0.0;
	
    N_host[j].x=0.0;
    N_host[j].y=0.0;
	
	NW_host[j].x=0.0;
    NW_host[j].y=0.0;
	
    uw_host[j].x=0.0;
    uw_host[j].y=0.0;

    Ww_host[j].x=0.0;
    Ww_host[j].y=0.0;
	
    aux[j].x=0.0;
    aux[j].y=0.0;	

    aux_2[j].x=0.0;
    aux_2[j].y=0.0;	

    aux_3[j].x=0.0;
    aux_3[j].y=0.0;	

	aux_float[j].x=0.0f;
	aux_float[j].y=0.0f;

  }	
	
}

void solve_tridiagonal_in_place_destructive(double2* x, const size_t N, const double2* a, const double2* b, double2* c) {
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
    double m = 1.0 / (b[in].x - a[in].x * c[in - 1].x);
    c[in].x = c[in].x * m;
    x[in].x = (x[in].x - a[in].x * x[in - 1].x) * m;
  }
   
  /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
  for (in = N - 1; in-- > 0; )
    x[in].x = x[in].x - c[in].x * x[in + 1].x;

}


void writeUmean(float2* u,float2* w,domain_t domain){

  //char mess[30];
  //sprintf(mess,"MemInfo_Read_mean:%d",RANK) ;

	for(int j=0;j<NY;j++){
	aux_float[j].x=(float)(u_host[j].x+Uwall*QVELOCITY/LY*(double)(NX*(2*(NZ-1))));
	}
	CHECK_CUDART(cudaMemcpy(u,aux_float,NY*sizeof(float2), cudaMemcpyHostToDevice));
	for(int j=0;j<NY;j++){
	aux_float[j].x=(float)w_host[j].x;
	}	
	CHECK_CUDART(cudaMemcpy(w,aux_float,NY*sizeof(float2), cudaMemcpyHostToDevice) );
 
  return;
	

}


void readNmean(float2* u, float2* w, domain_t domain){

    CHECK_CUDART( cudaMemcpy(aux_float,u,NY*sizeof(float2), cudaMemcpyDeviceToHost) );
	for(int j=0;j<NY;j++){
	N_host[j].x=(double)aux_float[j].x;
	}	
    CHECK_CUDART( cudaMemcpy(aux_float,w,NY*sizeof(float2), cudaMemcpyDeviceToHost) );
	for(int j=0;j<NY;j++){
	NW_host[j].x=(double)aux_float[j].x;
	}	

  return;


}

void readUmean(float2* u){

  //cudaCheck(cudaMemcpy(aux,u,NY*sizeof(float2), cudaMemcpyDeviceToHost),"MemInfo1");

  for(int j=0;j<NY;j++){
    u[j].x=(float)u_host[j].x;
  }
	
  return;

}

void writeUmeanT(float2* u_r){
  for(int j=0;j<NY;j++){
    u_host[j].x=(double)u_r[j].x;
  }
}

void writeU(char* fname){

  FILE* fp_w;
  fp_w=fopen(fname,"wb");
  if(fp_w==NULL){printf("\nerror escritura: %s",fname); exit(1);}
  size_t fsize =fwrite( (unsigned char*)u_host,sizeof(double2),NY,fp_w);
  if(fsize!=NY){ printf("\nwriting error: %s",fname); exit(1);}
  fclose(fp_w);

  return;

}

void writeW(char* fname){

  FILE* fp_w;
  fp_w=fopen(fname,"wb");
  if(fp_w==NULL){printf("\nerror escritura: %s",fname); exit(1);}
  size_t fsize =fwrite( (unsigned char*)w_host,sizeof(double2),NY,fp_w);
  if(fsize!=NY){ printf("\nwriting error: %s",fname); exit(1);}
  fclose(fp_w);

  return;

}

void readU(char* fname){

  FILE* fp_r;
  fp_r=fopen(fname,"rb");
  if(fp_r==NULL){printf("\nerror lectura: %s",fname); exit(1);}
  size_t fsize =fread( (unsigned char*)u_host,sizeof(double2),NY,fp_r);
  if(fsize!=NY){ printf("\nreading error: %s,%d",fname,fsize); exit(1);}
  fclose(fp_r);
  return;

}

void readW(char* fname){

  FILE* fp_r;
  fp_r=fopen(fname,"rb");
  if(fp_r==NULL){printf("\nerror lectura: %s",fname); exit(1);}
  size_t fsize =fread( (unsigned char*)w_host,sizeof(double2),NY,fp_r);
  if(fsize!=NY){ printf("\nreading error: %s,%d",fname,fsize); exit(1);}
  fclose(fp_r);
  return;

}

void readUtau(float2* wz, domain_t domain){


  CHECK_CUDART( cudaMemcpy(&Utau_1,wz     ,sizeof(float2),cudaMemcpyDeviceToHost) );
  CHECK_CUDART( cudaMemcpy(&Utau_2,wz+NY-1,sizeof(float2),cudaMemcpyDeviceToHost) );

  double N2=NX*(2*NZ-2);
	
  Utau_1.x/=N2;
  Utau_2.x/=N2;

  Utau_1.x=sqrt(1.0/REYNOLDS*fabs(Utau_1.x));
  Utau_2.x=sqrt(1.0/REYNOLDS*fabs(Utau_2.x));
		
  //printf("\n %f\n",fabs(-1.0f));		
  //printf("\n(tau_1,tau_2)=[%d](%e,%e)",Utau_1.x,Utau_2.x,RANK);
	
  return;

}

static void forcing(double2* u, domain_t domain){

  double N2=NX*(2*NZ-2);

  //Calc caudal
  double Qt=0.0f;

  for(int j=0;j<NY-1;j++){
    double dy=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    Qt+=0.5*(u[j+1].x+u[j].x)*dy;
  }	

  //Forcing done in fourier space so Q has to be multiplied by N2	

  for(int i=1;i<NY-1;i++){
    u[i].x=u[i].x+(QVELOCITY*N2-Qt)/LY;
    u[i].y=0.0;}
	


}

void secondDerivative(double2* u){
START_RANGE_ASYNC("secondDericative",37)

  //SET DERIVATIVES 

  for(int j=0;j<NY;j++){
		
    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);			

    double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
    double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);

    udiag_host_yy[j].x=alpha;
    udiag_host_yy[j].y=0.0;
		
    cdiag_host_yy[j].x=1.0;
    cdiag_host_yy[j].y=0.0;
	
    ldiag_host_yy[j].x=betha;
    ldiag_host_yy[j].y=0.0;
		
  }

					
  int h_0=0;	

  ldiag_host_yy[h_0+0].x=0.0;
  udiag_host_yy[h_0+NY-1].x=0.0;	
		

  int j=0;

  double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
  double b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
		

  double alpha=(a+b)/(2.0*a-b);

  udiag_host_yy[h_0].x=alpha;
  cdiag_host_yy[h_0].x=1.0;

  j=NY-1;

  a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
  b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
			

  alpha=(a+b)/(2.0*a-b);			
			
  ldiag_host_yy[h_0+NY-1].x=alpha;
  cdiag_host_yy[h_0+NY-1].x=1.0;


  ///////////////////////////////

  for(int j=1;j<NY-1;j++){

    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
    double c;

    double A=-12.0*b/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double B= 12.0*a/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double C=-A-B;
    double E;		

    aux[j].x=A*u[j+1].x+C*u[j].x+B*u[j-1].x;
    aux[j].y=A*u[j+1].y+C*u[j].y+B*u[j-1].y;
		
  }

  double A;
  double B;
  double E;

  //if(j==0){

  j=0;

  a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
  b=Fmesh((j+2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
		
  A=  6.0/((a-b)*(2.0*a-b));
  B= -6.0*a/((a*b-b*b)*(2.0*a-b));
	
  E=-A-B;

  aux[j].x=E*u[0].x+A*u[1].x+B*u[2].x;
  aux[j].y=E*u[0].y+A*u[1].y+B*u[2].y;				

				
	
  //if(j==NY-1){

  j=NY-1;

  a=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
  b=Fmesh((j-2)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);

  A=  6.0/((a-b)*(2.0*a-b));
  B= -6.0*a/((a*b-b*b)*(2.0*a-b));
	
  E=-A-B;

  aux[j].x=E*u[NY-1].x+A*u[NY-2].x+B*u[NY-3].x;
  aux[j].y=E*u[NY-1].y+A*u[NY-2].y+B*u[NY-3].y;			

			

		

  for(int j=0;j<NY;j++){
    u[j].x=aux[j].x;}

  solve_tridiagonal_in_place_destructive(u,NY,(const double2*)ldiag_host_yy,(const double2*)cdiag_host_yy,udiag_host_yy);
	
END_RANGE_ASYNC

  return;


}

void implicitStepMeanU(double2* u,double betha_RK,double dt, domain_t domain){
START_RANGE_ASYNC("implicitStepMeanU",36)
  //Seconnd derivative

  for(int j=0;j<NY;j++){

    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	
	

    double A=-12.0*b/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double B= 12.0*a/(a*a*a-b*b*b-4.0*a*a*b+4.0*b*b*a);
    double C=-A-B;
		
    double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
    double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);

		
    double nu=1.0/REYNOLDS;

    //veamos

	
    ldiag_h[j].x=betha-nu*dt*betha_RK*B;
    ldiag_h[j].y=0.0;			
	
    cdiag_h[j].x=1.0-nu*dt*betha_RK*C;
    cdiag_h[j].y=0.0;		
	
    udiag_h[j].x=alpha-nu*dt*betha_RK*A;
    udiag_h[j].y=0.0;	

    //To be improved 
		
    if(j==0){
      ldiag_h[j].x=0.0;
      cdiag_h[j].x=1.0;
      udiag_h[j].x=0.0;
    }
	
    if(j==1){
      ldiag_h[j].x=0.0;
    }

    if(j==NY-1){
      ldiag_h[j].x=0.0;
      cdiag_h[j].x=1.0;
      udiag_h[j].x=0.0;
    }	
	
    if(j==NY-2){
      udiag_h[j].x=0.0;
    }

  }

		

  for(int j=0;j<NY;j++){


    double a=Fmesh((j+1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);
    double b=Fmesh((j-1)*DELTA_Y-1.0)-Fmesh(j*DELTA_Y-1.0);	

    double alpha=-(-b*b*b-a*b*b+a*a*b)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);
    double betha=-( a*a*a+b*a*a-b*b*a)/(a*a*a-4.0*a*a*b+4.0*a*b*b-b*b*b);


    aux[j].x=(alpha*u[j+1].x+u[j].x+betha*u[j-1].x);

    if(j==0){
      aux[j].x=0.0;
      aux[j].y=0.0;		
    }		
	
    if(j==NY-1){
      aux[j].x=0.0;
      aux[j].y=0.0;		
    }	

  }

  for(int j=0;j<NY;j++){
    u[j].x=aux[j].x;}

  solve_tridiagonal_in_place_destructive(u,NY,(const double2*)ldiag_h,(const double2*)cdiag_h,udiag_h);

END_RANGE_ASYNC
  return;

}

void meanURKstep_1(int in, domain_t domain){

START_RANGE_ASYNC("meanURKstep_1",35)

  //Calc first step

  //Second derivative with boundary conditions

  for( int j=0;j<NY;j++){
    aux_2[j].x=u_host[j].x;
    aux_2[j].y=u_host[j].y;	

	aux_3[j].x=w_host[j].x;
	aux_3[j].y=w_host[j].y;		
  }

  secondDerivative(aux_2);
  secondDerivative(aux_3);
  
 //First step
	
  for( int j=0;j<NY;j++){
	
    uw_host[j].x=(alpha[in]+betha[in])*(1.0/REYNOLDS)*aux_2[j].x+dseda[in]*N_host[j].x;
	Ww_host[j].x=(alpha[in]+betha[in])*(1.0/REYNOLDS)*aux_3[j].x+dseda[in]*NW_host[j].x;	
  }
END_RANGE_ASYNC
  return;

}
//static int printnow=0;
void meanURKstep_2(float dt, int in, domain_t domain, int counter, paths_t path){
START_RANGE_ASYNC("meanURKstep_2",34)

  //End

  for(int j=0;j<NY;j++){
    uw_host[j].x=(double)dt*(uw_host[j].x+gammad[in]*N_host[j].x);	
	Ww_host[j].x=(double)dt*(Ww_host[j].x+gammad[in]*NW_host[j].x);		
  }
	
  //Calc implicit step: solve tridiagonal 

 implicitStepMeanU(uw_host,betha[in],dt,domain);
 implicitStepMeanU(Ww_host,betha[in],dt,domain);
	
  for(int j=0;j<NY;j++){
    u_host[j].x=u_host[j].x+uw_host[j].x;
    w_host[j].x=w_host[j].x+Ww_host[j].x;	
  }
	
  forcing(u_host, domain);

  //END OF THE RK STEP

 if(in==2){

    double Umean=(double)QVELOCITY/LY;
    double nu=1.0/(double)REYNOLDS;
    double N2=(double)NX*(2*NZ-2);
    double u_tau;
    char thispath[100];

    //Statistics



    u_tau=sqrt(0.5*(pow(Utau_1.x,2.0)+pow(Utau_2.x,2.0)));

    if (counter%path.freq_print == 0){
      printf("\n****MEAN_PROFILE_STATISTICS****");
      printf("\n(tau_1,tau_2)=(%e,%e)",Utau_1.x,Utau_2.x);	
      printf("\nNu=%e",nu);
      printf("\n(RE_t,RE_c,RE_m)=(%e,%e,%e)",u_tau*LY*0.5/nu,u_host[NY/2].x*0.5*LY/(nu*N2),Umean*LY/nu);
      printf("\n(Dx+,Dz+)=(%e,%e)",(3.0/2.0)*u_tau*LX/(nu*NX),(3.0/2.0)*u_tau*LZ/(nu*(2*NZ-2)));
      printf("\nDy+(max,min)=(%f,%f)",u_tau*(Fmesh((NY/2+1)*DELTA_Y-1.0)-Fmesh((NY/2)*DELTA_Y-1.0))/(nu),u_tau*(Fmesh(1*DELTA_Y-1.0)-Fmesh(0*DELTA_Y-1.0))/(nu));	
      printf("\nC_f=%e",2.0*u_tau/(Umean*Umean));
      printf("\n(Um+,Ux+,Um/Uc)=(%f,%f,%f)",Umean/u_tau,u_host[NY/2].x/(u_tau*N2),Umean*N2/u_host[NY/2].x);
      printf("\n");
    }


    if (counter%path.freq_stats==0){
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
	  fprintf(fp," %e",u_host[j].x);
	}
      fprintf(fp," \n");
    
      
      for(int j=0;j<NY;j++){
	fprintf(fp1," %e",N_host[j].x);
      }
      fprintf(fp1," \n");
	
      fprintf(fp2," %e",u_tau);

      fprintf(fp3,"%e %e %e %e %e %e %e %e ",u_tau*LY*0.5/nu,u_host[NY/2].x*0.5*LY/(nu*N2),1.0/nu,Umean/u_tau,
	      u_host[NY/2].x/(u_tau*N2),Umean*N2/u_host[NY/2].x,
	      2.0*u_tau*u_tau/(Umean*Umean),2.0*u_tau*u_tau/(u_host[NY/2].x*u_host[NY/2].x));
      fprintf(fp3,"\n");
      
      fprintf(fp4,"%e %e %e %e",(3.0/2.0)*u_tau*LX/(nu*NX),(3.0/2.0)*u_tau*LZ/(nu*NZ),
	      u_tau*(Fmesh((NY/2+1)*DELTA_Y-1.0)-Fmesh((NY/2)*DELTA_Y-1.0))/(nu),u_tau*(Fmesh(1*DELTA_Y-1.0)-Fmesh(0*DELTA_Y-1.0))/(nu));
      fprintf(fp4,"\n");
	
      fclose(fp);
      fclose(fp1);
      fclose(fp2);
      fclose(fp3);
      fclose(fp4);
    }

  }
	
	
END_RANGE_ASYNC		

  return;

}



