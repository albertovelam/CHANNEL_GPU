#include"channel.h"


float2* ddv_w; 
float2* g_w;
float2* R_ddv;
float2* R_g;
float2* dv;

float2 *u;
float2 *v;
float2 *w;

float* umax;
float* gmax;
float* umax_d;
float* gmax_d;

const float betha[]={ 37.0f/160.0f, 5.0f/24.0f, 1.0f/6.0f};


void setRK3(domain_t domain){


  //Memory allocation
  //8 buffers of size SIZE single precision

  CHECK_CUDART( cudaMalloc(&ddv_w,SIZE) );
  CHECK_CUDART( cudaMalloc(&g_w,SIZE) );
	
  CHECK_CUDART( cudaMalloc(&R_ddv,SIZE) );
  CHECK_CUDART( cudaMalloc(&R_g,SIZE) );

  CHECK_CUDART( cudaMalloc(&u,SIZE) );
  CHECK_CUDART( cudaMalloc(&v,SIZE) );
  CHECK_CUDART( cudaMalloc(&w,SIZE) );
  CHECK_CUDART( cudaMalloc(&dv,SIZE) );

  set2zero(ddv_w,domain);
  set2zero(g_w,domain);

  set2zero(R_ddv,domain);
  set2zero(R_g,domain);
	
  set2zero(u,domain);
  set2zero(v,domain);
  set2zero(w,domain);	

  set2zero(dv,domain);

  umax=(float*)malloc(3*sizeof(float));
  gmax=(float*)malloc(2*sizeof(float));
  CHECK_CUDART( cudaHostRegister( umax, 3*sizeof(float),0 ) );
  CHECK_CUDART( cudaHostRegister( gmax, 2*sizeof(float),0 ) );
  CHECK_CUDART( cudaMalloc((void**)&umax_d, 3*sizeof(float) ) );
  CHECK_CUDART( cudaMalloc((void**)&gmax_d, 2*sizeof(float) ) );

return;

}

void calcVdV(float2* ddv,float2* v,float2* dv, domain_t domain){
START_RANGE("calcVdV",2)

  CHECK_CUDART( cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice) );
  hemholztSolver_double(v,domain);
  CHECK_CUDART( cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice) );
  deriv_Y_HO_double(dv,domain);

END_RANGE
  return;
}
static float calcDt(float2* u_x,float2* u_y,float2* u_z,float2* ddv,float2* g, domain_t domain, int counter,  paths_t path){
START_RANGE("calcDt",3)

  float N2=NX*(2*NZ-2);
  float CFL=0.5f;
  
  float dt;
  
//  float* umax=(float*)malloc(3*sizeof(float));
//  float* gmax=(float*)malloc(2*sizeof(float));
  
  //Calc max_velocity
  
  int size_l=2*NXSIZE*NY*NZ;
  int index;
  
  //calcUmax(u_x,u_y,u_z,umax,umax+1,umax+2,domain);
  //calc_Umax2(u_x, u_y, u_z, float* temp,domain_t domain);
  //calcDmax(ddv,g,gmax,gmax+1,domain);
  
  
  //Print
  
  float c=((NX/(2.0f*acos(-1.0f)*LX))*fabs(umax[0]/N2)+fabs(umax[1]/N2)/DELTA_Y+(NX/(2.0f*acos(-1.0f)*LZ))*fabs(umax[2]/N2));
  dt=CFL/c;		
  
  float dt_v=CFL*REYNOLDS/(1.0f/(DELTA_Y*DELTA_Y)+(1.0f/3.0f*NX/LX)*(1.0f/3.0f*NX/LX)+(1.0f/3.0f*NZ/LZ)*(1.0f/3.0f*NZ/LZ));
  
  if(domain.rank==0){
if(counter%path.freq_print==0){
    printf("*****RK_STATISTICS****");
    printf("\nmax_V=(%e,%e,%e)",umax[0]/N2,umax[1]/N2,umax[2]/N2);
    printf("\nmax_(ddV,G)=(%e,%e)",gmax[0],gmax[1]);
    printf("\n(dt_c,dt_v)=(%f,%f)",dt,dt_v);
    printf("\n");
}		
  }
  
  //Print
  
//  free(umax);
//  free(gmax);
  
  dt=fmin(dt,dt_v);
//END_RANGE  
  //dt=0.0f;
  return dt;
  
}

void RKstep(float2* ddv,float2* g,float time, domain_t domain, paths_t path){
  
  static float time_elapsed=0.0f;
  float dt=0;
  float dt_2=0;
  int counter=0;
  int frec=10;
  
  //Calc initial simulation
  imposeSymetry(ddv,g,domain);
  calcVdV(ddv,v,dv,domain);
  //while(time_elapsed<time){
  while(counter<path.nsteps){
START_RANGE("RKstep",counter%2)
double timer=MPI_Wtime();
    for(int n_step=0;n_step<3;n_step++){
START_RANGE("RKsubstep",n_step+3)
      //First step		
      RKstep_1(ddv,g,ddv_w,g_w,R_ddv,R_g,dt,n_step,domain);	

      //Mean step
      if(domain.rank==0){
	meanURKstep_1(n_step,domain);
      }
END_RANGE 

      //Calc non linear terms stored in R_1 and R_2
      calcNL(ddv,g,R_ddv,R_g,u,v,w,dv,n_step,counter,domain,path);
      
      //Time step
      if(n_step==0){
END_RANGE
END_RANGE
	dt_2=calcDt(u,v,w,ddv,g,domain,counter,path);	
      }
      
      //Second step
      if(domain.rank==0){
	meanURKstep_2(dt,n_step,domain,counter,path);
      }
if(n_step==0){
END_RANGE     
}else{
END_RANGE
END_RANGE
} 
      //Second step
      RKstep_2(ddv_w,g_w,R_ddv,R_g,dt,n_step,domain);	
      
      //Implicit step in u
      implicitSolver_double(g_w,betha[n_step],dt,domain);
      bilaplaSolver_double(ddv_w,v,dv,u,g,w,betha[n_step],dt,domain);	
      
      //Copy to final buffer		
      CHECK_CUDART( cudaMemcpy(ddv,ddv_w,SIZE,cudaMemcpyDeviceToDevice) );
      CHECK_CUDART( cudaMemcpy(  g,  g_w,SIZE,cudaMemcpyDeviceToDevice) );

END_RANGE
    }
    

    if(counter%frec==0){
      //enstrophyH5(ddv,g,v,dv,ddv_w,g_w,R_ddv,R_g,counter/frec);
    }
    
    if(counter%1000==0){
      imposeSymetry(ddv,g,domain);
    }
    
    //Advance
    dt=dt_2;
    time_elapsed+=dt;
    counter++;

timer = MPI_Wtime()-timer;
    
    if(domain.rank==0){
if((counter-1)%path.freq_print==0){
      printf("\033[0;31m*** wall_time: %1.6f sec (time,counter)=(%f,%d)\033[0m\n\n",timer,time_elapsed,counter);
      //printf("\n(time,counter)=(%f,%d)",time_elapsed,counter);
}
    }
END_RANGE    
  }
  
  
  return;
  
  
}


