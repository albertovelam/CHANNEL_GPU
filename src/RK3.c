#include"channel.h"


float2* ddv_w; 
float2* g_w;
float2* R_ddv;
float2* R_g;
float2* dv;

float2 *u;
float2 *v;
float2 *w;

const float betha[]={ 37.0f/160.0f, 5.0f/24.0f, 1.0f/6.0f};


void setRK3(domain_t domain){


  //Memory allocation
  //8 buffers of size SIZE single precision

  cudaCheck(cudaMalloc(&ddv_w,SIZE),domain,"malloc");
  cudaCheck(cudaMalloc(&g_w,SIZE),domain,"malloc");
	
  cudaCheck(cudaMalloc(&R_ddv,SIZE),domain,"malloc");
  cudaCheck(cudaMalloc(&R_g,SIZE),domain,"malloc");

  cudaCheck(cudaMalloc(&u,SIZE),domain,"malloc");
  cudaCheck(cudaMalloc(&v,SIZE),domain,"malloc");
  cudaCheck(cudaMalloc(&w,SIZE),domain,"malloc");
	
  cudaCheck(cudaMalloc(&dv,SIZE),domain,"malloc");

  set2zero(ddv_w,domain);
  set2zero(g_w,domain);

  set2zero(R_ddv,domain);
  set2zero(R_g,domain);
	
  set2zero(u,domain);
  set2zero(v,domain);
  set2zero(w,domain);	

  set2zero(dv,domain);


return;

}

void calcVdV(float2* ddv,float2* v,float2* dv, domain_t domain){


  cudaCheck(cudaMemcpy(v,ddv,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo_RK3_1");
  hemholztSolver_double(v,domain);
  cudaCheck(cudaMemcpy(dv,v,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1_RK3_2");
  deriv_Y_HO_double(dv,domain);


  return;
}

static float calcDt(float2* u_x,float2* u_y,float2* u_z,float2* ddv,float2* g, domain_t domain){


  float N2=NX*(2*NZ-2);
  float CFL=0.5f;
  
  float dt;
  
  float* umax=(float*)malloc(3*sizeof(float));
  float* gmax=(float*)malloc(2*sizeof(float));
  
  //Calc max_velocity
  
  int size_l=2*NXSIZE*NY*NZ;
  int index;
  
  calcUmax(u_x,u_y,u_z,umax,umax+1,umax+2,domain);
  calcDmax(ddv,g,gmax,gmax+1,domain);
  
  
  //Print
  
  float c=((NX/(2.0f*acos(-1.0f)*LX))*fabs(umax[0]/N2)+fabs(umax[1]/N2)/DELTA_Y+(NX/(2.0f*acos(-1.0f)*LZ))*fabs(umax[2]/N2));
  dt=CFL/c;		
  
  float dt_v=CFL*REYNOLDS/(1.0f/(DELTA_Y*DELTA_Y)+(1.0f/3.0f*NX/LX)*(1.0f/3.0f*NX/LX)+(1.0f/3.0f*NZ/LZ)*(1.0f/3.0f*NZ/LZ));
  
  if(domain.rank==0){
    printf("*****RK_STATISTICS****");
    printf("\nmax_V=(%e,%e,%e)",umax[0]/N2,umax[1]/N2,umax[2]/N2);
    printf("\nmax_(ddV,G)=(%e,%e)",gmax[0],gmax[1]);
    printf("\n(dt_c,dt_v)=(%f,%f)",dt,dt_v);
    printf("\n");		
  }
  
  //Print
  
  free(umax);
  free(gmax);
  
  dt=fmin(dt,dt_v);
  
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
  dealias(ddv,domain);
  dealias(g,domain);
  imposeSymetry(ddv,g,domain);
  calcVdV(ddv,v,dv,domain);
  int Nsteps=30000;
  
  //while(time_elapsed<time){
  while(counter<Nsteps){
    
    for(int n_step=0;n_step<3;n_step++){
      
      //First step		
      RKstep_1(ddv,g,ddv_w,g_w,R_ddv,R_g,dt,n_step,domain);	
      
      //Mean step
      if(domain.rank==0){
	meanURKstep_1(n_step);
      }
      
      //Calc non linear terms stored in R_1 and R_2
      calcNL(ddv,g,R_ddv,R_g,u,v,w,dv,n_step,counter,domain,path);
      
      //Time step
      if(n_step==0){
	dt_2=calcDt(u,v,w,ddv,g,domain);	
      }
      
      
      //Second step
      if(domain.rank==0){
	meanURKstep_2(dt,n_step,path);
      }
      
      //Second step
      RKstep_2(ddv_w,g_w,R_ddv,R_g,dt,n_step,domain);	
      
      //Implicit step in u
      implicitSolver_double(g_w,betha[n_step],dt,domain);
      bilaplaSolver_double(ddv_w,v,dv,betha[n_step],dt,domain);	
      
      //Copy to final buffer		
      cudaCheck(cudaMemcpy(ddv,ddv_w,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo_RK3_3");
      cudaCheck(cudaMemcpy(g,g_w,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo_RK3_4");
      dealias(ddv,domain);
      dealias(v,domain);
      dealias(g,domain);
      dealias(dv,domain);
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
    
    if(domain.rank==0){
      printf("\n(time,counter)=(%f,%d)",time_elapsed,counter);
    }
    
  }
  
  
  return;
  
  
}


