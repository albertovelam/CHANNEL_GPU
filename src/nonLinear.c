#include"channel.h"

void calcNL(float2* ddv,float2* g,float2* R_ddv,float2* R_g,
	    float2* u,float2* v,float2* w,
	    float2* dv,int ii,int counter, domain_t domain,
	    paths_t path){
START_RANGE("calcNL",21)  
  //Calculate (wy,d_y v)---> u_x, u_z
  calcUW(u,w,dv,g,domain);	
  
  if(ii==0 && counter%path.freq_stats==0){
    calcSt(R_ddv,dv,R_g,u,v,w,domain,path);
    //calcSpectra(dv,u,v,w);
  }
  
  //Introduce <u> in u_x and <w> in u_z
  if(domain.rank==0){
    writeUmean(u,w,domain);
  }

//  if(ii==0){
    convolution_max(ii,ddv,g,u,v,w,R_ddv,dv,R_g,domain);
//  }else{
    //Calculate nonlineal terms
//    convolution(u,v,w,R_ddv,dv,R_g,domain);
//  }

  //read mean R for computation of mean profile
  if(domain.rank==0){	
    readNmean(R_ddv,R_g,domain);
  }
  
  //Calculate Hg and Hv
  calcHvg(R_ddv,dv,R_g,domain);
//END_RANGE_ASYNC 
  return;
  
}		




















