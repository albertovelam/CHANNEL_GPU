#include"channel.h"

void calcNL(float2* ddv,float2* g,float2* R_ddv,float2* R_g,float2* u,float2* v,float2* w,float2* dv,int ii){

	

	//Calculate (wy,d_y v)---> u_x, u_z
	calcUW(u,w,dv,g);	

	if(ii==0){
	//calcSt(dv,u,v,w);
	//calcSpectra(dv,u,v,w);
	}

	//Introduce <u> in u_x
	if(RANK==0){
	writeUmean(u);}

	//Calculate nonlineal terms
	convolution(u,v,w,R_ddv,dv,R_g);

	//read mean R for computation of mean profile
	if(RANK==0){	
	readNmean(R_ddv);
	}

	//Calculate Hg and Hv
	calcHvg(R_ddv,dv,R_g);

	return;

}		




















