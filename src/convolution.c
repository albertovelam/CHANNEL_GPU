#include "channel.h"

//extern float2* aux_dev[6];
extern float* umax;
extern float* gmax;
extern float* umax_d;
extern float* gmax_d;

void convolution_max(int ii, float2* ddv, float2* g, float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz, domain_t domain){
START_RANGE("convolution_max",4)
  CHECK_CUDART( cudaMemcpy(wz,ux,SIZE,cudaMemcpyDeviceToDevice) );
  deriv_Y_HO_double(wz,domain);

  CHECK_CUDART( cudaMemcpy(wx,uz,SIZE,cudaMemcpyDeviceToDevice) );
  deriv_Y_HO_double(wx,domain);

  if(domain.rank==0){
    readUtau(wz,domain);
  }
  calcOmega(wx,wy,wz,ux,uy,uz,domain);

        fftBack1T_A(ux,0,domain);
        fftBack1T_A(uy,1,domain);
        fftBack1T_A(wx,3,domain);
        fftBack1T_A(wy,4,domain);

        fftBack1T_A(uz,2,domain);
        fftBack1T_A(wz,5,domain);

        fftBack1T_B(ux,0,domain);
        fftBack1T_B(uy,1,domain);
        fftBack1T_B(wx,3,domain);
        fftBack1T_B(wy,4,domain);

        calcRotor3(wx,wy,wz,ux,uy,uz,domain);
        fftForw1T_A(wz,4,domain);
        fftBack1T_B(uz,2,domain);
if(ii==0){
        CHECK_CUDART( cudaMemsetAsync(umax_d, 0, 3*sizeof(float), compute_stream) );
        CHECK_CUDART( cudaMemsetAsync(gmax_d, 0, 2*sizeof(float), compute_stream) );
        calc_Umax2(ux,uy,uz,umax_d,domain);
        calc_Dmax2(ddv,g,gmax_d,domain);
        CHECK_CUDART( cudaEventRecord(events[0],compute_stream) );
        CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[0],0) );
        CHECK_CUDART( cudaMemcpyAsync(umax,umax_d,3*sizeof(float),cudaMemcpyDeviceToHost, d2h_stream) );
        CHECK_CUDART( cudaMemcpyAsync(gmax,gmax_d,2*sizeof(float),cudaMemcpyDeviceToHost, d2h_stream) );
        CHECK_CUDART( cudaEventRecord(events[999],d2h_stream) );
}
        fftBack1T_B(aux_dev[0],5,domain);
  calcRotor12(wx,wy,aux_dev[0],ux,uy,uz,domain);
        fftForw1T_A(wx,0,domain);
        fftForw1T_A(wy,1,domain);
        fftForw1T_B(wz,4,domain);

        fftForw1T_B(wx,0,domain);
        fftForw1T_B(wy,1,domain);
  dealias(wx,domain);
  dealias(wy,domain);
  dealias(wz,domain);
if(ii==0){
        CHECK_CUDART( cudaEventSynchronize(events[999]) );
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE,&umax[0],1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE,&umax[1],1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE,&umax[2],1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE,&gmax[0],1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE,&gmax[1],1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD),"caca");
}

        //reduceMAX(&umax[0],&umax[1],&umax[2]);
END_RANGE
  return;
}


#if 0
void convolution(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz, domain_t domain){
START_RANGE("convolution",4)

	// Derivadas respeto de y
  cudaCheck(cudaMemcpy(wz,ux,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");
  deriv_Y_HO_double(wz,domain);
  
  cudaCheck(cudaMemcpy(wx,uz,SIZE,cudaMemcpyDeviceToDevice),domain,"MemInfo1");
  deriv_Y_HO_double(wx,domain);	

  //Read U_tau
  if(domain.rank==0){
    readUtau(wz,domain);
  }	
  
  // Calcular omega en fourier
  
  calcOmega(wx,wy,wz,ux,uy,uz,domain);
  
  // Transform to real	
/*  
  fftBackwardTranspose(ux, domain);	
  fftBackwardTranspose(uy, domain);	
  fftBackwardTranspose(uz, domain);	
  
  
  fftBackwardTranspose(wx, domain);	
  fftBackwardTranspose(wy, domain);	
  fftBackwardTranspose(wz, domain);	
*/
        fftBack1T_A(ux,0,domain);
        fftBack1T_A(uy,1,domain);
        fftBack1T_A(wx,3,domain);
        fftBack1T_A(wy,4,domain);

        fftBack1T_A(uz,2,domain);
        fftBack1T_A(wz,5,domain);

        fftBack1T_B(ux,0,domain);
        fftBack1T_B(uy,1,domain);
        fftBack1T_B(wx,3,domain);
        fftBack1T_B(wy,4,domain);

        calcRotor3(wx,wy,wz,ux,uy,uz,domain);
        fftForw1T_A(wz,4,domain);

        fftBack1T_B(uz,2,domain);
//double timer=MPI_Wtime();
        fftBack1T_B(aux_dev[0],5,domain);
  
  // Calculate rotor
  
//  calcRotor(wx,wy,wz,ux,uy,uz,domain);

//  calcRotor3(wx,wy,aux_dev[5],ux,uy,uz,domain);
  calcRotor12(wx,wy,aux_dev[0],ux,uy,uz,domain);
//  calcRotor3(wx,wy,wz,ux,uy,uz,domain);  

  // Transform back to Fourier space		
/*  
  fftForwardTranspose(wx, domain);	
  fftForwardTranspose(wy, domain);	
  fftForwardTranspose(wz, domain);	
*/
///*
        fftForw1T_A(wx,0,domain);
        fftForw1T_A(wy,1,domain);
//        fftForw1T_A(wz,2,domain);
//timer=MPI_Wtime()-timer;
//if(RANK==0) printf("submit work time: %g sec \n",timer);
        fftForw1T_B(wz,4,domain);

        fftForw1T_B(wx,0,domain);
        fftForw1T_B(wy,1,domain);
//        fftForw1T_B(wz,2,domain);
//*/  
  //Dealias	
  
  dealias(wx,domain);
  dealias(wy,domain);
  dealias(wz,domain);
  
END_RANGE  
  return;
  
			
}
#endif



