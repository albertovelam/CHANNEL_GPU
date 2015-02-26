#include "channel.h"

static cufftHandle fft2_r2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft2_sum; 

static cublasHandle_t cublasHandle;


//float2* u_host_1;
//float2* u_host_2;

float2* aux_host_1[6];
float2* aux_host_2[6];
float2* aux_dev[6];

int stream_idx=0;
cudaStream_t compute_stream;
cudaStream_t h2d_stream;
cudaStream_t d2h_stream;
cudaEvent_t events[1000];

static int MPIErr;

void fftSetup(domain_t domain)
{
  int nsum[2]={NXSIZE,2*NZ-2};
  int nRows = NX;
  int nCols = 2*NZ-2;
  int n2[2]={nRows, nCols};
  int idist = nRows*2*(nCols/2+1);//nRows*nCols;
  int odist = nRows*(nCols/2+1);
  int inembed[2] = {nRows, 2*(nCols/2+1) };//{nRows, nCols    };
  int onembed[2] = {nRows,    nCols/2+1  };
  int istride = 1;
  int ostride = 1;

  CHECK_CUFFT( cufftPlanMany( &fft2_r2c,2,n2,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,NYSIZE) );
  CHECK_CUFFT( cufftPlanMany( &fft2_c2r,2,n2,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,NYSIZE) );
  CHECK_CUFFT( cufftPlanMany(&fft2_sum,2,nsum,NULL,1,0,NULL,1,0,CUFFT_R2C,NY) );

  CHECK_CUDART( cudaStreamCreate(&compute_stream) );
  CHECK_CUDART( cudaStreamCreate(&h2d_stream) );
  CHECK_CUDART( cudaStreamCreate(&d2h_stream) );
  for(int i=0; i<1000; i++) CHECK_CUDART( cudaEventCreateWithFlags( &events[i], cudaEventDisableTiming) );


  float2* host_buffer  = (float2*)malloc(12*SIZE);
  CHECK_CUDART( cudaHostRegister(host_buffer,12*SIZE,0) );
  float2* dev_buffer;
  CHECK_CUDART( cudaMalloc((void**)&dev_buffer,6*SIZE) );

  for(int i=0;i<6;i++){
    aux_host_1[i]=(float2*)host_buffer + (size_t)i*2*SIZE/sizeof(float2);
    aux_host_2[i]=(float2*)host_buffer + (size_t)i*2*SIZE/sizeof(float2) + (size_t)SIZE/sizeof(float2);
    aux_dev[   i]= (float2*)dev_buffer + (size_t)i*SIZE/sizeof(float2);
  }

  return;
}

void fftDestroy(void)
{
  cufftDestroy(fft2_r2c);
  cufftDestroy(fft2_c2r);
  
  return;
}

void fftBack1T_A(float2* u1, int stid,domain_t domain){
        int myNx = NY/MPISIZE;
        int myNy = NX/MPISIZE;
        stream_idx = stid;

        trans_yzx_to_zyx_yblock(u1, aux_dev[stid], compute_stream,domain);

        CHECK_CUDART( cudaEventRecord(events[stid],compute_stream) );
#ifndef USE_GPU_MPI
        CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[stid],0) );

       int iter;
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
         CHECK_CUDART(cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream));
         cudaEventRecord(events[30+128*stid+iter],d2h_stream);
       }
#endif
}
//static double timer;
void fftBack1T_B(float2* u1, int stid,domain_t domain){
         int myNx = NY/MPISIZE;
         int myNy = NX/MPISIZE;
         int stream_idx = stid;
//if(stid!=0){
//timer = MPI_Wtime()-timer; printf("%d gap time= %1.6f \n",RANK,timer);
//}
//        cublasSetStream(cublasHandle,compute_stream);
        cufftSetStream(fft2_c2r,compute_stream);
START_RANGE_ASYNC("MPI_A2A",7)
       int iter;
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
#ifndef USE_GPU_MPI
         MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
#else
         MPI_Irecv(              u1+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
#endif
       }
#ifndef USE_GPU_MPI
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[30+128*stid+iter]);
START_RANGE_ASYNC("MPI_Send",6)
         MPI_Send(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
END_RANGE_ASYNC
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
         CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
       }
END_RANGE_ASYNC
//timer = MPI_Wtime();
        CHECK_CUDART( cudaEventRecord(events[20+stid],h2d_stream) );
        CHECK_CUDART( cudaStreamWaitEvent(compute_stream,events[20+stid],0) );

        trans_zyx_to_zxy(aux_dev[stid], u1, compute_stream,domain);
        cufftExecC2R(fft2_c2r,u1,(float*)u1);
#else
       CHECK_CUDART( cudaEventSynchronize(events[stid]) );
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
START_RANGE_ASYNC("MPI_Send",6)
         MPI_Send(   aux_dev[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
END_RANGE_ASYNC
       }
       for(iter=1; iter<MPISIZE; iter++){
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
       }
END_RANGE_ASYNC
        trans_zyx_to_zxy(u1, aux_dev[stid], compute_stream,domain);
        cufftExecC2R(fft2_c2r,aux_dev[stid],(float*)u1);
#endif
}
void fftForw1T_A(float2* u1, int stid,domain_t domain){
        int myNx = NY/MPISIZE;
        int myNy = NX/MPISIZE;
        stream_idx = stid;

        cufftSetStream(fft2_r2c,compute_stream);
#ifndef USE_GPU_MPI
        cufftExecR2C(fft2_r2c,(float*)u1,(float2*)u1);

        trans_zxy_to_zyx(u1, aux_dev[stid], compute_stream,domain);
        CHECK_CUDART( cudaEventRecord(events[stid],compute_stream) );
        CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[stid],0) );

       int iter;
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
         CHECK_CUDART( cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream) );
         CHECK_CUDART( cudaEventRecord(events[30+128*stid+iter],d2h_stream) );
       }
#else
        cufftExecR2C(fft2_r2c,(float*)u1,(float2*)aux_dev[stid]);

        trans_zxy_to_zyx(aux_dev[stid], u1, compute_stream,domain);
        CHECK_CUDART( cudaEventRecord(events[stid],compute_stream) );

#endif
}

void fftForw1T_B(float2* u1, int stid,domain_t domain){
        int myNx = NY/MPISIZE;
        int myNy = NX/MPISIZE;
        stream_idx = stid;
START_RANGE_ASYNC("MPI_A2A",7)
       int iter;
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
#ifndef USE_GPU_MPI
         MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
#else
         MPI_Irecv(   aux_dev[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
#endif
       }
#ifndef USE_GPU_MPI
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
         CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+iter]) );
START_RANGE_ASYNC("MPI_Send",6)
         MPI_Send(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
END_RANGE_ASYNC
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
         CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
       }
END_RANGE_ASYNC
        cudaEventRecord(events[20+stid],h2d_stream);
        cudaStreamWaitEvent(compute_stream,events[20+stid],0);
        trans_zyx_yblock_to_yzx(aux_dev[stid], u1, compute_stream,domain);
#else
       CHECK_CUDART( cudaEventSynchronize(events[stid]) );
       for(iter=1; iter<MPISIZE; iter++){
         int dest = RANK ^ iter;
START_RANGE_ASYNC("MPI_Send",6)
         MPI_Send(              u1+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
END_RANGE_ASYNC
       }
       for(iter=1; iter<MPISIZE; iter++){
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
       }
END_RANGE_ASYNC
        trans_zyx_yblock_to_yzx(aux_dev[stid], u1, compute_stream,domain);
#endif
}


void fftForward(float2* buffer, domain_t domain)
{
START_RANGE("FFT_F",3)
  CHECK_CUFFT( cufftExecR2C(fft2_r2c,(float*)buffer,(float2*)buffer) );
END_RANGE	
  return;
}

void fftBackward(float2* buffer, domain_t domain)
{
START_RANGE("FFT_B",4)
  CHECK_CUFFT( cufftExecC2R(fft2_c2r,(float2*)buffer,(float*)buffer) );
END_RANGE		
  return;	

}

void fftBackwardTranspose(float2* u2, domain_t domain){

	/*
	//Copy to host
	cudaCheck(cudaMemcpy(u_host_1,u2,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
	
	//Tranpose (x,z,y)-->(y,x,z)
	mpiCheck(chyzx2xyz((double*)u_host_1,(double*)u_host_2,NY,NX,NZ,RANK,MPISIZE),"Tr");

	//Copy to device
	cudaCheck(cudaMemcpy(u2,u_host_2,SIZE,cudaMemcpyHostToDevice),"MemInfo1");
	*/

  transposeYZX2XYZ(u2,
		   domain.ny,
		   domain.nx,
		   domain.nz,
		   domain.rank,
		   domain.size, domain);	

  fftBackward(u2,domain);
		
  return;

}

void fftForwardTranspose(float2* u2, domain_t domain){
  
  
  fftForward(u2,domain);	
	
  transposeXYZ2YZX(u2,
		   domain.ny,
		   domain.nx,
		   domain.nz,
		   domain.rank,
		   domain.size,domain);	
  
  /*
  //Copy to host
  cudaCheck(cudaMemcpy(u_host_1,u2,SIZE,cudaMemcpyDeviceToHost),"MemInfo1");
  
  //Tranpose (y,x,z)-->(x,z,y)
  mpiCheck(chxyz2yzx((double *)u_host_1,(double *)u_host_2,NY,NX,NZ,RANK,MPISIZE),"Tr");
  
  //Copy to device
  cudaCheck(cudaMemcpy(u2,u_host_2,SIZE,cudaMemcpyHostToDevice),"MemInfo1");
  */
  
  return;
  
}

/*
  void forwardTranspose(float2* u2){
  
  //Traspuesta de COMPLEJA
  
  //cufftCheck(cufftExecR2C(fft2_r2c,(float*)u2,(float2*)aux),"forward transform");
  
  
  //fftForward(u2);	
	
  //Transpuesta de [j,i,k][NY,NX,NZ] a -----> [i,k,j][NX,NZ,NY]
  transpose_B(aux,u2);
  cudaCheck(cudaMemcpy(u2,aux,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
  
  return;
  
  }
  
void backwardTranspose(float2* u2){

//Traspuesta de COMPLEJA

//Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
transpose_A(aux,u2);


cudaCheck(cudaMemcpy(u2,aux,SIZE,cudaMemcpyDeviceToDevice),"MemInfo1");
//fftBackward(u2);

//cufftCheck(cufftExecC2R(fft2_c2r,(float2*)aux,(float*)u2),"forward transform");

return;

}
*/

void calcUmax(float2* u_x,float2* u_y,float2* u_z,float* ux,float* uy,float* uz, domain_t domain)
{
START_RANGE("calcUmax",8)

	int size_l=2*NXSIZE*NY*NZ;
	int index;

	CHECK_CUBLAS( cublasIsamax (cublasHandle,size_l, (const float *)u_x,1,&index) );
	CHECK_CUDART( cudaMemcpy(ux,(float*)u_x+index-1, sizeof(float), cudaMemcpyDeviceToHost) );
	
	CHECK_CUBLAS( cublasIsamax (cublasHandle,size_l, (const float *)u_y,1,&index) );
	CHECK_CUDART( cudaMemcpy(uy,(float*)u_y+index-1, sizeof(float), cudaMemcpyDeviceToHost) );
	
	CHECK_CUBLAS( cublasIsamax (cublasHandle,size_l, (const float *)u_z,1,&index) );
	CHECK_CUDART( cudaMemcpy(uz,(float*)u_z+index-1, sizeof(float), cudaMemcpyDeviceToHost) );
	
	*ux=fabs(*ux);
	*uy=fabs(*uy);
	*uz=fabs(*uz);
	
	//MPI reduce
	reduceMAX(ux,uy,uz);

END_RANGE
	return;

}

void calcDmax(float2* u_x,float2* u_y,float* ux,float* uy,domain_t domain)
{
START_RANGE("calcDmax",9)

	int size_l=2*NXSIZE*NY*NZ;
	int index;

		
	CHECK_CUBLAS( cublasIsamax (cublasHandle,size_l, (const float *)u_x,1,&index) );
	CHECK_CUDART( cudaMemcpy(ux,(float*)u_x+index-1, sizeof(float), cudaMemcpyDeviceToHost) );
	
	CHECK_CUBLAS( cublasIsamax (cublasHandle,size_l, (const float *)u_y,1,&index) );
	CHECK_CUDART( cudaMemcpy(uy,(float*)u_y+index-1, sizeof(float), cudaMemcpyDeviceToHost) );
	
	*ux=fabs(*ux);
	*uy=fabs(*uy);

	float* uz=(float*)malloc(sizeof(float));
	*uz=0;
	
	//MPI reduce
	reduceMAX(ux,uy,uz);

	free(uz);		
END_RANGE
	return;

}



float sumElementsReal(float2* buffer_1, domain_t domain){
START_RANGE("sumElementsReal",10)
	//destroza lo que haya en el buffer

	float2 sum[NYSIZE];
	float sum_all=0;
	
	//Transformada NYSIZE*[NX*NZ]
	CHECK_CUFFT( cufftExecR2C(fft2_r2c,(float*)(buffer_1),buffer_1) );

	for(int i=0;i<NYSIZE;i++){
	  CHECK_CUDART( cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NX*NZ,sizeof(float2),cudaMemcpyDeviceToHost) );
	}

	for(int i=1;i<NYSIZE;i++){
          sum[0].x+=sum[i].x;
	}

	//MPI SUM
	reduceSUM((float*)sum,&sum_all);

END_RANGE
	return sum_all;
}

void sumElementsComplex(float2* buffer_1,float* out, domain_t domain){
START_RANGE("sumElementsComplex",11)
	//destroza lo que haya en el buffer

	float sum_all=0;
	float2 sum[NY];
	float2 sum1[NY];
	float2 sum2[NY];

	//Transpose [NXSIZE,NZ,NY] to [NY,NXSIZE,NZ]	
	transpose((float2*)AUX,(const float2*)buffer_1,NY,NXSIZE*NZ, domain);

	//Transformada NYSIZE*[NX*NZ]
	CHECK_CUFFT( cufftExecR2C(fft2_sum,(float*)(AUX),(float2*)AUX) );

	//Transpose [NXSIZE,NZ,NY] to [NY,NXSIZE,NZ]	
	transpose((float2*)buffer_1,(const float2*)AUX,NXSIZE*NZ,NY,domain);	

	CHECK_CUDART( cudaMemcpy((float2*)sum,(float2*)buffer_1,NY*sizeof(float2),cudaMemcpyDeviceToHost) );


 	mpiCheck(MPI_Allreduce((float*)sum,(float*)sum2,2*NY,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD),"caca");	

	for(int i=0;i<NY;i++){
	out[i]=sum2[i].x;
	}

END_RANGE
	return;

}
