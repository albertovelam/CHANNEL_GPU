#include "../src/channel.h"
#include <string.h>

void mean_interpolator(char *, char *, int, int, int, int, int, int);

int main(int argc, char* argv[]){
  int i;
  int Nxold, Nyold, Nzold;

  if (strcmp(argv[1],"--help") == 0){
    printf("Usage: meanpolator input output NXold NYold NZold\n");
  }
  else{
    Nxold = atoi(argv[3]);
    Nyold = atoi(argv[4]);
    Nzold = atoi(argv[5]);
    
    printf("Interpolating from %i to %i\n", Nyold, NY);
    mean_interpolator(argv[1],argv[2],Nxold,Nyold,Nzold,NX,NY,NZ*2);
  }
  
  return 0;
}

void mean_interpolator(char *filenamein, char *filenameout, int Nx,
		       int Ny, int Nz, int Nxnew, int Nynew, int Nznew){
  double *meanin;
  double *meanout;
  int j;
  int idx1;
  double w;
  size_t fsize;
  
  meanin = (double *) malloc(Ny*sizeof(double2));
  meanout = (double *) malloc(Nynew*sizeof(double2));
  
  FILE* fp_r;
  FILE* fp_w;
  
  fp_r=fopen(filenamein,"rb");
  if(fp_r==NULL){
    printf("Error reading file: %s\n",filenamein);
    exit(1);
  }
  fsize = fread( (unsigned char*)meanin,sizeof(double2),Ny,fp_r);
  if(fsize!=Ny){
    printf("Wrong size: %s,%d\n",filenamein,fsize);
    exit(1);
  }
  fclose(fp_r);

  meanout[0] = meanin[0];
  meanout[1] = 0.0;
  meanout[Nynew-2] = meanin[Ny-2];
  meanout[Nynew-1] = 0.0;
  
  for (j=1; j<Nynew-1; j++){
    idx1 = (int) ((double)j*((double)Ny-1.0)/((double)Nynew-1.0));
    w = 1.0 - fmod(((double)j*((double)Ny-1.0)/((double)Nynew-1.0)),1.0);
    meanout[2*j] = (w*meanin[2*idx1] + (1-w)*meanin[2*(idx1+1)])*(Nxnew*Nznew)/(Nx*Nz);
    meanout[2*j+1] = 0.0;
  }
  

  fp_w=fopen(filenameout,"wb");
  if(fp_w==NULL){
    printf("error escritura: %s\n",filenameout);
    exit(1);
  }
  fsize =fwrite( (unsigned char*)meanout,sizeof(double2),Nynew,fp_w);
  if(fsize!=NY){
    printf("writing error: %s\n",filenameout);
    exit(1);
  }
  fclose(fp_w);
  
}
