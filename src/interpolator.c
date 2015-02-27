#include "channel.h"
#include <string.h>

int main(argc, char* argv[]){
  int i;
  char filenamein[100];
  char filenameout[100];
  int Nxold, Nzold, Nznew;

  if (strcmp(argv[1],'--help') == 0){
    printf("Usage: interpolate input output NXold NYold NZold\n");
  }
  
  strcpy(&filenamein,argv[1]);
  strcpy(&filenameout,argv[2]);
  Nxold = *argv[3];
  Nyold = *argv[4];
  Nzold = *argv[5];

  printf("Interpolating from %i,%i,%i, to %i,%i,%i\n", NXold, NYold, NZold, NX, NY, NZ*2);
  xz_interpolate_float(filenamein,filenameout,Nxold,Nyold,Nzold,NY,NX,NZ*2);
  
  return 0
}


void interpolate_float(char *filenamein, char *filenameout, int Nx,
		       int Ny, int Nz, int Nxnew, int Nynew, int Nznew){
  /*    IMPORTANT     

     Nx in this routine corresponds to NX in the channel
     Ny in this routine corresponds to NZ in the channel
     Nz in this routine corresponds to NY in the channel

   */
  
  int i,j,k;
  int MPIErr;
  herr_t H5Err;
  hid_t ifileid, idsetid, idspaceid, imspaceid;
  hid_t ofileid, odsetid, odspaceid, omspaceid;
  hsize_t isizem[3], isized[3], istart[3], istride[3], icount[3];
  hsize_t osizem[3], osized[3], ostart[3], ostride[3], ocount[3];
  float *aux, *aux1;
  
  aux = (float *) malloc(Ny*Nz*sizeof(float));
  aux1 = (float *) malloc(Nynew*Nz*sizeof(float));


  ifileid = H5Fopen(filenamein, H5F_ACC_RDONLY, H5P_DEFAULT);
  istart[0] = 0; istart[1] = 0; istart[2] = 0;
  isized[0] = Nx; isized[1] = Ny; isized[2] = Nz;
  isizem[0] = 1; isizem[1] = Ny; isizem[2] = Nz;
  istride[0] = 1; istride[1] = Ny; istride[2] = Nz;
  icount[0] = 1; icount[1] = 1; icount[2] = 1;
  idsetid = H5Dopen(ifileid, "u", H5P_DEFAULT);
  idspaceid = H5Dget_space(idsetid);

  ofileid = H5Fcreate(filenameout, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  ostart[0] = 0; ostart[1] = 0; ostart[2] = 0;
  osized[0] = Nxnew; osized[1] = Nynew; osized[2] = Nz;
  osizem[0] = 1; osizem[1] = Nynew; osizem[2] = Nz;
  ostride[0] = 1; ostride[1] = Nynew; ostride[2] = Nz;
  ocount[0] = 1; ocount[1] = 1; ocount[2] = 1;
  odspaceid = H5Screate_simple(3,osized,osized);
  odsetid = H5Dcreate(ofileid, "u", H5T_NATIVE_FLOAT, odspaceid,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


  
  for (i=0; i<Nxnew; i++){
    for (j=0; j<Nynew; j++){
      for (k=0; k<Nz; k++){
	aux1[j*Nz + k] = 0.0f; /* Fill output array with zeros */
      }
    }
    printf("Processing plane %d\n",i);
    if (i < Nx){
      imspaceid = H5Screate_simple(3,isizem,isizem);
      istart[0] = 0;
      H5Err = H5Sselect_hyperslab(imspaceid,H5S_SELECT_SET,
				  istart,istride,icount,isizem);
      istart[0] = i;
      H5Err = H5Sselect_hyperslab(idspaceid,H5S_SELECT_SET,
				  istart,istride,icount,isizem);
      H5Err = H5Dread(idsetid,H5T_NATIVE_FLOAT,imspaceid,idspaceid,
		      H5P_DEFAULT,aux);
      H5Err = H5Sclose(imspaceid);
      if (Nynew > Ny){
	for (j=0; j<Ny; j++){
	  for (k=0; k<Nz; k++){
	    aux1[j*Nz + k] = aux[j*Nz + k];
	  }
	}
      }
      else{
	for (j=0; j<Nynew; j++){
	  for (k=0; k<Nz; k++){
	    aux1[j*Nz + k] = aux[j*Nz + k];
	  }
	}
      }
    }
    
    omspaceid = H5Screate_simple(3,osizem,osizem);
    ostart[0] = 0;
    H5Err = H5Sselect_hyperslab(omspaceid,H5S_SELECT_SET,
				ostart,ostride,ocount,osizem);
    ostart[0] = i;
    H5Err = H5Sselect_hyperslab(odspaceid,H5S_SELECT_SET,
				ostart,ostride,ocount,osizem);
    H5Err = H5Dwrite(odsetid,H5T_NATIVE_FLOAT,omspaceid,odspaceid,
		     H5P_DEFAULT,aux1);
    H5Err = H5Sclose(omspaceid);
    
  }
  
  
  H5Err = H5Dclose(idsetid);
  H5Err = H5Sclose(idspaceid);
  H5Err = H5Fclose(ifileid);
  H5Err = H5Dclose(odsetid);
  H5Err = H5Sclose(odspaceid);
  H5Err = H5Fclose(ofileid);

  free(aux);
  free(aux1);

}

