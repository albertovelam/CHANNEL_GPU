#include "channel.h"



void imposeSymetry(float2* u,float2* v, domain_t domain){

  fftBackwardTranspose(u, domain);
  fftBackwardTranspose(v, domain);

  fftForwardTranspose(u, domain);
  fftForwardTranspose(v, domain);

  normalize(u,domain);
  normalize(v,domain);

  return;
  
}

