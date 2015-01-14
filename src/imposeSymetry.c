#include "channel.h"



void imposeSymetry(float2* u,float2* v){

        fftBackwardTranspose(u);
        fftBackwardTranspose(v);

        fftForwardTranspose(u);
        fftForwardTranspose(v);

        normalize(u);
        normalize(v);

        return;

}

