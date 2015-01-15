CC = mpic++ -O3 -march=corei7-avx -mtune=corei7-avx -ffast-math
NVCC = nvcc -O3  -arch=compute_35 -code=sm_35
LD = mpic++ -O3 -march=corei7-avx -mtune=corei7-avx
LIBS = -lcusparse -lcudart -lcufft -lcublas -lcuda  -lstdc++ -lm -lhdf5  -lhdf5 -lhdf5_hl -lconfig
PATHS = -L/opt/cuda/lib64/ -L/usr/lib64 -L/usr/lib
INCLUDES = -I/opt/cuda/include
DEBUG = -g
NX:= $(shell python stripsizes.py NX)
NY:= $(shell python stripsizes.py NY)
NZ:= $(shell python stripsizes.py NZ)
SIZE = -DNX=${NX} -DNY=${NY} -DNZ=${NZ}
GPU_SOURCES = $(wildcard src/*.cu)
CPU_SOURCES = $(wildcard src/*.c)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)
CPU_OBJECTS = $(CPU_SOURCES:.c=.o)


all: $(GPU_OBJECTS) $(CPU_OBJECTS)
	$(LD) -o channelMPI.bin $(CPU_OBJECTS) $(GPU_OBJECTS) $(PATHS) $(LIBS)

$(CPU_OBJECTS): src/%.o: src/%.c
	$(CC) -c $(INCLUDES) $(SIZE) $(PATHS)  $< -o $@

$(GPU_OBJECTS): src/%.o: src/%.cu
	$(NVCC) -c $(INCLUDES) $(SIZE) $(PATHS)  $< -o $@

clean:
	rm src/*.o channelMPI.bin
