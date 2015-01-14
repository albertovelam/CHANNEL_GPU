CC = mpic++  -O3 -std=c99   
NVCC = /gpfs/apps/NVIDIA/CUDA/5.0/bin/nvcc -O3  
LD =mpic++   -O3  -std=c99
LIBS = -lcusparse -lcudart -lcufft -lcublas -lcuda  -lm  -lmpi -lhdf5 -lhdf5_hl -lszip -lconfig
PATHS = -L/gpfs/apps/NVIDIA/CUDA/5.0/lib64 -L/gpfs/apps/NVIDIA/HDF5/1.8.8/lib -L/gpfs/apps/NVIDIA/SZIP/2.1/lib -L/gpfs/home/upm79/upm79055/libs/lib -L/gpfs/apps/NVIDIA/VGL/2.3/lib 
INCLUDES = -I/opt/mpi/bullxmpi/1.1.11.1/include   -I/gpfs/apps/NVIDIA/CUDA/5.0/include  -I/gpfs/apps/NVIDIA/HDF5/1.8.8/include -I/gpfs/home/upm79/upm79055/libs/include
DEBUG = -g
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
