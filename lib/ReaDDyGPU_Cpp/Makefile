
default: ReaDDy

all : $(ALL_PROGS)

# Treat all .cpp source files the same way.
.cpp :
	g++ $(CFLAGS) -I$(INCLUDE_DIR) $< -L$(LIB_DIR) $(LIBS) -o $*

ReaDDy: ReaDDyGPU.cpp

	set CUDA_CACHE_MAXSIZE=512
	nvcc -shared --compiler-options "-fPIC -m64 -g" -O3 -arch=compute_13 -code=compute_13,sm_13,sm_20 -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"./CudaKernels" -L"./" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -o ReaDDyGPU.so Simulation.cpp XMLHandler/ParticleParametersHandler.cpp XMLHandler/GroupParametersHandler.cpp XMLHandler/GlobalParametersHandler.cpp XMLHandler/PotentialParametersHandler.cpp XMLHandler/XMLReader.cpp XMLHandler/CoordReader.cpp XMLHandler/GroupReader.cpp AuxillaryFunctions.cpp outputHandler/outputHandler.cpp CudaKernels/CudaSimulation.cu
 

alone: 

	set CUDA_CACHE_MAXSIZE=512
	nvcc -O3 -arch=compute_13 -code=compute_13,sm_13,sm_20 -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"./CudaKernels" -L"./" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -o ReaDDyGPU ReaDDyGPU.cpp Simulation.cpp XMLHandler/ParticleParametersHandler.cpp XMLHandler/GroupParametersHandler.cpp XMLHandler/GlobalParametersHandler.cpp XMLHandler/PotentialParametersHandler.cpp XMLHandler/XMLReader.cpp XMLHandler/CoordReader.cpp XMLHandler/GroupReader.cpp AuxillaryFunctions.cpp outputHandler/outputHandler.cpp CudaKernels/CudaSimulation.cu
 

debug:

	set CUDA_CACHE_MAXSIZE=512
	nvcc -O3 -arch=compute_13 -code=compute_13,sm_13,sm_20  -g -G -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"./CudaKernels" -L"./" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -o ReaDDyGPU ReaDDyGPU.cpp Simulation.cpp XMLHandler/ParticleParametersHandler.cpp XMLHandler/GroupParametersHandler.cpp XMLHandler/GlobalParametersHandler.cpp XMLHandler/PotentialParametersHandler.cpp XMLHandler/XMLReader.cpp XMLHandler/CoordReader.cpp XMLHandler/GroupReader.cpp AuxillaryFunctions.cpp outputHandler/outputHandler.cpp CudaKernels/CudaSimulation.cu

print:

	set CUDA_CACHE_MAXSIZE=512
	nvcc -O3 -arch=sm_21 -g -G -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"./CudaKernels" -L"./" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -o ReaDDyGPU ReaDDyGPU.cpp Simulation.cpp XMLHandler/ParticleParametersHandler.cpp XMLHandler/GroupParametersHandler.cpp XMLHandler/GlobalParametersHandler.cpp XMLHandler/PotentialParametersHandler.cpp XMLHandler/XMLReader.cpp XMLHandler/CoordReader.cpp XMLHandler/GroupReader.cpp AuxillaryFunctions.cpp outputHandler/outputHandler.cpp CudaKernels/CudaSimulation.cu



clean:


