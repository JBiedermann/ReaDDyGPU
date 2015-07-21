# ----------------------------------------------------------------------
# Makefile for OpenMM Preview Release 4 workshop "hello world" examples.
# August 18, 2009
# See https://simtk.org/home/openmm.
# ----------------------------------------------------------------------
# This assumes you have gcc compilers for whatever language you are
# using: g++ for C++ and C, gfortran for Fortran 95.
#
# For the C and Fortran examples, we're depending on your version of
# OpenMM to have been built with the automatically-generated API
# wrappers.
#
# This has had only minimal testing, although it has been known to
# work. It is likely to work fine for C and C++. For Fortran, you
# may need to add some of the C/C++ libraries:
#    -lc -lm -lstdc++ (or -lstdc++.6) -lgcc -lgcc_s
# but this wasn't required for these examples on Centos 5.2 using
# gcc 4.1.2.

# Check whether this is the right capitalization for your install directory.
OpenMM_INSTALL_DIR=$(OPENMM_PLUGIN_DIR)/../..
CFLAGS = -g
FFLAGS = -g -ffree-line-length-none

# Extra libraries required when gfortran links with a C++ module.
# If this doesn't work, look in /usr/lib to
# see what versions of libstdc++.so you have and try different
# versions.
FCPPLIBS = -lstdc++

LIB_DIR=$(OpenMM_INSTALL_DIR)/lib
INCLUDE_DIR=$(OpenMM_INSTALL_DIR)/include
LIBS= -lOpenMM

ALL_CPP_EXAMPLES = ReaDDyMM

ALL_PROGS = $(ALL_CPP_EXAMPLES)

default: ReaDDyMM

all : $(ALL_PROGS)

# Treat all .cpp source files the same way.
.cpp :
	g++ $(CFLAGS) -I$(INCLUDE_DIR) $< -L$(LIB_DIR) $(LIBS) -o $*

ReaDDyMM: ReaDDyMM.cpp simulation.h
	g++ -O3 -shared -m64 -g -Wl,--no-undefined  -fPIC -I$(INCLUDE_DIR) -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I ./ ReaDDyMM.cpp -L$(JAVA_HOME)/jre/lib/amd64/server -ljvm -L$(LIB_DIR) $(LIBS) -o libCReaDDyMM.so 

ReaDDyGPU: ReaDDyGPU.cpp simulationGPU.h
	#g++ -O3 -shared -m64 -g -Wl,--no-undefined -fPIC -I /ReaDDyGPU_Cpp -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I ./ -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"/ReaDDyGPU_Cpp/CudaKernels"  ReaDDyGPU.cpp -L$(JAVA_HOME)/jre/lib/amd64/server -ljvm -L /ReaDDyGPU_Cpp -libReaDDyGPU -L"./" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -o libCReaDDyGPU.so
	set CUDA_CACHE_MAXSIZE=512
	nvcc -shared -O3 -arch=compute_13 --compiler-options "-fPIC -m64 -g" -code=compute_13,sm_13,sm_20 -I"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/include" -I"./" -I"./ReaDDyGPU_Cpp" -I"./ReaDDyGPU_Cpp/CudaKernels" -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux ReaDDyGPU.cpp ReaDDyGPU_Cpp/Simulation.cpp ReaDDyGPU_Cpp/XMLHandler/ParticleParametersHandler.cpp ReaDDyGPU_Cpp/XMLHandler/GroupParametersHandler.cpp ReaDDyGPU_Cpp/XMLHandler/GlobalParametersHandler.cpp ReaDDyGPU_Cpp/XMLHandler/PotentialParametersHandler.cpp ReaDDyGPU_Cpp/XMLHandler/XMLReader.cpp ReaDDyGPU_Cpp/XMLHandler/CoordReader.cpp ReaDDyGPU_Cpp/XMLHandler/GroupReader.cpp ReaDDyGPU_Cpp/AuxillaryFunctions.cpp ReaDDyGPU_Cpp/outputHandler/outputHandler.cpp ReaDDyGPU_Cpp/CudaKernels/CudaSimulation.cu  -L"./" -L"./ReaDDyGPU_Cpp" -L"/home/jb/programs/xerces/xerces-c-3.1.1-x86_64-linux-gcc-3.4/lib" -lxerces-c -L$(JAVA_HOME)/jre/lib/amd64/server -ljvm -o libCReaDDyGPU.so

clean:
	rm $(ALL_PROGS) *.o *.mod *.obj *.exe 
	rm simulation.h
	rm libCReaDDyMM.so
	rm simulation.class
	rm libCReaDDyMM.d

