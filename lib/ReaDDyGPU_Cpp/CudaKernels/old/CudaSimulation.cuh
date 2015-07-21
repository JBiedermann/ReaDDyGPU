
# include <cuda.h>
# include <curand.h>
# include <cuda_runtime.h>
# include <curand_kernel.h>
# include <sm_11_atomic_functions.h>

using namespace std;

class Simulation;

class CudaOrderOnePotential{
public:
    int type;
    int subtype;
    double forceConst;
    double origin[3];
    double normal[3];
    double extension[3];
    double height;
    double radius;
};

class CudaSimulation{
public:
    CudaSimulation(Simulation*);
    int simulate();
    int copyPosFromDevice();
    int copyPosToDevice();
    int initialize();
    int createNeighborList();
    int testNeighborList();

    int cudaDevice;
    int numberOfThreads;
    size_t gridSize;
    size_t blockSize;

    Simulation * simulation;

    int numberOfLatticeFields;
    double maxCutoff;
    double * boxSize;
    int * latticeSize;
    int * hostNeighborList;
    int * hostNeighborListBegins;
    int * hostOrderOnePotentialsMatrix;
    CudaOrderOnePotential * cudaCudaOrderOnePotentials;
    double * hostCollisionRadiiMatrix;

    double * cudaCoords;
    double * cudaForces;
    double * cudaD;
    double * cudaBoxSize;
    int * cudaLatticeSize;
    int * cudaTypes;
    int * cudaNeighborList;
    int * cudaNeighborListBegins;
    int * cudaSemaphore;
    int * cudaOrderOnePotentialsMatrix;
    CudaOrderOnePotential * hostCudaOrderOnePotentials;
    double * cudaCollisionRadiiMatrix;
    curandState * globalRandStates;
};
