
# include <cuda.h>
# include <curand.h>
# include <cuda_runtime.h>
# include <curand_kernel.h>
# include <sm_11_atomic_functions.h>

using namespace std;

class Simulation;

struct GroupPart{
    int particle1;
    int particle2;
    int groupPot;
};

class CudaOrderOnePotential{
public:
    int type;
    int subtype;
    float forceConst;
    float origin[3];
    float normal[3];
    float extension[3];
    float height;
    float radius;
};

class CudaOrderTwoPotential{
public:
    int type;
    int subtype;
    float forceConst;
    float length;
    float depth;
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
    int copyRDFMatrix();
    int normalizeRDFFRame();
    int copyRDFMatrixToSimulation();
    int callRDFCalculation();


    int cudaDevice;
    int numberOfThreads;
    size_t gridSize;
    size_t blockSize;

    Simulation * simulation;

    int numberOfIndividualGroups;

    int numberOfLatticeFields;
    float maxCutoff;
    float * boxSize;
    int * latticeSize;
    int * hostNeighborList;
    int * hostNeighborListBegins;
    int * hostOrderOnePotentialsMatrix;
    CudaOrderOnePotential * hostCudaOrderOnePotentials;
    int * hostOrderTwoPotentialsMatrix;
    CudaOrderTwoPotential * hostCudaOrderTwoPotentials;
    float * hostCollisionRadiiMatrix;
    float * hostParticleRadiiMatrix;
    CudaOrderTwoPotential * hostCudaGroupPotentials;
    GroupPart * hostIndividualGroups;
    int * hostRDFMatrix;

    bool atomicAble;

    float * cudaCoords;
    float * cudaForces;
    float * cudaD;
    float * cudaBoxSize;
    int * cudaLatticeSize;
    int * cudaTypes;
    int * cudaNeighborList;
    int * cudaNeighborListBegins;
    int * cudaSemaphore;
    int * cudaOrderOnePotentialsMatrix;
    CudaOrderOnePotential * cudaCudaOrderOnePotentials;
    int * cudaOrderTwoPotentialsMatrix;
    CudaOrderTwoPotential * cudaCudaOrderTwoPotentials;
    float * cudaCollisionRadiiMatrix;
    float * cudaParticleRadiiMatrix;
    CudaOrderTwoPotential * cudaCudaGroupPotentials;
    GroupPart * cudaIndividualGroups;
    int* cudaRDFMatrix;
    curandState * globalRandStates;
};
