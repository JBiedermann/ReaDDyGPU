
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
    double forceConst;
    double origin[3];
    double normal[3];
    double extension[3];
    double height;
    double radius;
};

class CudaOrderTwoPotential{
public:
    int type;
    int subtype;
    double forceConst;
    double length;
    double depth;
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
    double maxCutoff;
    double * boxSize;
    int * latticeSize;
    int * hostNeighborList;
    int * hostNeighborListBegins;
    int * hostOrderOnePotentialsMatrix;
    CudaOrderOnePotential * hostCudaOrderOnePotentials;
    int * hostOrderTwoPotentialsMatrix;
    CudaOrderTwoPotential * hostCudaOrderTwoPotentials;
    double * hostCollisionRadiiMatrix;
    double * hostParticleRadiiMatrix;
    CudaOrderTwoPotential * hostCudaGroupPotentials;
    GroupPart * hostIndividualGroups;
    int * hostRDFMatrix;

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
    CudaOrderOnePotential * cudaCudaOrderOnePotentials;
    int * cudaOrderTwoPotentialsMatrix;
    CudaOrderTwoPotential * cudaCudaOrderTwoPotentials;
    double * cudaCollisionRadiiMatrix;
    double * cudaParticleRadiiMatrix;
    CudaOrderTwoPotential * cudaCudaGroupPotentials;
    GroupPart * cudaIndividualGroups;
    int* cudaRDFMatrix;
    curandState * globalRandStates;
};
