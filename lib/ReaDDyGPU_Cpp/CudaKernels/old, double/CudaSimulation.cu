
# include <ReaDDyGPU.hpp>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
# include <vector>
# include <cuda.h>
# include <curand.h>
# include <cuda_runtime.h>
# include <curand_kernel.h>
# include <sm_11_atomic_functions.h>
#include <initializer_list>

/// //////////////////////////////////////////////////////////////////////////////////////////
/// /
/// TODO:
///         - data structures
///         - dynamic arrays (amortized linear runtime) +sort?
///         - periodic boundaries
///         - lattice + neighbor lists
///         - usage of shared mem
/// /
/// //////////////////////////////////////////////////////////////////////////////////////////


__global__ void update(double* cudaCoords, double* cudaForces, int* cudaTypes, double* cudaD, int * cudaNeighborList, int * cudaNeighborListBegins, double * cudaBoxSize, int * cudaSemaphore, curandState* globalRandStates, double dt, int numberParticles, double KB, double T, double maxCutoff, int * latticeSize);
__global__ void orderOne(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderOnePotentialsMatrix, CudaOrderOnePotential * cudaCudaOrderOnePotentials, int numberOfOrderOnePotentials, int numberOfParticleTypes, double * cudaParticleRadiiMatrix);
__global__ void orderTwo(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderTwoPotentialsMatrix, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);
__global__ void groups(double* cudaCoords, double* cudaForces, int* cudaTypes, int numberParticles, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix, CudaOrderTwoPotential * cudaCudaGroupPotentials, GroupPart * cudaIndividualGroups, int numberOfIndividualGroups, double * cudaBoxSize);
__device__ void getNeighbors(int particleNumber, int * todo, double* cudaCoords, int * cudaLatticeSize, double * cudaBoxSize, int maxCutoff);
__device__ void calculateOrderTwoPotential(int particleNumber, int interactingParticle, int orderTwoPotentialNr, double r, double* cudaCoords, double* cudaForces, int* cudaTypes, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);
__global__ void calculateRDF(int * cudaRDFMatrix, double* cudaCoords, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, int numberParticles, int maxCutoff, int numberOfParticleTypes, int numberOfBins);
__global__ void setup_kernel ( curandState * state, unsigned long seed, int n );

__global__ void warpOrderTwo(int warpsize, double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderTwoPotentialsMatrix, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);
__device__ void calculateOrderTwoPotential(double * particleCoord, double * particleForce, int particleType, double * interactingParticleCoord, double * interactingParticleForce, int interactingParticleType, int orderTwoPotentialNr, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, double * cudaBoxSize, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);

CudaSimulation::CudaSimulation(Simulation* simulation){

    this->simulation = simulation;

}

/*__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old); return __longlong_as_double(old);
}*/
__device__ double atomicAdd(double* address, double val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, __float_as_int(val + __int_as_float(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old); return __int_as_float(old);
}
///Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz3GDRkYOfX


CudaOrderOnePotential toCudaOrderOnePotential(OrderOnePotential* orderOnePotential){
    CudaOrderOnePotential cudaOrderOnePotential = CudaOrderOnePotential();
    if(orderOnePotential->type.compare("DISK")==0){
        DiskPotential * diskPotential = reinterpret_cast<DiskPotential*>(orderOnePotential);
        cudaOrderOnePotential.type=diskPotential->typeID;
        cudaOrderOnePotential.subtype=diskPotential->subtypeID;
        cudaOrderOnePotential.forceConst=diskPotential->forceConst;
        std::copy ( diskPotential->center, diskPotential->center+3, cudaOrderOnePotential.origin );
        std::copy ( diskPotential->normal, diskPotential->normal+3, cudaOrderOnePotential.normal );
        cudaOrderOnePotential.radius=diskPotential->radius;
    }
    else if(orderOnePotential->type.compare("CYLINDER")==0){
        CylinderPotential * cylinderPotential = reinterpret_cast<CylinderPotential*>(orderOnePotential);
        cudaOrderOnePotential.type=cylinderPotential->typeID;
        cudaOrderOnePotential.subtype=cylinderPotential->subtypeID;
        cudaOrderOnePotential.forceConst=cylinderPotential->forceConst;
        std::copy ( cylinderPotential->center, cylinderPotential->center+3, cudaOrderOnePotential.origin );
        std::copy ( cylinderPotential->normal, cylinderPotential->normal+3, cudaOrderOnePotential.normal );
        cudaOrderOnePotential.radius=cylinderPotential->radius;
        cudaOrderOnePotential.height=cylinderPotential->height;
    }
    else if(orderOnePotential->type.compare("SPHERE")==0){
        SpherePotential * spherePotential = reinterpret_cast<SpherePotential*>(orderOnePotential);
        cudaOrderOnePotential.type=spherePotential->typeID;
        cudaOrderOnePotential.subtype=spherePotential->subtypeID;
        cudaOrderOnePotential.forceConst=spherePotential->forceConst;
        std::copy ( spherePotential->center, spherePotential->center+3, cudaOrderOnePotential.origin );
        cudaOrderOnePotential.radius=spherePotential->radius;
    }
    else if(orderOnePotential->type.compare("BOX")==0 || orderOnePotential->type.compare("CUBE")==0){
        BoxPotential * boxPotential = reinterpret_cast<BoxPotential*>(orderOnePotential);
        cudaOrderOnePotential.type=boxPotential->typeID;
        cudaOrderOnePotential.subtype=boxPotential->subtypeID;
        cudaOrderOnePotential.forceConst=boxPotential->forceConst;
        std::copy ( boxPotential->origin, boxPotential->origin+3, cudaOrderOnePotential.origin );
        std::copy ( boxPotential->extension, boxPotential->extension+3, cudaOrderOnePotential.extension );
    }
    return cudaOrderOnePotential;
}

CudaOrderTwoPotential toCudaOrderTwoPotential(OrderTwoPotential* orderTwoPotential){
    CudaOrderTwoPotential cudaOrderTwoPotential = CudaOrderTwoPotential();
    if(orderTwoPotential->type.compare("HARMONIC")==0){
        HarmonicPotential * harmonicPotential = reinterpret_cast<HarmonicPotential*>(orderTwoPotential);
        cudaOrderTwoPotential.type=harmonicPotential->typeID;
        cudaOrderTwoPotential.subtype=harmonicPotential->subtypeID;
        cudaOrderTwoPotential.forceConst = harmonicPotential->forceConst;
    }
    else if(orderTwoPotential->type.compare("WEAK_INTERACTION_HARMONIC")==0){
        WeakInteractionHarmonicPotential * weakInteractionHarmonicPotential = reinterpret_cast<WeakInteractionHarmonicPotential*>(orderTwoPotential);
        cudaOrderTwoPotential.type=weakInteractionHarmonicPotential->typeID;
        cudaOrderTwoPotential.subtype=weakInteractionHarmonicPotential->subtypeID;
        cudaOrderTwoPotential.forceConst=weakInteractionHarmonicPotential->forceConst;
        cudaOrderTwoPotential.length=weakInteractionHarmonicPotential->length;
        cudaOrderTwoPotential.depth=weakInteractionHarmonicPotential->depth;
    }
    return cudaOrderTwoPotential;
}

int CudaSimulation::initialize(){

    int numberOfCudaDevices = 0;
    cudaGetDeviceCount(&numberOfCudaDevices);

    if(numberOfCudaDevices==0){
        cout << "no cuda device availible" << endl;
        return 1;
    }
    if(simulation->testmode)
        cout << endl << endl << numberOfCudaDevices << " cuda devices found" << endl << endl;

    for(int i=0; i<numberOfCudaDevices; ++i){

        cudaSetDevice(i);
        struct cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, i);
        if(simulation->testmode){
            cout << prop.name << endl;
            cout << "compute capability: " << prop.major << "." << prop.minor << endl;

            cout << "total global Memory: " << (float)prop.totalGlobalMem/1024.0f/1024.0f/1024.0f << "GB" << endl;
            cout << "shared memory per block: " << (float)prop.sharedMemPerBlock/1024.0f << "KB" << endl;
            cout << "total constant memory: " << (float)prop.totalConstMem/1024.0f << "KB" << endl;
            cout << "memory clock rate: " << prop.memoryClockRate << "Hz" << endl;
            cout << "memory bus width: " << prop.memoryBusWidth << "bits" << endl;

            cout << "multi processors: " << prop.multiProcessorCount << endl;
            cout << "clock rate: " << prop.clockRate << "Hz" << endl;

            cout << "warpsize: " << prop.warpSize << endl;
            cout << "max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
            cout << "max threads dim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << endl;
            cout << "max grid size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;
            cout << endl;
        }
    }

    /// ////////////////////////////////////////////////////////////////////////
    cudaDevice = 3;
    numberOfThreads = 128;
    /// ////////////////////////////////////////////////////////////////////////

    cudaSetDevice(cudaDevice);

    gridSize = (simulation->numberParticles/numberOfThreads)+1;
    blockSize = numberOfThreads;
    //gridSize = 10;
    //blockSize = 10;

    if(simulation->testmode)
        cout << "use " << gridSize<< " blocks (grid size) and " << blockSize << " threads (block size) each" << endl;

    maxCutoff = simulation->maxCutoff;
    boxSize = simulation->latticeBounds;

if(simulation->testmode)
    cout << "initialize cuda random variables" << endl;

    /// initialize cuRand
    cudaMalloc ( (void**)&globalRandStates, simulation->numberParticles * sizeof( curandState ) );
    /// setup seeds
    setup_kernel <<< gridSize, blockSize >>> ( globalRandStates, time(NULL), simulation->numberParticles );


if(simulation->testmode)
    cout << "initialize several cuda arrays" << endl;

    /// Coords
    cudaMalloc((void**)&cudaCoords,( simulation->numberParticles * 3 * sizeof ( double ) ));
    copyPosToDevice();

    /// Forces
    cudaMalloc((void**)&cudaForces,( simulation->numberParticles * 3 * sizeof ( double ) ));
    cudaMemset( cudaForces,(double)0, ( simulation->numberParticles * 3 * sizeof ( double ) ));

    /// Diffusion const.
    double * hostDiffConst;
    hostDiffConst = new double[simulation->particleTypes.size()];
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        hostDiffConst[i]=simulation->particleTypes[i].D;
    }
    cudaMalloc((void**)&cudaD,( simulation->particleTypes.size() * sizeof ( double ) ));
    cudaMemcpy(cudaD, hostDiffConst, ( simulation->particleTypes.size() * sizeof ( double ) ), cudaMemcpyHostToDevice);

    /// types
    cudaMalloc((void**)&cudaTypes,( simulation->numberParticles * sizeof ( int ) ));
    cudaMemcpy(cudaTypes, simulation->types, ( simulation->numberParticles * sizeof ( int ) ), cudaMemcpyHostToDevice);

    if(createNeighborList()!=0){
        cout <<"neigborlist building problem" << endl;
        return 1;
    }

    cudaMemcpy(cudaNeighborList, hostNeighborList, ( simulation->numberParticles * 2 * sizeof ( int ) ), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaNeighborListBegins, hostNeighborListBegins, ( numberOfLatticeFields * sizeof ( int ) ), cudaMemcpyHostToDevice);

    cudaMalloc ((void**)&cudaBoxSize, ( 6 * sizeof ( double ) ));
    cudaMemcpy( cudaBoxSize, boxSize, ( 6 * sizeof ( double ) ), cudaMemcpyHostToDevice);
    cudaMalloc (   (void**)&cudaLatticeSize, ( 3 * sizeof ( int ) ));
    cudaMemcpy(cudaLatticeSize, latticeSize, ( 3 * sizeof ( int ) ), cudaMemcpyHostToDevice);

    /// cudaSemaphores for the lattice fields
    cudaMalloc((void**)&cudaSemaphore,( numberOfLatticeFields * sizeof ( int ) ));
    cudaMemset( cudaSemaphore,(int)0, ( numberOfLatticeFields * sizeof ( int ) ));

if(simulation->testmode)
    cout << "initialize cuda order one potentials" << endl;

    /// Matrix for order one potentials = matrix[pot][types] = matrix[simulation->orderOnePotentials.size()][simulation->particleTypes.size()]
    int orderOnePotentialsMatrixSize = simulation->particleTypes.size() * simulation->orderOnePotentials.size();
    hostOrderOnePotentialsMatrix = new int[orderOnePotentialsMatrixSize];
    for(int i=0; i<simulation->orderOnePotentials.size(); ++i){
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            hostOrderOnePotentialsMatrix[i*simulation->particleTypes.size()+j]=0;
        }
        for(int j=0; j<simulation->orderOnePotentials[i]->affectedParticleTypeIds.size(); ++j){
            hostOrderOnePotentialsMatrix[i*simulation->particleTypes.size()+simulation->orderOnePotentials[i]->affectedParticleTypeIds[j]]=1;
        }
    }
    cudaMalloc((void**)&cudaOrderOnePotentialsMatrix,( orderOnePotentialsMatrixSize * sizeof ( int ) ));
    cudaMemcpy(cudaOrderOnePotentialsMatrix, hostOrderOnePotentialsMatrix, ( orderOnePotentialsMatrixSize * sizeof ( int ) ), cudaMemcpyHostToDevice);

    /// create cuda order one pot
    hostCudaOrderOnePotentials = new CudaOrderOnePotential[simulation->orderOnePotentials.size()];
    for(int i=0; i<simulation->orderOnePotentials.size(); ++i){
        hostCudaOrderOnePotentials[i] = toCudaOrderOnePotential(simulation->orderOnePotentials[i]);
    }
    cudaMalloc((void**)&cudaCudaOrderOnePotentials,( simulation->orderOnePotentials.size() * sizeof ( CudaOrderOnePotential ) ));
    cudaMemcpy(cudaCudaOrderOnePotentials, hostCudaOrderOnePotentials, ( simulation->orderOnePotentials.size() * sizeof ( CudaOrderOnePotential ) ), cudaMemcpyHostToDevice);

if(simulation->testmode)
    cout << "initialize cuda order two potentials" << endl;

    /// Lookup for order two potentials
    int numberOfParticleTypes = simulation->particleTypes.size();
    int orderTwoPotentialsMatrixSize = numberOfParticleTypes * numberOfParticleTypes * simulation->orderTwoPotentials.size();
    hostOrderTwoPotentialsMatrix = new int[orderTwoPotentialsMatrixSize];
    for(int numberParticleTypes1=0; numberParticleTypes1<numberOfParticleTypes; ++numberParticleTypes1){
        //cout << numberParticleTypes1 << endl;
        for(int numberParticleTypes2=0; numberParticleTypes2<numberOfParticleTypes; ++numberParticleTypes2){
            //cout << " " << numberParticleTypes2 << endl;
            for(int numberOrderTwoPotentials=0; numberOrderTwoPotentials<simulation->orderTwoPotentials.size(); ++numberOrderTwoPotentials){
                //cout << "  " << numberOrderTwoPotentials << endl;
                hostOrderTwoPotentialsMatrix[numberParticleTypes1*numberOfParticleTypes+numberParticleTypes2*numberOfParticleTypes+numberOrderTwoPotentials]=0;
            }
        }
    }
    for(int orderTwoPotential=0; orderTwoPotential<simulation->orderTwoPotentials.size(); ++orderTwoPotential){
        for(int i=0; i<simulation->orderTwoPotentials[orderTwoPotential]->affectedParticleTypeIdPairs.size(); i+=2){
            //cout << " " << i << endl;
            int particleType1= simulation->orderTwoPotentials[orderTwoPotential]->affectedParticleTypeIdPairs[i];
            int particleType2= simulation->orderTwoPotentials[orderTwoPotential]->affectedParticleTypeIdPairs[i+1];
            hostOrderTwoPotentialsMatrix[particleType1*numberOfParticleTypes+particleType2*numberOfParticleTypes+orderTwoPotential]=1;
            hostOrderTwoPotentialsMatrix[particleType2*numberOfParticleTypes+particleType1*numberOfParticleTypes+orderTwoPotential]=1;
        }
    }

    cudaMalloc((void**)&cudaOrderTwoPotentialsMatrix,( orderTwoPotentialsMatrixSize * sizeof ( int ) ));
    cudaMemcpy(cudaOrderTwoPotentialsMatrix, hostOrderTwoPotentialsMatrix, ( orderTwoPotentialsMatrixSize * sizeof ( int ) ), cudaMemcpyHostToDevice);

    /// create cuda order two pot
    hostCudaOrderTwoPotentials = new CudaOrderTwoPotential[simulation->orderTwoPotentials.size()];
    for(int i=0; i<simulation->orderTwoPotentials.size(); ++i){
        hostCudaOrderTwoPotentials[i] = toCudaOrderTwoPotential(simulation->orderTwoPotentials[i]);
    }
    cudaMalloc((void**)&cudaCudaOrderTwoPotentials,( simulation->orderTwoPotentials.size() * sizeof ( CudaOrderTwoPotential ) ));
    cudaMemcpy(cudaCudaOrderTwoPotentials, hostCudaOrderTwoPotentials, ( simulation->orderTwoPotentials.size() * sizeof ( CudaOrderTwoPotential ) ), cudaMemcpyHostToDevice);

if(simulation->testmode)
    cout << "initialize cuda particle radii and collision radii matrix" << endl;

    /// create cuda paritcle radii matix (for order one ptoentials
    hostParticleRadiiMatrix = new double[simulation->particleTypes.size()];
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        hostParticleRadiiMatrix[i]=simulation->particleTypes[i].defaultRadius;
        //cout << hostParticleRadiiMatrix[i] << endl;
    }
    cudaMalloc((void**)&cudaParticleRadiiMatrix,( (simulation->particleTypes.size()) * sizeof ( double ) ));
    cudaMemcpy(cudaParticleRadiiMatrix, hostParticleRadiiMatrix, ( simulation->particleTypes.size() * sizeof ( double ) ), cudaMemcpyHostToDevice);

    /// create cuda collision radii matix for order two potentials
    hostCollisionRadiiMatrix = new double[(simulation->particleTypes.size())*simulation->particleTypes.size()];
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j] = simulation->particleTypes[i].radiiMatrix[j];
        }
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j] = hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j]== 0 ? simulation->particleTypes[i].defaultRadius : hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j];
        }
    }
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=i; j<simulation->particleTypes.size(); ++j){
            double x = hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j]+hostCollisionRadiiMatrix[j*simulation->particleTypes.size()+i];
            hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j] = x;
            hostCollisionRadiiMatrix[j*simulation->particleTypes.size()+i] = x;
        }
    }
    /*for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            cout << hostCollisionRadiiMatrix[i*simulation->particleTypes.size()+j] << " " ;
        }
        cout << endl;
    }*/
    cudaMalloc((void**)&cudaCollisionRadiiMatrix,( (simulation->particleTypes.size())*simulation->particleTypes.size() * sizeof ( double ) ));
    cudaMemcpy(cudaCollisionRadiiMatrix, hostCollisionRadiiMatrix, ( (simulation->particleTypes.size())*simulation->particleTypes.size() * sizeof ( double ) ), cudaMemcpyHostToDevice);

if(simulation->testmode)
    cout << "initialize cuda group potentials" << endl;

    /// create cuda group potentials
    int numberOfGroupPotentials=0;
    for(int i=0; i<simulation->groups.size(); ++i){
        numberOfGroupPotentials += simulation->groups[i].potentials.size();
    }
    hostCudaGroupPotentials = new CudaOrderTwoPotential[numberOfGroupPotentials];
    int continuousForceNumber=0;
    for(int i=0; i<simulation->groups.size(); ++i){
        for(int j=0; j<simulation->groups[i].potentials.size(); ++j, ++continuousForceNumber){
            hostCudaGroupPotentials[continuousForceNumber] = toCudaOrderTwoPotential(simulation->groups[i].potentials[j]);
//cout <<"group "  << i << " force "<< hostCudaGroupPotentials[continuousForceNumber].type << endl;
        }
    }
    cudaMalloc((void**)&cudaCudaGroupPotentials,( numberOfGroupPotentials * sizeof ( CudaOrderTwoPotential ) ));
    cudaMemcpy(cudaCudaGroupPotentials, hostCudaGroupPotentials, ( numberOfGroupPotentials * sizeof ( CudaOrderTwoPotential ) ), cudaMemcpyHostToDevice);

    /// list of particle pairs with potential calculations from a group
    vector<GroupPart> groupParts = vector<GroupPart>();
    continuousForceNumber=0;
    for(int i=0; i<simulation->groups.size(); ++i){
        for(int j=0; j<simulation->groups[i].individualGroups.size(); ++j){
            //for(int k=0; k<simulation->groups[i].individualGroups[j].size(); ++k){
            /// TODO: later for more than two!!
                GroupPart groupPart;
                groupPart.particle1=simulation->groups[i].individualGroups[j][0];
                groupPart.particle2=simulation->groups[i].individualGroups[j][1];
                groupPart.groupPot=continuousForceNumber;
                groupParts.push_back(groupPart);
//cout << simulation->groups[i].individualGroups[j][0] << " " <<simulation->groups[i].individualGroups[j][1] <<" " << continuousForceNumber<< endl;
            //}
            //++continuousForceNumber;
        }
    }
    hostIndividualGroups = &groupParts[0];
    numberOfIndividualGroups = groupParts.size();
    /*for(int i=0; i<numberOfIndividualGroups; ++i){
        cout << hostIndividualGroups[i].particle1 << " " << hostIndividualGroups[i].particle2 << " " << hostIndividualGroups[i].groupPot << " " << hostCudaGroupPotentials[hostIndividualGroups[i].groupPot].forceConst << " " << hostCudaGroupPotentials[hostIndividualGroups[i].groupPot].type << " " << hostCudaGroupPotentials[hostIndividualGroups[i].groupPot].subtype << endl;
    }*/
    cudaMalloc((void**)&cudaIndividualGroups,( groupParts.size() * sizeof ( GroupPart ) ));
    cudaMemcpy(cudaIndividualGroups, hostIndividualGroups, ( groupParts.size() * sizeof ( GroupPart ) ), cudaMemcpyHostToDevice);

    /// Matrix for RDF calculation
    /// carefull about reactions! !!!!!!!!!!!!!!!!!!!!!!!!!!
    if(simulation->RDFrequired>0){
        hostRDFMatrix = new int[simulation->particleTypes.size()*simulation->particleTypes.size()*simulation->numberOfRDFBins];
        cudaMalloc( (void**)&cudaRDFMatrix,  ( simulation->particleTypes.size() * simulation->particleTypes.size() * simulation->numberOfRDFBins *  sizeof ( int ) ));
        cudaMemset( cudaRDFMatrix,(int)0, ( simulation->particleTypes.size() * simulation->particleTypes.size() * simulation->numberOfRDFBins * sizeof ( int ) ));
    }

    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ){
        printf( "cuda error during initialization: %s\n",cudaGetErrorString(error) );
        return 1;
    }

if(simulation->testmode)
    cout << "initialization done" << endl;

    return 0;
}

int CudaSimulation::copyRDFMatrix(){

    /// copy from GPU
    cudaMemcpy(hostRDFMatrix, cudaRDFMatrix, simulation->particleTypes.size() * simulation->particleTypes.size() * simulation->numberOfRDFBins * sizeof ( int ), cudaMemcpyDeviceToHost);

    if(simulation->testmode){
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            printf( "cuda error: %s\n",cudaGetErrorString(error) );
            return 1;
        }
    }

    /// copy to simulation and normalize
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j <simulation->particleTypes.size(); ++j){
            for(int k=0; k<simulation->numberOfRDFBins; ++k){
                simulation->RDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k] += (double)hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k]/(double)simulation->numberOfParticlesPerType[i]/(double)simulation->numberOfParticlesPerType[j];
            }
        }
    }

    return 0;
}
int CudaSimulation::copyRDFMatrixToSimulation(){
    return 0;
}

int CudaSimulation::normalizeRDFFRame(){
    /*for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j <simulation->particleTypes.size(); ++j){
            for(int k=0; k<simulation->numberOfRDFBins; ++k){
                /// normalize over particle numbers
                cout << i << "x" << j << "(" << k << "): "<< hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k] << " / " <<  simulation->numberOfParticlesPerType[j] << " / " <<  simulation->numberOfParticlesPerType[i] << endl;
                cout << "->" << (double)hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k]/(double)simulation->numberOfParticlesPerType[i]/(double)simulation->numberOfParticlesPerType[j] << endl;
                //hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k] = hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k]/simulation->numberOfParticlesPerType[i]/simulation->numberOfParticlesPerType[j];
                /// normalize 2D
                //hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k] = hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k];
                /// normalize 3D
                //hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k] = hostRDFMatrix[i*simulation->particleTypes.size()*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+k];
            }
        }
    }*/
    return 0;
}

int CudaSimulation::callRDFCalculation(){

    //cout << "RDF" << endl;
    calculateRDF<<<gridSize,blockSize>>>(cudaRDFMatrix, cudaCoords, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, simulation->numberParticles, maxCutoff, simulation->particleTypes.size(), simulation->numberOfRDFBins);

    if(simulation->testmode){
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            printf( "RDF, cuda error: %s\n",cudaGetErrorString(error ));
            return 1;
        }
    }
    ++simulation->numberOfRDFFrames;
    return 0;
}

int CudaSimulation::createNeighborList(){

    numberOfLatticeFields = (boxSize[1]-boxSize[0])/maxCutoff*(boxSize[3]-boxSize[2])/maxCutoff*(boxSize[5]-boxSize[4])/maxCutoff;
    latticeSize = new int[3];
    latticeSize[0] = (boxSize[1]-boxSize[0])/maxCutoff;
    latticeSize[1] = (boxSize[3]-boxSize[2])/maxCutoff;
    latticeSize[2] = (boxSize[5]-boxSize[4])/maxCutoff;
    cudaMalloc((void**)&cudaNeighborList,( simulation->numberParticles * 2 * sizeof ( int ) ));
    cudaMalloc((void**)&cudaNeighborListBegins,( numberOfLatticeFields * sizeof ( int ) ));

    hostNeighborList = new int[simulation->numberParticles * 2];
    hostNeighborListBegins= new int[numberOfLatticeFields];

    for(int i=0; i<numberOfLatticeFields; ++i){
        hostNeighborListBegins[i]=-1;
    }
    /*if(simulation->testmode){
        cout << "lattice informations:  " << endl;
        cout << "simulation size x[nm]: " << boxSize[1]-boxSize[0] << endl;
        cout << "simulation size y[nm]: " << boxSize[3]-boxSize[2] << endl;
        cout << "simulation size z[nm]: " << boxSize[5]-boxSize[4] << endl;
        cout << "number of voxels:      " << numberOfLatticeFields << endl;
        cout << "voxel edge length:     " << maxCutoff << endl;
        cout << "lattice size x:        " << latticeSize[0] << endl;
        cout << "lattice size y:        " << latticeSize[1] << endl;
        cout << "lattice size z:        " << latticeSize[2] << endl << endl;
    }*/

    for(int i=0; i<simulation->numberParticles; ++i){

        int field=((int)floor((simulation->coords[3*i+2]-boxSize[4])/maxCutoff)%latticeSize[2])*latticeSize[0]*latticeSize[1]
                 +((int)floor((simulation->coords[3*i+1]-boxSize[2])/maxCutoff)%latticeSize[1])*latticeSize[0]
                 +((int)floor((simulation->coords[3*i+0]-boxSize[0])/maxCutoff)%latticeSize[0]);

        if(field<0 || field>numberOfLatticeFields){
            cout << "particle is out of the Box: " << i << " [" <<simulation->coords[3*i+0] << ", " << simulation->coords[3*i+1] << ", " << simulation->coords[3*i+2] << "]" << endl;
            return 1;
        }

        if(hostNeighborListBegins[field]==-1){
            /// this particle is the first in this field. it is its own predecessor and successor
            hostNeighborListBegins[field]=i;
            hostNeighborList[2*i+1]=i;
            hostNeighborList[2*i]=i;
        }
        else{
            /// x f y -> x p f y
            /// particles successor is the fields first particle
            /// S'(p) = f
            hostNeighborList[2*i+1]=hostNeighborListBegins[field];
            /// sucessor of the first particles predecessor is the particle
            /// S(P(f))=p , P(f)=x -> S'(x)=p
            hostNeighborList[2*hostNeighborList[2*hostNeighborListBegins[field]]+1]=i;
            /// particles predecessor is the predecessor of the fields first particle
            /// P'(p)=P(f)=x
            hostNeighborList[2*i]=hostNeighborList[2*hostNeighborListBegins[field]];
            /// fields first particles new predecessor is the current particle
            /// P'(f)=p
            hostNeighborList[2*hostNeighborListBegins[field]]=i;
            //hostNeighborListBegins[field]=i;
        }
    }

    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ){
        printf( "cuda error: %s\n",cudaGetErrorString(error) );
        return 1;
    }

    return 0;
}

int CudaSimulation::testNeighborList(){

    cudaMemcpy(hostNeighborList, cudaNeighborList, ( simulation->numberParticles * 2 * sizeof ( int ) ), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNeighborListBegins, cudaNeighborListBegins, ( numberOfLatticeFields * sizeof ( int ) ), cudaMemcpyDeviceToHost);

    int count = 0;
    int count2 = 0;
    int x;
    for(int i=0; i<numberOfLatticeFields; ++i){
        x=hostNeighborListBegins[i];
        //cout << i << ":" << x << endl;
        if(x!=-1){
            do{
                count++;
                //cout << hostNeighborList[2*x+0] << " " << x << " " << hostNeighborList[2*x+1] << endl;
                x=hostNeighborList[2*x+1];
                if(x==hostNeighborListBegins[i])
                    break;
                //char a;
                //cin >> a;
            }while(true);
        }
        else{
            ++count2;
        }
    }
    cout << "Neighborlist check:" <<  "count: "<< count << "    part num: " << simulation->numberParticles << "     (check 2:" << count2 << " empty fields)"<<  endl;
    if(count!=simulation->numberParticles){
        cout << "Neighborlist broken!" << endl;
        return 1;
    }

    //cout << "Neighborlist okay!" << endl;

    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ){
        printf( "cuda error: %s\n",cudaGetErrorString(error) );
        return 1;
    }

    return 0;
}

int CudaSimulation::copyPosToDevice(){

    cudaMemcpy(cudaCoords, simulation->coords, simulation->numberParticles * 3 * sizeof(double), cudaMemcpyHostToDevice);

    if(simulation->testmode){
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            printf( "cuda error: %s\n",cudaGetErrorString(error) );
            return 1;
        }
    }
    return 0;
}

int CudaSimulation::copyPosFromDevice(){

    cudaMemcpy(simulation->coords, cudaCoords, simulation->numberParticles * 3 * sizeof ( double ), cudaMemcpyDeviceToHost);

    if(simulation->testmode){
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            printf( "cuda error: %s\n",cudaGetErrorString(error) );
            return 1;
        }
    }
    return 0;
}


int CudaSimulation::simulate(){


    //cout << "order one" << endl;
    //orderOne<<<1,1>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderOnePotentialsMatrix, cudaCudaOrderOnePotentials, simulation->orderOnePotentials.size(), simulation->particleTypes.size(), cudaCollisionRadiiMatrix);
    orderOne<<<gridSize,blockSize>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderOnePotentialsMatrix, cudaCudaOrderOnePotentials, simulation->orderOnePotentials.size(), simulation->particleTypes.size(), cudaParticleRadiiMatrix);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "order one, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }
    //cout << "order two" << endl;
    /*orderTwo<<<gridSize,blockSize>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderTwoPotentialsMatrix, cudaCudaOrderTwoPotentials, simulation->orderTwoPotentials.size(), simulation->particleTypes.size(), cudaCollisionRadiiMatrix);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "order two, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }*/
    //cout << "order two" << endl;
    int warpsize=32;
    warpOrderTwo<<<numberOfLatticeFields,warpsize, (warpsize*3*4*sizeof(double)+(27+warpsize*3+5)*sizeof(int))>>>(warpsize, cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderTwoPotentialsMatrix, cudaCudaOrderTwoPotentials, simulation->orderTwoPotentials.size(), simulation->particleTypes.size(), cudaCollisionRadiiMatrix);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "warp order two, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }
    //cout << "groups" << endl;
    /// TODO!: use different grid and block size!
        groups<<<gridSize,blockSize>>>(cudaCoords, cudaForces, cudaTypes, simulation->numberParticles, simulation->particleTypes.size(), cudaCollisionRadiiMatrix, cudaCudaGroupPotentials, cudaIndividualGroups, numberOfIndividualGroups, cudaBoxSize);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "groups, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }

    //cout << "update" << endl;
    update<<<gridSize,blockSize>>>(  cudaCoords, cudaForces, cudaTypes, cudaD, cudaNeighborList, cudaNeighborListBegins, cudaBoxSize, cudaSemaphore, globalRandStates, simulation->stepSizeInPs, simulation->numberParticles, simulation->boltzmann,  simulation->temperature, maxCutoff, cudaLatticeSize);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "update, cuda error: %s\n",cudaGetErrorString(error ));
                return 1;
            }
    }

    return 0;
}


/// /////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cuda kernels ////////////////////////////////////////////////////////////////////////////////////////////////
/// /////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void orderOne(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderOnePotentialsMatrix, CudaOrderOnePotential * cudaCudaOrderOnePotentials, int numberOfOrderOnePotentials, int numberOfParticleTypes, double * cudaParticleRadiiMatrix){


    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

    ////////////////////////////////////////////////////
    bool considerParticleRadius = true;
    //////////////////////////////////


    if(particleNumber<numberParticles){
        curandState localState = globalRandStates[particleNumber];

        /// do calculation of forces and maybe reactions here ...

        /// go through all order one potetntials
        for(int orderOnePotential=0; orderOnePotential<numberOfOrderOnePotentials; ++orderOnePotential){
            ///  lookup in matrix whether they apply to the current particle type
            if(cudaOrderOnePotentialsMatrix[orderOnePotential*numberOfParticleTypes+cudaTypes[particleNumber]]==1){
                /// check what kind of potential it is
                if(cudaCudaOrderOnePotentials[orderOnePotential].type==1){/// Disk

                    /// calculation depends on the normal vector. assign x,y and z coordinates to variables
                    int normal, side1, side2;
                    /// normal vector on x axis -> assign x to normal and y and z to the lateral (on Disk) directions
                    if(cudaCudaOrderOnePotentials[orderOnePotential].normal[0]==1){
                        normal=0;side1=1;side2=2;
                    }
                    /// y
                    else if(cudaCudaOrderOnePotentials[orderOnePotential].normal[1]==1){
                        normal=1;side1=0;side2=2;
                    }
                    /// x
                    else {
                        normal=2;side1=1;side2=0;
                    }

                    /// different subtypes
                    if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==1){/// attractive

                        double r = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-cudaCoords[3*particleNumber+normal];
                        cudaForces[3*particleNumber+normal]+=-cudaCudaOrderOnePotentials[orderOnePotential].forceConst*r;

                        /// particle radius!
                        r = sqrt(
                                    pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                    +
                                    pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                );
                        if (r > cudaCudaOrderOnePotentials[orderOnePotential].radius) {
                            cudaForces[3*particleNumber+side1]+=
                                    -cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                    *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                    /r
                                    *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]-cudaCoords[3*particleNumber+side1]);
                            cudaForces[3*particleNumber+side2]+=
                                    -cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                    *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                    /r
                                    *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]-cudaCoords[3*particleNumber+side2]);

                        }
                    }
                    else if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==2){/// repulsive
                        // makes no sense ...
                        /*
                        // force along normal vector
                        r = distToDiskPlane;// actual
                        r0 = pRadius;// desired
                        double r_1 = distToCenterWithinDiskPlane - pRadius;
                        double r0_1 = diskRadius;
                        if (r < r0 && r_1 < r0_1) {

                            precompute = (k * (-r0 + r) / r);

                                gradient[0] = gradient[0]+ precompute * ( pointOnDiskPlane[0]-coords1[0]);
                                gradient[1] = gradient[1]+ precompute * ( pointOnDiskPlane[1]-coords1[1]);
                                gradient[2] = gradient[2]+ precompute * ( pointOnDiskPlane[2]-coords1[2]);

                        }*/
                    }
                }/// end Disk
                else if(cudaCudaOrderOnePotentials[orderOnePotential].type==2){/// Cylinder

                        int normal, side1, side2;
                        if(cudaCudaOrderOnePotentials[orderOnePotential].normal[0]==1){normal=0;side1=1;side2=2;}
                        else if(cudaCudaOrderOnePotentials[orderOnePotential].normal[1]==1){normal=1;side1=0;side2=2;}
                        else {normal=2;side1=1;side2=0;}

                        if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==1){/// attractive

                            double r = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-cudaCoords[3*particleNumber+normal];
                            if(fabsf(r)>cudaCudaOrderOnePotentials[orderOnePotential].height*0.5)
                            cudaForces[3*particleNumber+normal]+=-cudaCudaOrderOnePotentials[orderOnePotential].forceConst*(fabsf(r)-cudaCudaOrderOnePotentials[orderOnePotential].height*0.5)/fabsf(r)*r;

                            /// particle radius!
                            r = sqrt(
                                        pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                        +
                                        pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                    );
                            if (r > cudaCudaOrderOnePotentials[orderOnePotential].radius) {
                                cudaForces[3*particleNumber+side1]+=
                                        -cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                        *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                        /r
                                        *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]-cudaCoords[3*particleNumber+side1]);
                                cudaForces[3*particleNumber+side2]+=
                                        -cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                        *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                        /r
                                        *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]-cudaCoords[3*particleNumber+side2]);

                            }
                        }
                        else if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==2){/// repulsive

                            double r = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-cudaCoords[3*particleNumber+normal];
                            if(fabsf(r)<cudaCudaOrderOnePotentials[orderOnePotential].height*0.5)
                            cudaForces[3*particleNumber+normal]+=cudaCudaOrderOnePotentials[orderOnePotential].forceConst*(fabsf(r)-cudaCudaOrderOnePotentials[orderOnePotential].height*0.5)/fabsf(r)*r;

                            /// particle radius!
                            r = sqrt(
                                        pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                        +
                                        pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                    );
                            if (r < cudaCudaOrderOnePotentials[orderOnePotential].radius) {
                                cudaForces[3*particleNumber+side1]+=
                                        cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                        *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                        /r
                                        *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]-cudaCoords[3*particleNumber+side1]);
                                cudaForces[3*particleNumber+side2]+=
                                        cudaCudaOrderOnePotentials[orderOnePotential].forceConst
                                        *(r-cudaCudaOrderOnePotentials[orderOnePotential].radius)
                                        /r
                                        *(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]-cudaCoords[3*particleNumber+side2]);

                            }
                        }
                }/// end Cylinder
                else if(cudaCudaOrderOnePotentials[orderOnePotential].type==3){/// Sphere
                        double dist = 0;
                        for(int dim=0; dim<3; ++dim){
                            dist += (cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]-cudaCoords[3*particleNumber+dim])*(cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]-cudaCoords[3*particleNumber+dim]);
                        }
                        dist = sqrt(dist);

                        if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==1 && dist>cudaCudaOrderOnePotentials[orderOnePotential].radius){/// attractive
                            double precompute = cudaCudaOrderOnePotentials[orderOnePotential].forceConst * (dist - cudaCudaOrderOnePotentials[orderOnePotential].radius) / dist;
                            for(int dim=0; dim<3; ++dim){
                                cudaForces[3*particleNumber+dim] += -precompute * (cudaCudaOrderOnePotentials[orderOnePotential].origin[dim] - cudaCoords[3*particleNumber+dim]);
                            }
                        }
                        else if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==2 && dist<cudaCudaOrderOnePotentials[orderOnePotential].radius){/// repulsive
                            double precompute = cudaCudaOrderOnePotentials[orderOnePotential].forceConst * (dist - cudaCudaOrderOnePotentials[orderOnePotential].radius) / dist;
                            for(int dim=0; dim<3; ++dim){
                                cudaForces[3*particleNumber+dim] += -precompute * (cudaCudaOrderOnePotentials[orderOnePotential].origin[dim] - cudaCoords[3*particleNumber+dim]);
                            }
                        }
                }/// end Sphere
                else if(cudaCudaOrderOnePotentials[orderOnePotential].type==4){/// Box
                        if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==1){/// attractive
                            for(int dim=0; dim<3; ++dim){
                                double distToBoxBegin = cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]+cudaCudaOrderOnePotentials[orderOnePotential].extension[dim]-(considerParticleRadius ? cudaParticleRadiiMatrix[cudaTypes[particleNumber]] : 0);
                                if(cudaCoords[3*particleNumber+dim] > distToBoxBegin){
                                    cudaForces[3*particleNumber+dim]+= -cudaCudaOrderOnePotentials[orderOnePotential].forceConst *
                                            (distToBoxBegin - cudaCoords[3*particleNumber+dim]);
                                }
                                else{
                                    double distToBoxEnd = cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]+(considerParticleRadius ? cudaParticleRadiiMatrix[cudaTypes[particleNumber]] : 0);
                                    if(cudaCoords[3*particleNumber+dim] < distToBoxEnd){
                                    cudaForces[3*particleNumber+dim]+= cudaCudaOrderOnePotentials[orderOnePotential].forceConst *
                                            (cudaCoords[3*particleNumber+dim]-distToBoxEnd);
                                    }
                                }
                            }
                        }
                        else if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==2){/// repulsive
                            for(int dim=0; dim<3; ++dim){
                                double distToBoxBegin = cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]+cudaCudaOrderOnePotentials[orderOnePotential].extension[dim]+(considerParticleRadius ? cudaParticleRadiiMatrix[cudaTypes[particleNumber]] : 0);
                                if(cudaCoords[3*particleNumber+dim] > distToBoxBegin){
                                    cudaForces[3*particleNumber+dim] += cudaCudaOrderOnePotentials[orderOnePotential].forceConst *
                                            (distToBoxBegin - cudaCoords[3*particleNumber+dim]);
                                }
                                else{
                                    double distToBoxEnd = cudaCudaOrderOnePotentials[orderOnePotential].origin[dim]-(considerParticleRadius ? cudaParticleRadiiMatrix[cudaTypes[particleNumber]] : 0);
                                    if(cudaCoords[3*particleNumber+dim] < distToBoxEnd){
                                    cudaForces[3*particleNumber+dim] += -cudaCudaOrderOnePotentials[orderOnePotential].forceConst *
                                            (cudaCoords[3*particleNumber+dim]-distToBoxEnd);
                                    }
                                }
                            }
                        }
                }/// end Box
            }/// endif order one potentials matrix
        }/// end iterate over order one potentials

        globalRandStates[particleNumber] = localState;
    }
    return;
}

__device__ void calculateOrderTwoPotential(int particleNumber, int interactingParticle, int orderTwoPotentialNr, double r, double* cudaCoords, double* cudaForces, int* cudaTypes, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix){

    double r0 = cudaCollisionRadiiMatrix[cudaTypes[particleNumber]*numberOfParticleTypes+cudaTypes[interactingParticle]];
    if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].type==1){        /// Harmonic Potential
        double precompute = cudaCudaOrderTwoPotentials[orderTwoPotentialNr].forceConst * (r-r0)/r;
        for(int coord=0; coord<3; ++coord){
            double force = precompute * (cudaCoords[interactingParticle*3+coord]-cudaCoords[particleNumber*3+coord]);
            if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==1 && r>r0){ /// attractive
                atomicAdd(&cudaForces[particleNumber*3+coord], -force );
                atomicAdd(&cudaForces[interactingParticle*3+coord], force );
            }
            else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==2 && r<r0){ /// repulsive
                atomicAdd(&cudaForces[particleNumber*3+coord], -force );
                atomicAdd(&cudaForces[interactingParticle*3+coord], force );
            }
            else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==3){ /// spring
                atomicAdd(&cudaForces[particleNumber*3+coord], -force );
                atomicAdd(&cudaForces[interactingParticle*3+coord], force );
            }
        }
    }
    else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].type==2){     /// Harmonic weak interaction Potential
        double iradius = cudaCudaOrderTwoPotentials[orderTwoPotentialNr].length + r0;
        if (r < iradius && r > r0) {
            double precompute =  ( cudaCudaOrderTwoPotentials[orderTwoPotentialNr].forceConst * (-r0 + r) * (iradius -r));
            for(int coord=0; coord<3; ++coord){
                double force = precompute * (cudaCoords[interactingParticle*3+coord]-cudaCoords[particleNumber*3+coord]);
                atomicAdd(&cudaForces[particleNumber*3+coord], -force );
                atomicAdd(&cudaForces[interactingParticle*3+coord], force );
            }
        }
    }
}

__device__ void getNeighbors(int particleNumber, int * todo, double* cudaCoords, int * cudaLatticeSize, double * cudaBoxSize, int maxCutoff){

    int x,y,z;
    int field=((int)floor((cudaCoords[3*particleNumber+2]-cudaBoxSize[4])/maxCutoff)%cudaLatticeSize[2])*cudaLatticeSize[0]*cudaLatticeSize[1]
             +((int)floor((cudaCoords[3*particleNumber+1]-cudaBoxSize[2])/maxCutoff)%cudaLatticeSize[1])*cudaLatticeSize[0]
             +((int)floor((cudaCoords[3*particleNumber+0]-cudaBoxSize[0])/maxCutoff)%cudaLatticeSize[0]);

    /// surrounding, for calculation imprtant fields
    /// TODO: CHECK! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for(x=-1; x<2;x++){
            for(y=-1; y<2;y++){
                for(z=-1; z<2;z++){
                    todo[(x+1)+(y+1)*3+(z+1)*9]=
                    (
                        (
                            (field%(cudaLatticeSize[0]))
                            +x+cudaLatticeSize[0]
                        )
                        %cudaLatticeSize[0]
                    )

                    +cudaLatticeSize[0]*
                    (
                        (
                            (int)floorf
                            (
                                (float)(field%(cudaLatticeSize[0]*cudaLatticeSize[1]))
                                /
                                (float)(cudaLatticeSize[0])
                            )
                            +y+cudaLatticeSize[1]
                        )
                        %cudaLatticeSize[1]
                    )

                    +cudaLatticeSize[0]*cudaLatticeSize[1]*
                    (
                        (
                            (int)floorf
                            (
                                (float)(field)
                                /
                                (float)(cudaLatticeSize[0]*cudaLatticeSize[1])
                            )
                            +z+cudaLatticeSize[2]
                        )
                        %cudaLatticeSize[2]
                    );
                }
            }
        }


}

__global__ void orderTwo(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderTwoPotentialsMatrix, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix){

    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

    if(particleNumber<numberParticles){
        //curandState localState = globalRandStates[particleNumber];

        int todo[27];

        getNeighbors(particleNumber, todo, cudaCoords, cudaLatticeSize, cudaBoxSize, maxCutoff);

        /// do calculation of forces and maybe reactions here ...

            /** TODO!!:
             *bring matrices to shared mem
             *interaction matrix could contain the specific cutoffs
             *think about parameter storage
             *then dynamik arrays
             *later think about accellerations due to ideas below
             *
             * need:   - interaction matices (radii, forces) for every force (aligned in one array, +array size)
             *      - more parameter? how to store?
             *  data alignment: x,y,z,type,rand?,force?
             *
             * call voxel per warp(n threads)
             * load first n coords in shared mem
             * calculate distances to particles in surrounding fields (always load one particle and calc. n distances)
             *
             * check for periodic boundaries ...
             */


            /// loop over all "todo" fields around the current field
            for(int x=0; x<27; x++){
                /// begin link to the first element from the list of the field
                int interactingParticle=cudaNeighborListBegins[todo[x]];
                if(interactingParticle!=-1){
                    do
                    {
                        //if(interactingParticle!=particleNumber){
                        /// calculating interaction just once, and apply it for both particles
                        if(interactingParticle<particleNumber){

                            float r=0.0f;
                            float rij[3];
                            for (int dim=0;dim<3;dim++){
                                rij[dim]=cudaCoords[3*particleNumber+dim]-cudaCoords[3*interactingParticle+dim];
                                if(rij[dim]>( (cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
                                if(rij[dim]<(-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]+(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
                                r=r+(rij[dim]*rij[dim]);
                            }
                            r=sqrtf(r);

                            for(int orderTwoPotentialNr=0; orderTwoPotentialNr<numberOfOrderTwoPotentials; ++orderTwoPotentialNr){
                                if(cudaOrderTwoPotentialsMatrix[cudaTypes[particleNumber]*numberOfParticleTypes+cudaTypes[interactingParticle]*numberOfParticleTypes+orderTwoPotentialNr]!=0 && r<=maxCutoff){
                                    calculateOrderTwoPotential(particleNumber, interactingParticle, orderTwoPotentialNr, r, cudaCoords, cudaForces, cudaTypes, cudaCudaOrderTwoPotentials, numberOfParticleTypes, cudaCollisionRadiiMatrix);
                                }
                            }
                        }
                        interactingParticle=cudaNeighborList[2*interactingParticle+1];
                    }
                    while(interactingParticle!=cudaNeighborListBegins[todo[x]]);/// loop/list end
                }
            }
        //globalRandStates[particleNumber] = localState;
    }
    return;
}

__device__ void warpGetNeighbors(int field, int * todo, int * cudaLatticeSize){

    if(threadIdx.x<27){
        int x,y,z;

        /// /////////////////////////////////////////////////////////////
        /// modulo is slow!!
        /// /////////////////////////////////////////////////////////////

        x=threadIdx.x/9%3 -1;
        y=threadIdx.x/3%3 -1;
        z=threadIdx.x%3 -1;

        //for(x=-1; x<2;x++){
            //for(y=-1; y<2;y++){
                //for(z=-1; z<2;z++){
                    todo[(x+1)+(y+1)*3+(z+1)*9]=
                    (
                        (
                            (field%(cudaLatticeSize[0]))
                            +x+cudaLatticeSize[0]
                        )
                        %cudaLatticeSize[0]
                    )

                    +cudaLatticeSize[0]*
                    (
                        (
                            (int)floorf
                            (
                                (float)(field%(cudaLatticeSize[0]*cudaLatticeSize[1]))
                                /
                                (float)(cudaLatticeSize[0])
                            )
                            +y+cudaLatticeSize[1]
                        )
                        %cudaLatticeSize[1]
                    )

                    +cudaLatticeSize[0]*cudaLatticeSize[1]*
                    (
                        (
                            (int)floorf
                            (
                                (float)(field)
                                /
                                (float)(cudaLatticeSize[0]*cudaLatticeSize[1])
                            )
                            +z+cudaLatticeSize[2]
                        )
                        %cudaLatticeSize[2]
                    );
                //}
            //}
        //}
    }
}

__device__ void calculateOrderTwoPotential(double * particleCoord, double * particleForce, int particleType, double * interactingParticleCoord, double * interactingParticleForce, int interactingParticleType, int orderTwoPotentialNr, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, double * cudaBoxSize, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix){

    float r=0.0f;
    float rij[3];
    for (int dim=0;dim<3;dim++){
        rij[dim]=particleCoord[dim]-interactingParticleCoord[dim];
        if(rij[dim]>( (cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
        if(rij[dim]<(-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]+(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
        r=r+(rij[dim]*rij[dim]);
    }
    r=sqrtf(r);

    double r0 = cudaCollisionRadiiMatrix[particleType*numberOfParticleTypes+interactingParticleType];
    if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].type==1){        /// Harmonic Potential
        double precompute = cudaCudaOrderTwoPotentials[orderTwoPotentialNr].forceConst * (r-r0)/r;
        for(int dim=0; dim<3; ++dim){
        /*    double force = precompute * (interactingParticleCoord[dim]-particleCoord[dim]);
            if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==1 && r>r0){ /// attractive
                atomicAdd(&particleForce[dim], -force );
                atomicAdd(&interactingParticleForce[dim], force );
            }
            else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==2 && r<r0){ /// repulsive
                atomicAdd(&particleForce[dim], -force );
                atomicAdd(&interactingParticleForce[dim], force );
            }
            else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].subtype==3){ /// spring
                atomicAdd(&particleForce[dim], -force );
                atomicAdd(&interactingParticleForce[dim], force );
            }*/
        }
    }
    else if(cudaCudaOrderTwoPotentials[orderTwoPotentialNr].type==2){     /// Harmonic weak interaction Potential
        double iradius = cudaCudaOrderTwoPotentials[orderTwoPotentialNr].length + r0;
        if (r < iradius && r > r0) {
            double precompute =  ( cudaCudaOrderTwoPotentials[orderTwoPotentialNr].forceConst * (-r0 + r) * (iradius -r));
            for(int dim=0; dim<3; ++dim){
                double force = precompute * (interactingParticleCoord[dim]-particleCoord[dim]);
              //  atomicAdd(&particleForce[dim], -force );
              //  atomicAdd(&interactingParticleForce[dim], force );
            }
        }
    }
}

__global__ void warpOrderTwo(int warpsize, double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderTwoPotentialsMatrix, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, int numberOfOrderTwoPotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix){

    //int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;
    //curandState localState;
    //curandState localState = globalRandStates[particleNumber]
    ;
    int thread = threadIdx.x;
    int field = blockIdx.x;
    int particleNumber = cudaNeighborListBegins[field];
    if(particleNumber==-1){
        return;
    }

    /// link to shared memory
    extern __shared__ double s[];
    //int *integerData = s;                        // nI ints
    //float *floatData = (float*)&integerData[nI]; // nF floats
    //char *charData = (char*)&floatData[nF];      // nC chars

    //__shared__ double localParticleCoords[warpsize*3];
    double * localParticleCoords = s;
    //__shared__ double localParticleForces[warpsize*3];
    double * localParticleForces = &localParticleCoords[warpsize*3];
    //__shared__ double localInteractingParticleCoords[warpsize*3];
    double * localInteractingParticleCoords = &localParticleForces[warpsize*3];
    //__shared__ double localInteractingParticleForces[warpsize*3];
    double * localInteractingParticleForces = &localInteractingParticleCoords[warpsize*3];

    //__shared__ int todo[27];
    int * todo = (int*)&localInteractingParticleForces[warpsize*3];
    //__shared__ int localParticleTypes[warpsize];
    int * localParticleTypes = &todo[27];
    //__shared__ int localInteractingParticleTypes[warpsize];
    int * localInteractingParticleTypes = &localParticleTypes[warpsize];
    //__shared__ int localInteractingParticleNumber[warpsize];
    int * localInteractingParticleNumber = &localInteractingParticleTypes[warpsize];
    //__shared__ int lastParticle;
    int * lastParticle = &localInteractingParticleNumber[warpsize];
    //__shared__ int end;
    int * end = &lastParticle[1];
    //__shared__ int todoEnd;
    int * todoEnd = &end[1];
    //__shared__ int lastTodoField;
    int * lastTodoField = &todoEnd[1];
    //__shared__ int lastInteractingParticle;
    int * lastInteractingParticle = &lastTodoField[1];
    /// sum = (warpsize*3*4*sizeof(double)+(27+warpsize*3+5)*sizeof(int))

    warpGetNeighbors(field, todo, cudaLatticeSize);

    /// iterate over particles in this field
    do{
        /// assign each thread a particle //////////////////////////////////////////////////////////////////
        for(int i=0; i<thread; i++){
            particleNumber = cudaNeighborList[particleNumber*2+1];
            if(particleNumber == cudaNeighborListBegins[field]){
                atomicExch(end, 1);
                return;
            }
        }
        /// store particle properties local
        for(int dim=0; dim<3; ++dim){
            localParticleCoords[3*thread+dim]=cudaCoords[3*particleNumber+dim];
            localParticleForces[3*thread+dim]=0;
        }
        localParticleTypes[thread]=cudaTypes[particleNumber];


        int x=0;
        int interactingParticleNumber=cudaNeighborListBegins[todo[x]];

        /// for all interacting fields ///////////////////////////////////////////////////////////////////
        do{
            /// get one interacting particle for all threads
            for(int i=0; i<=thread && x>27; i++){
                /// first(0) thread gets first particle in list
                /// if it occures again, we had all from this field
                if(interactingParticleNumber == cudaNeighborListBegins[todo[x]] && i!=0){
                    ++x;    /// next todo field
                    interactingParticleNumber = cudaNeighborListBegins[todo[x]];   /// start at next todo field
                    --i;
                    continue;
                }
                /// field is empty
                if(interactingParticleNumber==-1){
                    ++x;    /// next todo field
                    interactingParticleNumber = cudaNeighborListBegins[todo[x]];   /// start at next todo field
                    --i;
                    continue;
                }
                interactingParticleNumber = cudaNeighborList[interactingParticleNumber*2+1];
            }
            /// store interacting particle properties local
            for(int dim=0; dim<3; ++dim){
                localInteractingParticleCoords[3*thread+dim]=cudaCoords[3*interactingParticleNumber+dim];
                localInteractingParticleForces[3*thread+dim]=0;
            }
            localInteractingParticleTypes[thread]=cudaTypes[interactingParticleNumber];
            localInteractingParticleNumber[thread]=interactingParticleNumber;



            /// calc
            /// iterate more intelligent -> not double, order
            for(int interactingThread=0; interactingThread<warpsize; ++interactingParticleNumber){
                interactingParticleNumber = localInteractingParticleNumber[interactingThread];
                if(interactingParticleNumber<particleNumber){

                    for(int orderTwoPotentialNr=0; orderTwoPotentialNr<numberOfOrderTwoPotentials; ++orderTwoPotentialNr){
                        if(cudaOrderTwoPotentialsMatrix[cudaTypes[particleNumber]*numberOfParticleTypes+cudaTypes[interactingParticleNumber]*numberOfParticleTypes+orderTwoPotentialNr]!=0){
                            //__device__ void calculateOrderTwoPotential(double * particleCoord, double * particleForce, int particleType, double * interactingParticleCoord, double * interactingParticleForce, int interactingParticleType, int orderTwoPotentialNr, CudaOrderTwoPotential * cudaCudaOrderTwoPotentials, double * cudaBoxSize, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);
                            calculateOrderTwoPotential(&localParticleCoords[thread], &localParticleForces[thread], localParticleTypes[thread], &localInteractingParticleCoords[interactingParticleNumber], &localInteractingParticleForces[interactingParticleNumber], localInteractingParticleTypes[interactingParticleNumber], orderTwoPotentialNr, cudaCudaOrderTwoPotentials, cudaBoxSize, numberOfParticleTypes, cudaCollisionRadiiMatrix);
                            //calculateOrderTwoPotential(particleNumber, interactingParticle, orderTwoPotentialNr, r, cudaCoords, cudaForces, cudaTypes, cudaCudaOrderTwoPotentials, numberOfParticleTypes, cudaCollisionRadiiMatrix);
                        }
                    }
                }
            }

            interactingParticleNumber = localInteractingParticleNumber[thread];
            /// just the last thread, with the last interacting particle stores its number and field, so that the threads can start again from there.
            if(blockIdx.x==warpsize){
                /// there is a next particle in this field
                if(cudaNeighborList[interactingParticleNumber*2+1]!=cudaNeighborListBegins[x]){
                    /// store the particle and field number
                    atomicExch(lastInteractingParticle, cudaNeighborList[interactingParticleNumber*2+1]);
                    atomicExch(lastTodoField, x);
                    atomicExch(todoEnd, 0);    /// we are not done jet
                }
                /// if we are done with this todo field
                else{
                    atomicExch(todoEnd, 1);    /// we are done, if there are no more particle -> check
                    /// as long, as we have fields to consider (todo)
                    while(x>27){
                        ++x;    /// take next todo field
                        if(cudaNeighborListBegins[todo[x]]==-1)   /// empty?
                            continue;                       /// again. -> take next
                        else{   /// todo field not empty -> store field and first particle
                            atomicExch(lastInteractingParticle, cudaNeighborListBegins[todo[x]]);
                            atomicExch(lastTodoField, x);
                            atomicExch(todoEnd, 0);    /// we are not done jet
                        }
                    }
                }
            }
            /// block wise sync
            __syncthreads();
            x=lastTodoField[0];
            interactingParticleNumber=lastInteractingParticle[0];
        }while(todoEnd[0]==0); /// end interacting fields

        /// store locally saved changes global (interacting particles)

        /// if more particle in this field continue
        if(blockIdx.x==warpsize){
            if(cudaNeighborList[particleNumber*2+1]!=cudaNeighborListBegins[field]){
                atomicExch(lastParticle, cudaNeighborList[particleNumber*2+1]);
                atomicExch(end, 0);
            }
            else{
                atomicExch(end, 1);
            }
        }
        /// block wise sync
        __syncthreads();
        particleNumber=lastParticle[0];
    }while(end[0]==0); /// end loop over particles from this field

    /// store locally saved changes global (particles)

    //globalRandStates[particleNumber] = localState;
    return;
}

__global__ void calculateRDF(int * cudaRDFMatrix, double* cudaCoords, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, int numberParticles, int maxCutoff, int numberOfParticleTypes, int numberOfBins){

    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

    if(particleNumber<numberParticles){

        int todo[27];

        getNeighbors(particleNumber, todo, cudaCoords, cudaLatticeSize, cudaBoxSize, maxCutoff);

        /// loop over all "todo" fields around the current field
        for(int x=0; x<27; x++){
            /// begin link to the first element from the list of the field
            int interactingParticle=cudaNeighborListBegins[todo[x]];
            if(interactingParticle!=-1){
                do
                {
                    //if(interactingParticle!=particleNumber){
                    /// calculating interaction just once, and apply it for both particles
                    if(interactingParticle<particleNumber){

                        float r=0.0f;
                        float rij[3];
                        for (int dim=0;dim<3;dim++){
                            rij[dim]=cudaCoords[3*particleNumber+dim]-cudaCoords[3*interactingParticle+dim];
                            if(rij[dim]>((cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
                            if(rij[dim]<(-(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2])/2)){rij[dim]=rij[dim]+(cudaBoxSize[dim*2+1]-cudaBoxSize[dim*2]);}
                            r=r+(rij[dim]*rij[dim]);
                        }
                        r=sqrtf(r);
/// carefull about reactions! !!!!!!!!!!!!!!!!!!!!!!!!!!
                        if(r<maxCutoff){
                            int bin = (int)(r*numberOfBins/maxCutoff);
                            atomicAdd(&cudaRDFMatrix[cudaTypes[particleNumber]*numberOfParticleTypes*numberOfBins+cudaTypes[interactingParticle]*numberOfBins+bin], 1);
                            atomicAdd(&cudaRDFMatrix[cudaTypes[interactingParticle]*numberOfParticleTypes*numberOfBins+cudaTypes[particleNumber]*numberOfBins+bin], 1);
                        }
                    }
                    interactingParticle=cudaNeighborList[2*interactingParticle+1];
                }
                while(interactingParticle!=cudaNeighborListBegins[todo[x]]);/// loop/list end
            }
        }
    }
}

__global__ void groups(double* cudaCoords, double* cudaForces, int* cudaTypes, int numberParticles, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix, CudaOrderTwoPotential * cudaCudaGroupPotentials, GroupPart * cudaIndividualGroups, int numberOfIndividualGroups, double * cudaBoxSize){

    int pairNumber=blockIdx.x * blockDim.x + threadIdx.x;

    if(pairNumber<numberOfIndividualGroups){
        float r=0.0f;
        float rij[3];
        for (int dimension=0;dimension<3;dimension++){
            rij[dimension]=cudaCoords[3*cudaIndividualGroups[pairNumber].particle1+dimension]-cudaCoords[3*cudaIndividualGroups[pairNumber].particle2+dimension];
            if(rij[dimension]>((cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2])/2)){rij[dimension]=rij[dimension]-(cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2]);}
            if(rij[dimension]<(-(cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2])/2)){rij[dimension]=rij[dimension]+(cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2]);}
            r=r+(rij[dimension]*rij[dimension]);
        }
        r=sqrtf(r);

        calculateOrderTwoPotential(cudaIndividualGroups[pairNumber].particle1, cudaIndividualGroups[pairNumber].particle2, cudaIndividualGroups[pairNumber].groupPot, r, cudaCoords, cudaForces, cudaTypes, cudaCudaGroupPotentials, numberOfParticleTypes, cudaCollisionRadiiMatrix);
    }
}

__global__ void update(double* cudaCoords, double* cudaForces, int* cudaTypes, double* cudaD, int * cudaNeighborList, int * cudaNeighborListBegins, double * cudaBoxSize, int * cudaSemaphore, curandState* globalRandStates, double dt, int numberParticles, double KB, double T, double maxCutoff, int * cudaLatticeSize){

    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

    if(particleNumber<numberParticles){
        curandState localState = globalRandStates[particleNumber];

        int oldVoxel=   ((int)floor((cudaCoords[3*particleNumber+2]-cudaBoxSize[4])/maxCutoff)%cudaLatticeSize[2])*cudaLatticeSize[0]*cudaLatticeSize[1]
                        +((int)floor((cudaCoords[3*particleNumber+1]-cudaBoxSize[2])/maxCutoff)%cudaLatticeSize[1])*cudaLatticeSize[0]
                        +((int)floor((cudaCoords[3*particleNumber+0]-cudaBoxSize[0])/maxCutoff)%cudaLatticeSize[0]);

        for(int dimension=0; dimension<3; ++dimension){
            /// apply diffusion and forces -> update positions
            /// x(t+dt) = x(t) - dt*D*(F(x(t))/kT) + sqrt(2Ddt)*N(0,1)
            cudaCoords[particleNumber*3+dimension] += -dt*cudaD[cudaTypes[particleNumber]]*cudaForces[particleNumber*3+dimension]/KB/T + sqrt(2*cudaD[cudaTypes[particleNumber]]*dt)*curand_normal( &localState );
            //cudaCoords[particleNumber*3+dimension] += -dt*cudaD[cudaTypes[particleNumber]]*cudaForces[particleNumber*3+dimension]/KB/T ;
            //cudaCoords[particleNumber*3+dimension] += cudaForces[particleNumber*3+dimension] ;
            cudaForces[particleNumber*3+dimension]=0.0f;
            /// periodic boundary condition
            while(cudaCoords[3*particleNumber+dimension]>cudaBoxSize[dimension*2+1]){cudaCoords[3*particleNumber+dimension]=cudaCoords[3*particleNumber+dimension]-(cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2+0]);}
            while(cudaCoords[3*particleNumber+dimension]<cudaBoxSize[dimension*2+0]){cudaCoords[3*particleNumber+dimension]=cudaCoords[3*particleNumber+dimension]+(cudaBoxSize[dimension*2+1]-cudaBoxSize[dimension*2+0]);}
        }
        /// lattice field changed?
        int newVoxel=   ((int)floor((cudaCoords[3*particleNumber+2]-cudaBoxSize[4])/maxCutoff)%cudaLatticeSize[2])*cudaLatticeSize[0]*cudaLatticeSize[1]
                    +((int)floor((cudaCoords[3*particleNumber+1]-cudaBoxSize[2])/maxCutoff)%cudaLatticeSize[1])*cudaLatticeSize[0]
                    +((int)floor((cudaCoords[3*particleNumber+0]-cudaBoxSize[0])/maxCutoff)%cudaLatticeSize[0]);

        /// apply voxel-changes ...
        if(newVoxel!=oldVoxel){
            bool leaveLoop = false;
            /// delete form old list
            while(!leaveLoop){
                /// Lock
                if(atomicExch(&(cudaSemaphore[oldVoxel]),1)==0){
                    int prev=cudaNeighborList[2*particleNumber];
                    int next=cudaNeighborList[2*particleNumber+1];
                    cudaNeighborList[2*prev+1]=next;
                    cudaNeighborList[2*next]=prev;
                    /// was this partilce begin of the linked list?
                    if(cudaNeighborListBegins[oldVoxel]==particleNumber){
                        /// was the particle the only one in this field?
                        if(cudaNeighborList[2*particleNumber]==particleNumber){
                            cudaNeighborListBegins[oldVoxel]=-1;
                        }
                        else{
                            cudaNeighborListBegins[oldVoxel]=cudaNeighborList[2*particleNumber+1];
                        }
                    }
                    leaveLoop=true;
                    /// unLock
                    atomicExch(&(cudaSemaphore[oldVoxel]),0);
                }
            }
            leaveLoop = false;
            /// push ontop of the new list
            while(!leaveLoop){
                /// Lock
                if(atomicExch(&(cudaSemaphore[newVoxel]),1)==0){
                    /// is new list empty?
                    if(cudaNeighborListBegins[newVoxel]!=-1){/// no
                        cudaNeighborList[2*particleNumber]=cudaNeighborList[2*cudaNeighborListBegins[newVoxel]];
                        cudaNeighborList[2*particleNumber+1]=cudaNeighborListBegins[newVoxel];
                        cudaNeighborList[2*cudaNeighborList[2*cudaNeighborListBegins[newVoxel]]+1]=particleNumber;;
                        cudaNeighborList[2*cudaNeighborListBegins[newVoxel]]=particleNumber;
                        cudaNeighborListBegins[newVoxel]=particleNumber;
                    }
                    else{/// first one in new list
                        cudaNeighborList[2*particleNumber+1]=particleNumber;
                        cudaNeighborList[2*particleNumber]=particleNumber;
                        cudaNeighborListBegins[newVoxel]=particleNumber;
                    }
                    leaveLoop=true;
                    /// unLock
                    atomicExch(&(cudaSemaphore[newVoxel]),0);
                }
            }
        }

        globalRandStates[particleNumber] = localState;
    }
    return;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed, int n ){
    int id=blockIdx.x * blockDim.x + threadIdx.x;
    if(id<n){
        curand_init ( seed, id, 0, &state[id] );
    }
}


/// pos force radii forceconst types todo links linkbegins
__device__ void lennardJones(){

    return;
}
