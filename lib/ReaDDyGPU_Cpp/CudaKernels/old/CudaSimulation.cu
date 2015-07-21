
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
__global__ void orderOne(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderOnePotentialsMatrix, CudaOrderOnePotential * cudaCudaOrderOnePotentials, int numberOfOrderOnePotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix);
__global__ void orderTwo(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff);
__global__ void setup_kernel ( curandState * state, unsigned long seed, int n );


CudaSimulation::CudaSimulation(Simulation* simulation){

    this->simulation = simulation;

}

CudaOrderOnePotential toCudaOrderOnePotential(OrderOnePotential* orderOnePotential){
    CudaOrderOnePotential cudaOrderOnePotential = CudaOrderOnePotential();
    cudaOrderOnePotential.subtype=orderOnePotential->subtypeID;
    if(orderOnePotential->type.compare("DISK")==0){
        DiskPotential * diskPotential = reinterpret_cast<DiskPotential*>(orderOnePotential);
        cudaOrderOnePotential.type=1;
        cudaOrderOnePotential.forceConst=diskPotential->forceConst;
        std::copy ( diskPotential->center, diskPotential->center+3, cudaOrderOnePotential.origin );
        //cudaOrderOnePotential.origin=diskPotential->center;
        std::copy ( diskPotential->normal, diskPotential->normal+3, cudaOrderOnePotential.normal );
        //cudaOrderOnePotential.normal=diskPotential->normal;
        cudaOrderOnePotential.radius=diskPotential->radius;
    }
    else if(orderOnePotential->type.compare("CYLINDER")==0){
        CylinderPotential * cylinderPotential = reinterpret_cast<CylinderPotential*>(orderOnePotential);
        cudaOrderOnePotential.type=2;
        cudaOrderOnePotential.forceConst=cylinderPotential->forceConst;
        std::copy ( cylinderPotential->center, cylinderPotential->center+3, cudaOrderOnePotential.origin );
        //cudaOrderOnePotential.origin=cylinderPotential->center;
        std::copy ( cylinderPotential->normal, cylinderPotential->normal+3, cudaOrderOnePotential.normal );
        //cudaOrderOnePotential.normal=cylinderPotential->normal;
        cudaOrderOnePotential.radius=cylinderPotential->radius;
        cudaOrderOnePotential.height=cylinderPotential->height;
    }
    return cudaOrderOnePotential;
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

    /// initialize cuRand
    cudaMalloc ( (void**)&globalRandStates, simulation->numberParticles * sizeof( curandState ) );
    /// setup seeds
    setup_kernel <<< gridSize, blockSize >>> ( globalRandStates, time(NULL), simulation->numberParticles );


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

    /// create cuda collision radii matix -> matrix[nTypes+1]*[nTypes] (+1 for default)
    hostCollisionRadiiMatrix = new double[(simulation->particleTypes.size()+1)*simulation->particleTypes.size()];
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        hostCollisionRadiiMatrix[i]=simulation->particleTypes[i].defaultRadius;
    }
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            hostCollisionRadiiMatrix[(i+1)*simulation->particleTypes.size()+j]=simulation->particleTypes[i].radiiMatrix[j];
        }
    }
    cudaMalloc((void**)&cudaCollisionRadiiMatrix,( (simulation->particleTypes.size()+1)*simulation->particleTypes.size() * sizeof ( double ) ));
    cudaMemcpy(cudaCollisionRadiiMatrix, hostCollisionRadiiMatrix, ( (simulation->particleTypes.size()+1)*simulation->particleTypes.size() * sizeof ( double ) ), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ){
        printf( "cuda error during initialization: %s\n",cudaGetErrorString(error) );
        return 1;
    }

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
    if(simulation->testmode){
        cout << "lattice informations: " << endl;
        cout << "simulation size x[nm]: " << boxSize[1]-boxSize[0] << endl;
        cout << "simulation size y[nm]: " << boxSize[3]-boxSize[2] << endl;
        cout << "simulation size z[nm]: " << boxSize[5]-boxSize[4] << endl;
        cout << "number of voxels:      " << numberOfLatticeFields << endl;
        cout << "voxel edge length:     " << maxCutoff << endl;
        cout << "lattice size x:        " << latticeSize[0] << endl;
        cout << "lattice size y:        " << latticeSize[1] << endl;
        cout << "lattice size z:        " << latticeSize[2] << endl << endl;
    }

    for(int i=0; i<simulation->numberParticles; ++i){

        int field=((int)floor((simulation->coords[3*i+2]-boxSize[4])/maxCutoff)%latticeSize[2])*latticeSize[0]*latticeSize[1]
                 +((int)floor((simulation->coords[3*i+1]-boxSize[2])/maxCutoff)%latticeSize[1])*latticeSize[0]
                 +((int)floor((simulation->coords[3*i+0]-boxSize[0])/maxCutoff)%latticeSize[0]);
        /*
        cout << "particle nr: " << i << endl;
        cout << "x: " << simulation->coords[3*i+0] << endl;
        cout << "y: " << simulation->coords[3*i+1] << endl;
        cout << "z: " << simulation->coords[3*i+2] << endl;
        cout << ((int)floor((simulation->coords[3*i+2]-boxSize[4])/maxCutoff)%latticeSize[2]) << endl;
        cout << "-> " << ((int)floor((simulation->coords[3*i+2]-boxSize[4])/maxCutoff)%latticeSize[2])*latticeSize[0]*latticeSize[1] << endl;
        cout << ((int)floor((simulation->coords[3*i+1]-boxSize[2])/maxCutoff)%latticeSize[1]) << endl;
        cout << "-> " << ((int)floor((simulation->coords[3*i+1]-boxSize[2])/maxCutoff)%latticeSize[1])*latticeSize[0] << endl;
        cout << ((int)floor((simulation->coords[3*i+0]-boxSize[0])/maxCutoff)%latticeSize[0]) << endl;
        cout << "-> " << ((int)floor((simulation->coords[3*i+0]-boxSize[0])/maxCutoff)%latticeSize[0]) << endl;
        cout << field << endl;*/


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


    //cout << "1" << endl;
    //orderOne<<<1,1>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderOnePotentialsMatrix, cudaCudaOrderOnePotentials, simulation->orderOnePotentials.size(), simulation->particleTypes.size(), cudaCollisionRadiiMatrix);
    orderOne<<<gridSize,blockSize>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff, cudaOrderOnePotentialsMatrix, cudaCudaOrderOnePotentials, simulation->orderOnePotentials.size(), simulation->particleTypes.size(), cudaCollisionRadiiMatrix);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "order one, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }
    //cout << "2" << endl;
    //orderTwo<<<gridSize,blockSize>>>(cudaCoords, cudaForces, cudaTypes, cudaNeighborListBegins, cudaNeighborList, cudaLatticeSize, cudaBoxSize, globalRandStates, simulation->numberParticles, maxCutoff);
    if(simulation->testmode){
        cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if ( cudaSuccess != error ){
                printf( "order two, cuda error: %s\n",cudaGetErrorString(error) );
                return 1;
            }
    }
    //cout << "3" << endl;
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


__global__ void orderOne(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff, int * cudaOrderOnePotentialsMatrix, CudaOrderOnePotential * cudaCudaOrderOnePotentials, int numberOfOrderOnePotentials, int numberOfParticleTypes, double * cudaCollisionRadiiMatrix){


    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

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
                     /*  for(int i=0; i<3; ++i){
                            //cudaCoords[3*particleNumber+i]=cudaCudaOrderOnePotentials[orderOnePotential].origin[i];
                            //cudaForces[3*particleNumber+i]+= 10;
                        }
                       cudaCoords[3*particleNumber+0]=cudaCudaOrderOnePotentials[orderOnePotential].forceConst;
                       cudaCoords[3*particleNumber+1]=cudaCudaOrderOnePotentials[orderOnePotential].radius;
                    }

                    if(false && cudaCudaOrderOnePotentials[orderOnePotential].subtype==1){/// attractive
                    */

                       /* r = distToDiskPlane;// actual
                        r0 = 0;// desired
                        if (r > r0) {
                            precompute = (k * (-r0 + r) / r);

                                gradient[0] = gradient[0]+precompute * (pointOnDiskPlane[0]-coords1[0]);
                                gradient[1] = gradient[1]+precompute * (pointOnDiskPlane[1]-coords1[1]);
                                gradient[2] = gradient[2]+precompute * (pointOnDiskPlane[2]-coords1[2]);

                        }*/
                        double r = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-cudaCoords[3*particleNumber+normal];
                        cudaForces[3*particleNumber+normal]+=-cudaCudaOrderOnePotentials[orderOnePotential].forceConst*r;
                        //cudaForces[3*particleNumber+normal]+=cudaCudaOrderOnePotentials[orderOnePotential].origin[normal];
                        // force within disc plane
                        /*r = distToCenterWithinDiskPlane + pRadius;// actual
                        r0 = diskRadius;// desired
                        if (r > r0) {
                            precompute = (k * (-r0 + r) / r);

                                gradient[0] = gradient[0]+precompute * (center[0]-pointOnDiskPlane[0]);
                                gradient[1] = gradient[1]+precompute * (center[1]-pointOnDiskPlane[1]);
                                gradient[2] = gradient[2]+precompute * (center[2]-pointOnDiskPlane[2]);

                        }*/
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
/*
                        double distToOriginWithinDisk = fminf(
                                                cudaCudaOrderOnePotentials[orderOnePotential].radius
                                                -
                                                sqrt(
                                                    pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                                    +
                                                    pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                                    )
                                            ,0);

                        double force = cudaCudaOrderOnePotentials[orderOnePotential].forceConst * (
                                    pow( cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-cudaCoords[3*particleNumber+normal], 2)
                                    +
                                    pow(distToOriginWithinDisk,2)
                                    );

                        cudaForces[3*particleNumber+normal]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]+cudaCoords[3*particleNumber+normal]);
                        if(distToDisk>cudaCudaOrderOnePotentials[orderOnePotential].radius){
                            // in my opinion this calculation is wrong, because we should take the distance to the edge of the disk, instead the distance to its center
                            cudaForces[3*particleNumber+side1]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]+cudaCoords[3*particleNumber+side1]);
                            cudaForces[3*particleNumber+side2]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]+cudaCoords[3*particleNumber+side2]);
                        }*/
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
/****/                else if(cudaCudaOrderOnePotentials[orderOnePotential].type==2){/// Cylinder

                        int normal, side1, side2;
                        if(cudaCudaOrderOnePotentials[orderOnePotential].normal[0]==1){normal=0;side1=1;side2=2;}
                        else if(cudaCudaOrderOnePotentials[orderOnePotential].normal[1]==1){normal=1;side1=0;side2=2;}
                        else {normal=2;side1=1;side2=0;}

                        if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==1){/// attractive

        /*                    r = distToDiskPlane + pRadius;// actual
                            r0 = 0.5 * this.height;// desired
                            if (r > r0) {
                                precompute = (k * (-r0 + r) / r);

                                gradient[0] = gradient[0] + precompute * (pointOnDiskPlane[0] - coords1[0]);
                                gradient[1] = gradient[1] + precompute * (pointOnDiskPlane[1] - coords1[1]);
                                gradient[2] = gradient[2] + precompute * (pointOnDiskPlane[2] - coords1[2]);

                            }*/
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


               /*             double distToDiskSide = fminf(
                                                    cudaCudaOrderOnePotentials[orderOnePotential].radius
                                                    -
                                                    sqrt(
                                                        pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                                        +
                                                        pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                                        )
                                                ,0);
                            double distToDiskPlane = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]+0.5*cudaCudaOrderOnePotentials[orderOnePotential].height<cudaCoords[3*particleNumber+normal]?
                                                cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]+0.5*cudaCudaOrderOnePotentials[orderOnePotential].height-cudaCoords[3*particleNumber+normal] : 0
                                                +
                                                cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-0.5*cudaCudaOrderOnePotentials[orderOnePotential].height>cudaCoords[3*particleNumber+normal]?
                                                cudaCoords[3*particleNumber+normal]-cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-0.5*cudaCudaOrderOnePotentials[orderOnePotential].height : 0;

                            double force = cudaCudaOrderOnePotentials[orderOnePotential].forceConst * (
                                        pow(distToDiskPlane, 2)
                                        +
                                        pow(distToDiskSide,2)
                                        );

                            cudaForces[3*particleNumber+normal]-=force*(distToDiskPlane);
                            if(distToDiskSide>cudaCudaOrderOnePotentials[orderOnePotential].radius){
                                // in my opinion this calculation is wrong, because we should take the distance to the edge of the disk, instead the distance to its center
                                cudaForces[3*particleNumber+side1]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]+cudaCoords[3*particleNumber+side1]);
                                cudaForces[3*particleNumber+side2]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]+cudaCoords[3*particleNumber+side2]);
                            }*/
                        }

                        else if(cudaCudaOrderOnePotentials[orderOnePotential].subtype==2){/// repulsive

                            double distToDiskSide = fminf(
                                                        sqrt(
                                                            pow(cudaCoords[3*particleNumber+side1]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side1],2)
                                                            +
                                                            pow(cudaCoords[3*particleNumber+side2]- cudaCudaOrderOnePotentials[orderOnePotential].origin[side2],2)
                                                        )
                                                        -
                                                        cudaCudaOrderOnePotentials[orderOnePotential].radius
                                                    ,0);

                            double distToDiskPlane = cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]+0.5*cudaCudaOrderOnePotentials[orderOnePotential].height>cudaCoords[3*particleNumber+normal]?
                                                    cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]+0.5*cudaCudaOrderOnePotentials[orderOnePotential].height-cudaCoords[3*particleNumber+normal] : 0
                                                    +
                                                    cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-0.5*cudaCudaOrderOnePotentials[orderOnePotential].height<cudaCoords[3*particleNumber+normal]?
                                                    cudaCoords[3*particleNumber+normal]-cudaCudaOrderOnePotentials[orderOnePotential].origin[normal]-0.5*cudaCudaOrderOnePotentials[orderOnePotential].height : 0;

                            double force = cudaCudaOrderOnePotentials[orderOnePotential].forceConst * (
                                        pow(distToDiskPlane, 2)
                                        +
                                        pow(distToDiskSide,2)
                                        );

                            cudaForces[3*particleNumber+normal]-=force*(distToDiskPlane);
                            if(distToDiskSide>cudaCudaOrderOnePotentials[orderOnePotential].radius){
                                // in my opinion this calculation is wrong, because we should take the distance to the edge of the disk, instead the distance to its center
                                cudaForces[3*particleNumber+side1]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side1]+cudaCoords[3*particleNumber+side1]);
                                cudaForces[3*particleNumber+side2]-=force*(cudaCudaOrderOnePotentials[orderOnePotential].origin[side2]+cudaCoords[3*particleNumber+side2]);
                            }
                        }
                }/// end Cylinder
            }/// endif order one potentials matrix
        }/// end iterate over order one potentials

        globalRandStates[particleNumber] = localState;
    }
    return;
}

__global__ void orderTwo(double* cudaCoords, double* cudaForces, int* cudaTypes, int * cudaNeighborListBegins, int * cudaNeighborList, int * cudaLatticeSize, double * cudaBoxSize, curandState* globalRandStates, int numberParticles, int maxCutoff){

    int k=blockIdx.x * blockDim.x + threadIdx.x;

    if(k<numberParticles){
        curandState localState = globalRandStates[k];

        int todo[27];
        int x,y,z;

        int field=((int)floor((cudaCoords[3*k+2]-cudaBoxSize[4])/maxCutoff)%cudaLatticeSize[2])*cudaLatticeSize[0]*cudaLatticeSize[1]
                 +((int)floor((cudaCoords[3*k+1]-cudaBoxSize[2])/maxCutoff)%cudaLatticeSize[1])*cudaLatticeSize[0]
                 +((int)floor((cudaCoords[3*k+0]-cudaBoxSize[0])/maxCutoff)%cudaLatticeSize[0]);

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


        /// do calculation of forces and maybe reactions here ...


            /*
             *
             *first:
             *do it plain
             *think about parameter storage
             *then dynamik arrays
             *later think about accellerations due to ideas below
             *
             */




        /*
          for every near particle with higher ID:
            for every force
                if Interaction Matrix != 0
                    calculate necessary forces
                    atomic add force to both interactiong particles

        need:   - interaction matices (radii, forces) for every force (aligned in one array, +array size)
                - more parameter? how to store?
            data alignment: x,y,z,type,rand?,force?
        */
        /*
         * call voxel per warp(n threads)
         * load first n coords in shared mem
         * calculate distances to particles in surrounding fields (always load one particle and calc. n dist.)
         * calulate all necessary forces somehow
         */
        globalRandStates[k] = localState;
    }
    return;
}

__global__ void update(double* cudaCoords, double* cudaForces, int* cudaTypes, double* cudaD, int * cudaNeighborList, int * cudaNeighborListBegins, double * cudaBoxSize, int * cudaSemaphore, curandState* globalRandStates, double dt, int numberParticles, double KB, double T, double maxCutoff, int * cudaLatticeSize){

    int particleNumber=blockIdx.x * blockDim.x + threadIdx.x;

    if(particleNumber<numberParticles){
        curandState localState = globalRandStates[particleNumber];

        int oldVoxel=   ((int)floor((cudaCoords[3*particleNumber+2]-cudaBoxSize[4])/maxCutoff)%cudaLatticeSize[2])*cudaLatticeSize[0]*cudaLatticeSize[1]
                        +((int)floor((cudaCoords[3*particleNumber+1]-cudaBoxSize[2])/maxCutoff)%cudaLatticeSize[1])*cudaLatticeSize[0]
                        +((int)floor((cudaCoords[3*particleNumber+0]-cudaBoxSize[0])/maxCutoff)%cudaLatticeSize[0]);


        /// check for periodic boundaries ...

        /// /

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
