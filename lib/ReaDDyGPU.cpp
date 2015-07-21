
#include <ReaDDyGPU_Cpp/ReaDDyGPU.hpp>
//#include <ReaDDyGPU_Cpp/ReaDDyGPU.so>
#include <iostream>
#include <string>
#include <vector>
# include <cuda.h>
#include <typeinfo>
#include <time.h>
#include <sys/time.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <sstream>
#include <stdio.h>
#include <sys/time.h>
#include <jni.h>
#include <simulationGPU.h>
#include <string.h>
#include <algorithm>

using namespace std;



/// main function. is called from ReaDDy. Creates and runs the simulation.
JNIEXPORT void JNICALL Java_readdy_impl_sim_top_TopGPU_cCreateSimulation(JNIEnv *env, jobject obj, jboolean testmode, jstring paramParticles, jstring tplgyPotentials, jstring paramGroups, jstring tplgyCoord, jstring tplgyGroups, jstring paramGlobal){

    /// Timer
    timeval startClock;
    gettimeofday(&startClock, 0);

    //testmode=true;
    testmode=false;
    /// in testmode the program gives serveral outputs for simulation progress and for debugging
    if(testmode){
        printf("c code\n");
    }

    if(testmode)
        cout << "get input file path..." << endl;

    char * param_particles;
    const char *jparam_particles = env->GetStringUTFChars(paramParticles, 0);
    param_particles = (char*) malloc(strlen(jparam_particles) + 1);
    strcpy(param_particles, jparam_particles);
    env->ReleaseStringUTFChars( paramParticles, jparam_particles);

    char * tplgy_potentials;
    const char *jtplgy_potentials = env->GetStringUTFChars(tplgyPotentials, 0);
    tplgy_potentials = (char*) malloc(strlen(jtplgy_potentials) + 1);
    strcpy(tplgy_potentials, jtplgy_potentials);
    env->ReleaseStringUTFChars( tplgyPotentials, jtplgy_potentials);

    char * param_groups;
    const char *jparam_groups = env->GetStringUTFChars(paramGroups, 0);
    param_groups = (char*) malloc(strlen(jparam_groups) + 1);
    strcpy(param_groups, jparam_groups);
    env->ReleaseStringUTFChars( paramGroups, jparam_groups);

    char * tplgy_coordinates;
    const char *jtplgyCoord = env->GetStringUTFChars(tplgyCoord, 0);
    tplgy_coordinates = (char*) malloc(strlen(jtplgyCoord) + 1);
    strcpy(tplgy_coordinates, jtplgyCoord);
    env->ReleaseStringUTFChars( tplgyCoord, jtplgyCoord);

    char * tplgy_groups;
    const char *jtplgy_groups = env->GetStringUTFChars(tplgyGroups, 0);
    tplgy_groups = (char*) malloc(strlen(jtplgy_groups) + 1);
    strcpy(tplgy_groups, jtplgy_groups);
    env->ReleaseStringUTFChars( tplgyGroups, jtplgy_groups);

    char * param_global;
    const char *jparam_global = env->GetStringUTFChars(paramGlobal, 0);
    param_global = (char*) malloc(strlen(jparam_global) + 1);
    strcpy(param_global, jparam_global);
    env->ReleaseStringUTFChars( paramGlobal, jparam_global);

    /// get from ReaDDy: ////////////////

    //bool testmode=true;
    //string param_particles      = "ReaDDy_input/param_particles.xml";
    //string tplgy_potentials     = "ReaDDy_input/tplgy_potentials.xml";
    //string param_groups         = "ReaDDy_input/param_groups.xml";
    //string tplgy_coordinates    = "ReaDDy_input/tplgy_coordinates.xml";
    //string tplgy_groups         = "ReaDDy_input/tplgy_groups.xml";
    //string param_global         = "ReaDDy_input/param_global.xml";
    //double totalNumberOfFrames = 1000;
    //double stepSizeInPs    = 0.00000001 ;
    //int    stepsPerFrame   = 100 ;
    //double temperature     = 300;              /// kelvin
    double friction        = 1  ;              /// picoseconds
    //double boltzmann       = 0.0083144621;
    //double periodicBoundaries[3] = {100,100,100};
    //double latticeBounds[6] = {-250,250,-250,250,-100,150};
    //double latticeBounds[6] = {-60,60,-60,60,-10,110};
    double maxCutoff       = 10;

    /// /////////////////////////////////

    if(testmode)
        cout << "create simulation..." << endl;

    Simulation simulation = Simulation();

    simulation.testmode=testmode;
    //simulation.totalNumberOfFrames= totalNumberOfFrames;
    //simulation.stepSizeInPs = stepSizeInPs;
    //simulation.stepsPerFrame = stepsPerFrame;
    //simulation.temperature = temperature;
    //simulation.boltzmann = boltzmann;
    simulation.friction = friction;
    //simulation.periodicBoundaries = periodicBoundaries;
    //simulation.latticeBounds = latticeBounds;
    simulation.maxCutoff = maxCutoff;
    simulation.RDFrequired = 0;
    simulation.numberOfRDFBins=0;
    //simulation.cudaDevice = 0;
    bool output = false;

    /// read input files
    if(testmode)
        cout << "reading input files..." << endl;
    simulation.readXML(param_particles, tplgy_potentials, param_groups, tplgy_coordinates, tplgy_groups);
    simulation.readGlobalXML(param_global);
    /// build potentials etc. ...
    simulation.parse();

    testmode=simulation.testmode;
    if(testmode)
        cout <<  simulation.numberParticles << " particles loaded" << endl;
    /// initialize output

    PDBTrajectoryOutputHandler PDBOutputHandler = PDBTrajectoryOutputHandler(&simulation);
    XMLTrajectoryOutputHandler XMLOutputHandler = XMLTrajectoryOutputHandler(&simulation);
    RDFOutputHandler RDFOutput = RDFOutputHandler(&simulation);
    if(output){
        char PDBOutputFile[] = "ReaDDy_output/pdb.pdb";
        if(PDBOutputHandler.openOutputFile(PDBOutputFile)!=0){
            cout << "error while opening PDB output file" << endl;
            return;
        }
        char XMLOutputFile[] = "ReaDDy_output/out_traj.xml";
        if(XMLOutputHandler.openOutputFile(XMLOutputFile)!=0){
            cout << "error while opening XML output file" << endl;
            return;
        }
        if(simulation.RDFrequired>0){
            char RDFOutputFile[] = "ReaDDy_output/out_rdf.csv";
            if(RDFOutput.openOutputFile(RDFOutputFile)!=0){
                cout << "error while opening RDF output file" << endl;
                return;
            }
        }
    }


    /*cout << simulation.totalNumberOfFrames << endl;
    cout << simulation.stepSizeInPs  << endl;
    cout << simulation.stepsPerFrame << endl;
    cout << simulation.temperature  << endl;
    cout << simulation.boltzmann  << endl;
    cout << simulation.friction  << endl;
    cout << simulation.periodicBoundaries << endl;
    cout << simulation.latticeBounds << endl;
    cout << simulation.maxCutoff << endl;
    cout << simulation.RDFrequired << endl;
    cout << simulation.numberOfRDFBins << endl;
    cout << simulation.cudaDevice << endl;*/

    /// obtain callback method form Java
    if(testmode){
        cout << "receive Java callback Method"<< endl;
        cout << "get class" << endl;
    }
    jclass mClass = env->GetObjectClass(obj);
    if(testmode){
        cout << "get method ID" << endl;
    }
    jmethodID mid = env->GetMethodID(mClass, "frameCallback", "(I[F)Z");
    if (mid==0){ cout << "mthod not found" << endl; return;}



    /// initialize Simulation
    if(testmode)
        cout << "initialize Simulation..." << endl;
    CudaSimulation cudaSimulation = CudaSimulation(&simulation);
    simulation.cudaSimulation = &cudaSimulation;
    if(cudaSimulation.initialize()==1){
        cout << "error during initialization" << endl;
        return;
    }


    /// Timer
    timeval startSimulationClock;
    gettimeofday(&startSimulationClock, 0);

    /// call simulation loop
    if(testmode)
        cout << "call simulation loop..." << endl;
    cout << simulation.totalNumberOfFrames << " frames with each " << simulation.stepsPerFrame << " steps" << endl;

    for(int frame=0; frame<simulation.totalNumberOfFrames; frame++){
        //env->PushLocalFrame(512);

/*        if(simulation.cudaSimulation->testNeighborList()!=0){
            cout << "neighborlist broken" << endl;
            return;
        }
*/
        if(testmode)
            cout << "integrate " << simulation.stepsPerFrame<< " steps" << endl;
        /// inner simulation loop:
        for(int i=0; i<simulation.stepsPerFrame; ++i){
            if(simulation.simulate()!=0){
                cerr << "error" << endl;
                return;
            }
        }
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            cout << "cuda error: " << cudaGetErrorString(error ) << endl;
            return;
        }

        //output
        cout << "frame: " << frame << " (step " << frame*simulation.stepsPerFrame << ")" <<  endl;

        if(simulation.cudaSimulation->copyPosFromDevice()!=0){
            cout << "error while copying positions from device" << endl;
            return;
        }

        if(output){
            if(PDBOutputHandler.writeFrame(frame)!=0){
                cout << " error while writing in PDB output file" << endl;
                return;
            }
            if(XMLOutputHandler.writeFrame(frame)!=0){
                cout << " error while writing in XML output file" << endl;
                return;
            }
            if(simulation.RDFrequired>0){
                if(simulation.cudaSimulation->callRDFCalculation()!=0){
                    cout << "error while calculating RDF values" << endl;
                    return;
                }
                if(simulation.cudaSimulation->copyRDFMatrix()!=0){
                    cout << " error while copying RDF values" << endl;
                    return;
                }
            }
        }

        /// return positions to Java/ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////

        /// open Java-Array for C
        jfloatArray JPos = (env)->NewFloatArray((jsize)simulation.numberParticles*5+1);
        jfloat *Pos = (jfloat*)(env)->GetPrimitiveArrayCritical( JPos, NULL);
            if(Pos == NULL){
                cout << "cant get array pointer"<<endl;
                return;
            }

            /// IMPORTANT: There is a distinct ReaDDyParticleId and a OpenMMParticleId.
            /// In every communication between ReaDDy and OpenMM, they have to be converted into
            /// one another. This is what the following code snippet does.
            ///
            /// first value: number of particles, than add [x,y,z,id(OpenMM), id(ReaDDy)] for each particle
            Pos[0]=simulation.numberParticles;
            int currentPos=0;
            /// copy positions to Java-Array
            for (int j = 0; j < simulation.numberParticles; j++){
                if(simulation.types[j]!=-1){
                    for( int i=0; i<3; i++){
                        /// check for errors in particle positions
                        Pos[1+currentPos*5+i]=simulation.coords[j*3+i]; /// coordinates
                    }
                    Pos[1+currentPos*5+3]=j;        /// index
                    Pos[1+currentPos*5+4]=simulation.ReaDDyNumber[j];        /// ReaDDy Particle ID
                    currentPos++;
                }
            }

        /// release Java-Array
        (env)->ReleasePrimitiveArrayCritical( JPos, Pos, 0);

        /// callback Java/ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////

        /// callback and check for error
        if(!(env->CallBooleanMethod(obj, mid, frame, JPos))){
            cout << "error in java"<<endl;
            return;
        }
        (env) -> DeleteLocalRef(JPos);

        /// get reactions form ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////


        /// obtain the reaction array directly from within the java object
        jfieldID fid = env->GetFieldID(mClass, "cReactions", "[F");
        jobject mydata = env->GetObjectField(obj, fid);
        jfloatArray *jReactions = reinterpret_cast<jfloatArray*>(&mydata);
        jfloat *reactions = (jfloat*)(env)->GetPrimitiveArrayCritical( *jReactions, NULL);


        if(testmode){
            cout << reactions[0] << " changes " << endl;
        }
        if(reactions[0]>0){
                /// note: index means in the context of this file "particleId" in OpenMM
                /// reaction-code:  if index==-1 -> create new particle
                ///                 if newType==-1 -> delete particle
                ///                 else -> modify particle (first delete, and than create)
                /// array: [0]number reactions, then 6-tupel: [ID(ReaDDy), type, posX, posY, posZ, index(ID OpenMM)]

                int oldNumberOfParticles = simulation.numberParticles;
                /// vector with new positions (copy old positions)
                std::vector<double> newCoords;
                newCoords.assign(simulation.coords, &simulation.coords[simulation.numberParticles*3]);
                std::vector<int> newTypes;
                newTypes.assign(simulation.types, &simulation.types[simulation.numberParticles]);
                std::vector<int> newIDs;
                newIDs.assign(simulation.ReaDDyNumber, &simulation.ReaDDyNumber[simulation.numberParticles]);
                /*
                    std::vector<double> newCoords(simulation.numberParticles*3, 0);
                    std::vector<int> newTypes(simulation.numberParticles, 0);
                    std::vector<int> newIDs(simulation.numberParticles, 0);
                    for(int i=0; i<simulation.numberParticles; ++i){
                    newCoords[i*3]=simulation.coords[i*3];
                    newCoords[i*3+1]=simulation.coords[i*3+1];
                    newCoords[i*3+1]=simulation.coords[i*3+2];
                    newTypes[i]=simulation.types[i];
                    newIDs[i]=simulation.ReaDDyNumber[i];
                }*/
                int numReactions=reactions[0];
                for(int r=0; r<numReactions; r++){
                    /// new particle properties:
                    int ID = reactions[1+(6*r)];        /// particleID in ReaDDy
                    int newType = reactions[1+(6*r)+1]; /// new particle type
                    //OpenMM::Vec3 newPos = OpenMM::Vec3(reactions[3+(6*r)],reactions[4+(6*r)],reactions[5+(6*r)]);   /// new particle positions
                    int index = reactions[1+6*r+5]; /// particleindex in OpenMM

                    if(index!=-1){
                        /// -> delete particle
                        newTypes[index]=-1;
                        simulation.numberParticles--;
                    }

                    if(newType!=-1){
                        /// -> new particle
                        newCoords.push_back(reactions[3+(6*r)]);
                        newCoords.push_back(reactions[4+(6*r)]);
                        newCoords.push_back(reactions[5+(6*r)]);
                        newTypes.push_back(newType);
                        newIDs.push_back(ID);
                        simulation.numberParticles++;
                    }
                }


                /*int* fields = new int[simulation.numberParticles];
                for(int i=0; i<simulation.numberParticles; ++i){

                    fields[i]=((int)floor((simulation->coords[3*i+2]-simulation.latticeBounds[4])/simulation.maxCutoff)%cudaSimulation.latticeSize[2])*cudaSimulation.latticeSize[0]*cudaSimulation.latticeSize[1]
                             +((int)floor((simulation->coords[3*i+1]-simulation.latticeBounds[2])/simulation.maxCutoff)%cudaSimulation.latticeSize[1])*cudaSimulation.latticeSize[0]
                             +((int)floor((simulation->coords[3*i+0]-simulation.latticeBounds[0])/simulation.maxCutoff)%cudaSimulation.latticeSize[0]);

                }*/



                /*if(simulation.cudaSimulation->copyPosFromDevice()!=0){
                    cout << "error while copying positions from device.." << endl;
                    return;
                }
                for(int i=0; i<oldNumberOfParticles; ++i){
                    if(newTypes[i]!=-1){
                        if( simulation.coords[i*3]!=newCoords[i*3]
                                ||simulation.coords[i*3+1]!=newCoords[i*3+1]
                                ||simulation.coords[i*3+2]!=newCoords[i*3+2]
                                ){
                            cout << "errooooorrr!!" << i << " " << i<< " " <<simulation.coords[i*3]<<" " <<newCoords[i*3] <<endl;
                            return;
                        }
                    }
                }*/


                free(simulation.coords);
                free(simulation.types);
                free(simulation.ReaDDyNumber);

                simulation.coords = new float[simulation.numberParticles*3];
                simulation.types = new int[simulation.numberParticles];
                simulation.ReaDDyNumber = new int[simulation.numberParticles];

                int count=0;
                for(int i=0; i<newTypes.size(); ++i){
                    if(newTypes[i]!=-1){
                        simulation.coords[count*3]=newCoords[i*3];
                        simulation.coords[count*3+1]=newCoords[i*3+1];
                        simulation.coords[count*3+2]=newCoords[i*3+2];
                        simulation.types[count]=newTypes[i];
                        simulation.ReaDDyNumber[count]=newIDs[i];
                        count++;
                    }
                }
                newCoords.clear();
                newTypes.clear();
                newIDs.clear();

                if(testmode)
                    cout << "copyPosToDev" << endl;
                if(cudaSimulation.copyPosToDevice()!=0){
                    cout << "error!" << endl;
                    return;
                }
                if(testmode)
                    cout << "copyTypesToDev" << endl;
                if(cudaSimulation.copyTypesToDevice()!=0){
                    cout << "error!" << endl;
                    return;
                }
                if(testmode)
                    cout << "createNeighborList" << endl;
                if(cudaSimulation.createNeighborList()!=0){
                    cout << "error!" << endl;
                    return;
                }
                if(testmode)
                    cout << "copyNeighborListToDevice" << endl;
                if(cudaSimulation.copyNeighborListToDevice()!=0){
                    cout << "error!" << endl;
                    return;
                }
                /*if(testmode)
                    cout << "testNeighborList" << endl;
                if(cudaSimulation.testNeighborListBigTest()!=0){
                    cout << "error!" << endl;
                     return;
                    }*/
                if(testmode)
                    cout << "allocateForces" << endl;
                if(cudaSimulation.allocateForces()!=0){
                    cout << "error!" << endl;
                    return;
                }
                if(testmode)
                    cout << "resizeCuRand" << endl;
                if(cudaSimulation.resizeCuRand()!=0){
                    cout << "error!" << endl;
                    return;
                }
        }
        (env)->ReleasePrimitiveArrayCritical( *jReactions, reactions, 0);
        (env) -> DeleteLocalRef(mydata);



        /// ///////////////////////////////////////////////////////////////

        //if(testmode){
            cout << "runtime: " << getTime(startSimulationClock) << endl;
            cout << (double)(frame+1)/(double)simulation.totalNumberOfFrames*100 << "%" << " approximate residual runtime: " <<
                    (1.-(double)(frame+1)/(double)simulation.totalNumberOfFrames)*(getTime(startSimulationClock)/((double)(frame+1)/(double)simulation.totalNumberOfFrames))<< endl;
        //}
        //env->PopLocalFrame(NULL);

    }

    if(simulation.cudaSimulation->copyPosFromDevice()!=0){
        cout << "error while copying positions from device" << endl;
        return;
    }
    //output

    if(simulation.RDFrequired>0){
        if(simulation.normalizeRDF()!=0){
            cout << " error while RDF normalization" << endl;
            return;
        }
        if(RDFOutput.writeOutput()!=0){
            cout << " error while writing in RDF output file" << endl;
            return;
        }
        RDFOutput.closeOutputFile();
    }

    if(output){
        PDBOutputHandler.closeOutputFile();
        XMLOutputHandler.closeOutputFile();
    }

    cout << "END, total runtime: "<<getTime(startClock)<<endl;

}
/*
        /// return positions to Java/ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////

        if(testmode){
            printf("return positions\n");
            gettimeofday(&timeX, 0);
        }
        /// open Java-Array for C
        jfloatArray JPos = (env)->NewFloatArray((jsize)NumOfParticles*5+1);
        jfloat *Pos = (jfloat*)(env)->GetPrimitiveArrayCritical( JPos, NULL);
            if(Pos == NULL){
                cout << "cant get array pointer"<<endl;
                return;
            }

            /// IMPORTANT: There is a distinct ReaDDyParticleId and a OpenMMParticleId.
            /// In every communication between ReaDDy and OpenMM, they have to be converted into
            /// one another. This is what the following code snippet does.
            ///
            /// first value: number of particles, than add [x,y,z,id(OpenMM), id(ReaDDy)] for each particle
            Pos[0]=NumOfParticles;
            int currentPos=0;
            /// copy positions to Java-Array
            for (int j = 0; j < (int)posInNm.size(); j++){
                if(partType[j]!=-1){
                    for( int i=0; i<3; i++){
                        /// check for errors in particle positions
                        if(testmode && fabs((float)posInNm[j][i])> 100000){
                            cout << "error at pos "<< j <<"["<<i<<"] = "<<posInNm[j][i] << "(Type: "<<partType[j]<< ")" << endl;
                            fail==true;
                            return;
                        }
                        Pos[1+currentPos*5+i]=(float)posInNm[j][i]; /// coordinates
                    }
                    Pos[1+currentPos*5+3]=j;        /// index
                    Pos[1+currentPos*5+4]=jID[j];   /// ReaDDy Particle ID
                    currentPos++;
                }
            }
            if(fail){
                return;
            }

        /// release Java-Array
        (env)->ReleasePrimitiveArrayCritical( JPos, Pos, 0);

        /// callback Java/ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////

        if(testmode){
            timeCpyPos+=(float)getTime(timeX);
            cout<< "callback" <<endl;
            gettimeofday(&timeX, 0);
        }

        /// callback and check for error
        if(!(env->CallBooleanMethod(obj, mid, frameNumber, JPos))){
            cout << "error in java"<<endl;
            return;
        }
        (env) -> DeleteLocalRef(JPos);

        if(testmode){
            timeCallback+=(float)getTime(timeX);
            printf("back again in native c code\n");
        }

        /// get reactions form ReaDDy ///////////////////////////////////////////////////////////////////////////////////////////////

        if(testmode){
            cout << "Get Reactions" << endl;
            gettimeofday(&timeX, 0);
        }

        /// obtain the reaction array directly from within the java object
        jfieldID fid = env->GetFieldID(mClass, "cReactions", "[F");
        jobject mydata = env->GetObjectField(obj, fid);
        jfloatArray *jReactions = reinterpret_cast<jfloatArray*>(&mydata);
        jfloat *reactions = (jfloat*)(env)->GetPrimitiveArrayCritical( *jReactions, NULL);

        if(testmode){
            cout << reactions[0] << " changes " << endl;
        }
        /// note: index means in the context of this file "particleId" in OpenMM
        /// reaction-code:  if index==-1 -> create new particle
        ///                 if newType==-1 -> delete particle
        ///                 else -> modify particle (first delete, and than create)
        /// array: [0]number reactions, then 6-tupel: [ID(ReaDDy), type, posX, posY, posZ, index(ID OpenMM)]

        /// vector with new positions (copy old positions)
        std::vector<OpenMM::Vec3> posInNmUnlocked(posInNm);
        int numReactions=reactions[0];
        for(int r=0; r<numReactions; r++){
            /// new particle properties:
            int ID = reactions[1+(6*r)];        /// particleID in ReaDDy
            int newType = reactions[1+(6*r)+1]; /// new particle type
            OpenMM::Vec3 newPos = OpenMM::Vec3(reactions[3+(6*r)],reactions[4+(6*r)],reactions[5+(6*r)]);   /// new particle positions
            int index = reactions[1+6*r+5]; /// particleindex in OpenMM

            if(index!=-1){
                /// -> delete particle
                if(testmode)
                    cout <<"deletion: index:" <<index << " type:" << partType[index] << ", ID:" << ID << " pos: " << posInNmUnlocked.at(index)[0] << " " << posInNmUnlocked.at(index)[1] << " " << posInNmUnlocked.at(index)[2] << endl;
                posInNmUnlocked.at(index)=OpenMM::Vec3(rand()%1000+1000,rand()%1000+1000,rand()%1000+1000);
                /// delete from all forces
                /// //////////////////////////////////////////////////////////////////////////////
                for(int force=0; force<customExternalForces.size(); force++){
                    if(affectedParticleTypesCustomExternalForces[force][partType[index]]==1){
                        double *para = new double[ParametersCustomExternalForces[force][0].size()];
                        for(int parameter=0; parameter<ParametersCustomExternalForces[force][0].size(); parameter++){
                            para[parameter]=0;
                        }
                        const std::vector<double> param (para, para + ParametersCustomExternalForces[force][0].size() );
                        customExternalForces[force]->setParticleParameters(termIndices[force][index], index, param);
                    }
                }
                for(int force=0; force<customNonbondForces.size(); force++){
                    double *para = new double[ParametersCustomNonbondForces[force][0].size()];
                    for(int parameter=0; parameter<ParametersCustomNonbondForces[force][0].size(); parameter++){
                        para[parameter]=0;
                    }
                    const std::vector<double> param (para, para + ParametersCustomNonbondForces[force][0].size() );
                    customNonbondForces[force]->setParticleParameters(index,param);
                }

                /// ////////////////////////////////////////////////////////////////////////////////
                /// edit Type, jID, and set respective number of (free) particles
                int oldType = partType[index];
                partType[index]= -1;
                jID[index]=-1;
                NumOfParticles--;
                numberOfFreePositions[oldType]++;
                /// save position(OpenMM ID) from the free slot
                freePositionsInParticleArray[oldType].push_back(index);
            }

            if(newType!=-1){
                /// create a new particle
                NumOfParticles++;
                if(numberOfFreePositions[newType]==0){
                    cerr << "out of dummieparticles! restart with more dummieparticles for type " << newType << endl;
                    return;
                }
                /// use a free slot in dummy particle array
                numberOfFreePositions[newType]--;
                int newIndex=freePositionsInParticleArray[newType].back();
                freePositionsInParticleArray[newType].pop_back();
                index=newIndex;
                if(testmode){
                    cout << "creation: index: " << index << " newType: " << newType << " ID: " << ID <<endl;
                }
                /// Positions
                posInNmUnlocked.at(newIndex)=newPos;
                /// Type
                partType[newIndex]= newType;
                int type = newType;
                jID[newIndex]=ID;
                /// add particle to all necessary forces
                /// //////////////////////////////////////////////////////////////////////////////
                for(int force=0; force<customExternalForces.size(); force++){
                    double *para = new double[ParametersCustomExternalForces[force][0].size()];
                    for(int parameter=0; parameter<ParametersCustomExternalForces[force][0].size(); parameter++){
                        para[parameter]=ParametersCustomExternalForces[force][type][parameter];
                    }
                    if(affectedParticleTypesCustomExternalForces[force][type]==1){
                        const std::vector<double> param (para, para + ParametersCustomExternalForces[force][0].size() );
                        customExternalForces[force]->setParticleParameters(termIndices[force][index], index, param);
                    }
                }
                for(int force=0; force<customNonbondForces.size(); force++){
                    double *para = new double[ParametersCustomNonbondForces[force][0].size()];
                    for(int parameter=0; parameter<ParametersCustomNonbondForces[force][0].size(); parameter++){
                        para[parameter]=ParametersCustomNonbondForces[force][type][parameter];
                    }
                    const std::vector<double> param (para, para + ParametersCustomNonbondForces[force][0].size() );
                    customNonbondForces[force]->setParticleParameters(index, param);
                }

                /// ////////////////////////////////////////////////////////////////////////////////
                if(testmode)
                    cout << "-> " << newIndex << "(type:"<< newType << ", ID:" << ID << "): " << posInNmUnlocked.at(index)[0] << " " << posInNmUnlocked.at(index)[1] << " " << posInNmUnlocked.at(index)[2] <<endl;
            }
        }


        (env)->ReleasePrimitiveArrayCritical( *jReactions, reactions, 0);
*/

