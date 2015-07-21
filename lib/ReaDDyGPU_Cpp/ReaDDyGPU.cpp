
#include <ReaDDyGPU.hpp>
#include <iostream>
#include <string>
#include <vector>
# include <cuda.h>
#include <typeinfo>
#include <time.h>
#include <sys/time.h>

using namespace std;


int main (int argc, char* args[]) {

    /// Timer
    timeval startClock;
    gettimeofday(&startClock, 0);

    /// get from ReaDDy: ////////////////

    bool testmode=true;
    string param_particles      = "ReaDDy_input/param_particles.xml";
    string tplgy_potentials     = "ReaDDy_input/tplgy_potentials.xml";
    string param_groups         = "ReaDDy_input/param_groups.xml";
    string tplgy_coordinates    = "ReaDDy_input/tplgy_coordinates.xml";
    string tplgy_groups         = "ReaDDy_input/tplgy_groups.xml";
    string param_global         = "ReaDDy_input/param_global.xml";
    double totalNumberOfFrames = 1000;
    double stepSizeInPs    = 0.00000001 ;
    int    stepsPerFrame   = 100 ;
    double temperature     = 300;              /// kelvin
    double friction        = 1  ;              /// picoseconds
    double boltzmann       = 0.0083144621;
    double periodicBoundaries[3] = {100,100,100};
    //double latticeBounds[6] = {-250,250,-250,250,-100,150};
    double latticeBounds[6] = {-60,60,-60,60,-10,110};
    double maxCutoff       = 10;

    /// /////////////////////////////////

    Simulation simulation = Simulation();

    simulation.testmode=testmode;
    simulation.totalNumberOfFrames= totalNumberOfFrames;
    simulation.stepSizeInPs = stepSizeInPs;
    simulation.stepsPerFrame = stepsPerFrame;
    simulation.temperature = temperature;
    simulation.boltzmann = boltzmann;
    simulation.friction = friction;
    simulation.periodicBoundaries = periodicBoundaries;
    simulation.latticeBounds = latticeBounds;
    simulation.maxCutoff = maxCutoff;
    simulation.RDFrequired = 1;
    simulation.numberOfRDFBins=200;
    simulation.cudaDevice = 0;

    /// read input files
    if(testmode)
        cout << "reading input files..." << endl;
    simulation.readXML(param_particles, tplgy_potentials, param_groups, tplgy_coordinates, tplgy_groups);
    simulation.readGlobalXML(param_global);
    /// build potentials etc. ...
    simulation.parse();

    /// initialize output

    PDBTrajectoryOutputHandler PDBOutputHandler = PDBTrajectoryOutputHandler(&simulation);
    char PDBOutputFile[] = "ReaDDy_output/pdb.pdb";
    if(PDBOutputHandler.openOutputFile(PDBOutputFile)!=0){
        cout << "error while opening PDB output file" << endl;
        return 1;
    }
    XMLTrajectoryOutputHandler XMLOutputHandler = XMLTrajectoryOutputHandler(&simulation);
    char XMLOutputFile[] = "ReaDDy_output/out_traj.xml";
    if(XMLOutputHandler.openOutputFile(XMLOutputFile)!=0){
        cout << "error while opening XML output file" << endl;
        return 1;
    }
    RDFOutputHandler RDFOutput = RDFOutputHandler(&simulation);
    if(simulation.RDFrequired>0){
        char RDFOutputFile[] = "ReaDDy_output/out_rdf.csv";
        if(RDFOutput.openOutputFile(RDFOutputFile)!=0){
            cout << "error while opening RDF output file" << endl;
            return 1;
        }
    }

    if(argc > 1){
        simulation.cudaDevice = atoi(args[1]);
    }

    /// initialize Simulation
    if(testmode)
        cout << "initialize Simulation..." << endl;
    CudaSimulation cudaSimulation = CudaSimulation(&simulation);
    simulation.cudaSimulation = &cudaSimulation;
    if(cudaSimulation.initialize()==1){
        cout << "error during initialization" << endl;
        return 1;
    }

    /// Timer
    timeval startSimulationClock;
    gettimeofday(&startSimulationClock, 0);

    /// call simulation loop
    if(testmode)
        cout << "call simulation loop..." << endl;
    cout << simulation.totalNumberOfFrames << " frames with each " << simulation.stepsPerFrame << " steps" << endl;

    for(int frame=0; frame<simulation.totalNumberOfFrames; frame++){

        //output
        cout << "frame: " << frame << " (step " << frame*simulation.stepsPerFrame << ")" <<  endl;
        if(simulation.cudaSimulation->copyPosFromDevice()!=0){
            cout << "error while copying positions from device" << endl;
            return 1;
        }
        //cout << "position of particle one: " << simulation.coords[0] << " " << simulation.coords[1]<< " " << simulation.coords[2] << endl;
        if(PDBOutputHandler.writeFrame(frame)!=0){
            cout << " error while writing in PDB output file" << endl;
            return 1;
        }
        if(XMLOutputHandler.writeFrame(frame)!=0){
            cout << " error while writing in XML output file" << endl;
            return 1;
        }
        if(simulation.RDFrequired>0){
            if(simulation.cudaSimulation->callRDFCalculation()!=0){
                cout << "error while calculating RDF values" << endl;
                return 1;
            }
            if(simulation.cudaSimulation->copyRDFMatrix()!=0){
                cout << " error while copying RDF values" << endl;
                return 1;
            }
        }

/*        if(simulation.cudaSimulation->testNeighborList()!=0){
            cout << "neighborlist broken" << endl;
            return 1;
        }
*/
        /// inner simulation loop:
        for(int i=0; i<simulation.stepsPerFrame; ++i){
            if(simulation.simulate()!=0){
                cerr << "error" << endl;
                return 1;
            }
        }
        cudaError_t error = cudaGetLastError();
        if ( cudaSuccess != error ){
            cout << "cuda error: " << cudaGetErrorString(error ) << endl;
            return 1;
        }

        if(testmode){
            cout << "runtime: " << getTime(startSimulationClock) << endl;
            cout << (double)(frame+1)/(double)simulation.totalNumberOfFrames*100 << "%" << " approximate residual runtime: " <<
                    (1.-(double)(frame+1)/(double)simulation.totalNumberOfFrames)*(getTime(startSimulationClock)/((double)(frame+1)/(double)simulation.totalNumberOfFrames))<< endl;
        }

    }

    if(simulation.cudaSimulation->copyPosFromDevice()!=0){
        cout << "error while copying positions from device" << endl;
        return 1;
    }
    //output

    if(simulation.RDFrequired>0){
        if(simulation.normalizeRDF()!=0){
            cout << " error while RDF normalization" << endl;
            return 1;
        }
        if(RDFOutput.writeOutput()!=0){
            cout << " error while writing in RDF output file" << endl;
            return 1;
        }
        RDFOutput.closeOutputFile();
    }

    PDBOutputHandler.closeOutputFile();
    XMLOutputHandler.closeOutputFile();

    cout << "END, total runtime: "<<getTime(startClock)<<endl;

    return 0;
}
