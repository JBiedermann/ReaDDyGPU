

#include "ReaDDyGPU.hpp"
#include <string>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <stdio.h>

using namespace std;

RDFOutputHandler::RDFOutputHandler(Simulation* simulation){
    this->simulation=simulation;
}
int RDFOutputHandler::openOutputFile(char * fileName){
    this->fileName = fileName;
    myOfile =fopen(fileName, "w");
    if (myOfile==NULL)
        return 1;

    if(fprintf(myOfile, "d")<0)
        return 1;
    for(int i=0; i<simulation->particleTypes.size(); ++i){
        for(int j=0; j<simulation->particleTypes.size(); ++j){
            if(fprintf(myOfile, ",\tRDF(%s,%s)", simulation->particleTypes[i].name.c_str(), simulation->particleTypes[j].name.c_str())<0)
                return 1;
        }
    }
    if(fprintf(myOfile, "\n")<0)
        return 1;

    return 0;
}

int RDFOutputHandler::writeOutput()
{
    /// wirte average RDF
    int numberOfParticleTypes = simulation->particleTypes.size();
    for (int numBin = 0; numBin < simulation->numberOfRDFBins; ++numBin){
        if(fprintf(myOfile, "%8.5f", ((double)numBin*(double)simulation->maxCutoff/(double)simulation->numberOfRDFBins) )<0)
            return 1;
        for(int i=0; i<simulation->particleTypes.size(); ++i){
            for(int j=0; j<simulation->particleTypes.size(); ++j){
                if(fprintf(myOfile, ", %8.3f", simulation->RDFMatrix[i*numberOfParticleTypes*simulation->numberOfRDFBins+j*simulation->numberOfRDFBins+numBin])<0)
                    return 1;
            }
        }
        if(fprintf(myOfile, "\n")<0)
            return 1;
    }
    return 0;
}

int RDFOutputHandler::writeFrame(int frame)
{
    // write the rdf for just one frame
    // BINS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    /*for (int a = 0; a < simulation->; ++a)
    {
        if(fprintf(myOfile, "%8.3f%8.3f%8.3f 15.00  0.00\n", simulation->coords[a*3+0], simulation->coords[a*3+1], simulation->coords[a*3+2])<0)
            return 1;
    }*/
    return 0;
}



int RDFOutputHandler::closeOutputFile(){
    return fclose(myOfile);
}

PDBTrajectoryOutputHandler::PDBTrajectoryOutputHandler(Simulation * simulation){
    this->simulation=simulation;
}
int PDBTrajectoryOutputHandler::openOutputFile(char * fileName){
    this->fileName = fileName;
    myOfile =fopen(fileName, "w");
    if (myOfile==NULL){
        //cout << "(myOfile==NULL)"<< endl;
        return 1;
    }
    return 0;
}

int PDBTrajectoryOutputHandler::closeOutputFile(){
    return fclose(myOfile);
}

// Handy homebrew PDB writer for quick-and-dirty trajectory output., from OpenMM Example
int PDBTrajectoryOutputHandler::writeFrame( int frame)
{

    // Use PDB MODEL cards to number trajectory frames
    if(fprintf(myOfile, "MODEL     %d\n", frame)<0) // start of frame
        return 1;
    for (int a = 0; a < simulation->numberParticles; ++a)
    {
        if(fprintf(myOfile, "ATOM  %5d   %i    %i     1    ", a+1, simulation->types[a], simulation->types[a] )<0)
            return 1;
        if(fprintf(myOfile, "%8.3f%8.3f%8.3f 15.00  0.00\n", simulation->coords[a*3+0], simulation->coords[a*3+1], simulation->coords[a*3+2])<0)
            return 1;
    }
    if(fprintf(myOfile, "ENDMDL\n")<0)// end of frame
        return 1;
    return 0;
}





XMLTrajectoryOutputHandler::XMLTrajectoryOutputHandler(Simulation * simulation){
    this->simulation=simulation;
}
int XMLTrajectoryOutputHandler::openOutputFile(char * fileName){
    this->fileName = fileName;
    myOfile =fopen(fileName, "w");
    if (myOfile==NULL)
        return 1;
    if(fprintf(myOfile, "<traj>\n")<0)
        return 1;
    return 0;
}

int XMLTrajectoryOutputHandler::closeOutputFile(){
    if(fprintf(myOfile, "<\traj>\n")<0)
        return 1;
    return fclose(myOfile);
}
// Handy homebrew PDB writer for quick-and-dirty trajectory output., from OpenMM Example
int XMLTrajectoryOutputHandler::writeFrame( int frame)
{
    // Use PDB MODEL cards to number trajectory frames
    if(fprintf(myOfile, "<tplgy_coords version=\"1.1\" stepId=\"%d\">\n", frame)<0) // start of frame
        return 1;
    for (int a = 0; a < simulation->numberParticles; ++a)
    {
          if(fprintf(myOfile, "<p id=\"%d\" type=\"%d\" c=\"%58.3f,%58.3f,%58.3f\"/>\n", a+1, simulation->types[a], simulation->coords[a*3+0], simulation->coords[a*3+1], simulation->coords[a*3+2])<0)
            return 1;
    }
    if(fprintf(myOfile, "</tplgy_coords>\n")<0)// end of frame
        return 1;
    return 0;
}
