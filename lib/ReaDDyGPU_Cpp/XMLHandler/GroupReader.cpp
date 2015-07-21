#include "ReaDDyGPU.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <stdio.h>

using namespace std;

int GroupReader::readGroups(Simulation *simulation, string tplgyGrp){

    string line;

    ifstream myfile2 (tplgyGrp.c_str());
    if (myfile2.is_open())
    {
      while ( myfile2.good() )
        {
        getline (myfile2,line);
        // <g id="0" type="0" internalAndParticleId="[[0,0];[1,1]]"/>
        if(line[1]=='g'){
            string typeS = split(line, "\"")[3];
            string oneS = split(split( split(line, "[")[2], ",")[1], "]")[0];
            string twoS = split(split( split(line, "[")[3], ",")[1], "]")[0];
            int grType = atoi(typeS.c_str());
            int one = atoi(oneS.c_str());
            int two = atoi(twoS.c_str());
            vector < int > individualGroup;
            individualGroup.push_back(one);
            individualGroup.push_back(two);
            simulation->groups[grType].individualGroups.push_back(individualGroup);
        }
      }
      myfile2.close();
    }

}
