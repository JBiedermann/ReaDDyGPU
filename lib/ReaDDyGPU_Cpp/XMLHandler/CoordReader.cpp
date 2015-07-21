 

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

int CoordReader::readCoords(Simulation *simulation, string tplgyCrd){

    vector<Coord> initPosInNm = vector<Coord>();
    vector<int> partType = vector<int>();
    vector<int> reaDDyNumber = vector<int>();

    string line;
    ifstream myfile (tplgyCrd.c_str());
    if (myfile.is_open()){
        while ( myfile.good() ){
            getline (myfile,line);
            if(line[1]=='p'){

                /// example line + parsing type and positions
                //<p id="0" type="1" c="-31.09250630276887,-42.34489538041901,34.654154873058644"/>

                int id=atoi(split(line,"\"")[1].c_str());
                reaDDyNumber.push_back(id);

                /// read and store type of current particle
                int type=atoi(split(line,"\"")[3].c_str());
                partType.push_back(type);

                /// read and store position of current particle
                string pos=split(line,"\"")[5];
                initPosInNm.push_back( Coord(string_to_double(split(pos,",")[0].c_str()),string_to_double(split(pos,",")[1].c_str()),string_to_double(split(pos,",")[2].c_str()) ) ); // location, nm
                //const double position[3] = {string_to_double(split(pos,",")[0].c_str()),string_to_double(split(pos,",")[1].c_str()),string_to_double(split(pos,",")[2].c_str()) };
                //initPosInNm.push_back( *position ); // location, nm
                //initPosInNm.push_back( new double[3] ); // location, nm
                //initPosInNm.back()[0] = string_to_double(split(pos,",")[0].c_str());
                //initPosInNm.back()[1] = string_to_double(split(pos,",")[1].c_str());
                //initPosInNm.back()[2] = string_to_double(split(pos,",")[2].c_str());
            }
        }
        myfile.close();
    }

    simulation->coords = new float[initPosInNm.size()*3];
    simulation->types = new int[initPosInNm.size()];
    simulation->ReaDDyNumber = new int[initPosInNm.size()];
    simulation->numberParticles=initPosInNm.size();
    for(int i=0; i<initPosInNm.size(); ++i){
        simulation->coords[i*3] = initPosInNm[i].x;
        simulation->coords[i*3+1] = initPosInNm[i].y;
        simulation->coords[i*3+2] = initPosInNm[i].z;
        simulation->types[i] = partType[i];
        simulation->ReaDDyNumber[i] = reaDDyNumber[i];
    }

}
