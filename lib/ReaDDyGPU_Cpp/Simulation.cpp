
#include <ReaDDyGPU.hpp>
#include <string>
#include <vector>
#include <AuxillaryFunctions.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLString.hpp>
#include <typeinfo>
#include <stdexcept>      // std::out_of_range

using namespace std;


Simulation::Simulation(){
    particleTypes = vector<ParticleType>();
    orderOnePotentials = vector<OrderOnePotential*>();
    //groupPotentials = vector<GroupPotential*>();
    orderTwoPotentials = vector<OrderTwoPotential*>();
    plainPotentials=vector<map<string, string> >();
    groups=vector<Group>();
    numberOfRDFFrames=0;
}

Group::Group(){
    buildingBlocks = vector<BuildingBlock>();
    //potentials = vector<GroupPotential>();
    potentials = vector<OrderTwoPotential*>();
    plainPotentials = vector<map<string, string> >();
    individualGroups = vector< vector <int> >();
}

Coord::Coord(double x,double y, double z){
    this->x=x;
    this->y=y;
    this->z=z;
}

int Simulation::normalizeRDF(){
    for(int i=0; i<particleTypes.size(); ++i){
        for(int j=0; j <particleTypes.size(); ++j){
            //RDFMatrix[i*particleTypes.size()*numberOfRDFBins+j] += RDFMatrix[j*particleTypes.size()+i]/numberOfRDFFrames;
            //RDFMatrix[j*particleTypes.size()+i] += RDFMatrix[i*particleTypes.size()+j];
            for(int bin=0; bin<numberOfRDFBins; ++bin){
                double pi = 3.14159265359;
                double r = (float)bin/200.0f*maxCutoff;
                /// normalize 2D Disk
                // double diskRadius = ;
                //RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+r] = RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+r]/((pi*(r+1)*(r+1))-(pi*(r)*(r)))*(pi*diskRadius*diskRadius);
                /// normalize 3D Cube
                double binVolume = (4/3*pi*(r+1)*(r+1)*(r+1))-(4/3*pi*r*r*r);
                double boxVolume = 100*100*100;
                RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+bin] = RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+bin]/binVolume*boxVolume;

                /// normalize over #timesteps
                RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+bin] = RDFMatrix[j*particleTypes.size()*numberOfRDFBins+i*numberOfRDFBins+bin]/(double)numberOfRDFFrames;
            }
        }
    }
    return 0;
}

/// catch "all"
vector<int> parseAffectedIds(string inputString , Simulation* simulation){
    if(inputString.compare("all")==0){
        vector<int> myVector;
        for(int i=0; i<simulation->particleTypes.size(); ++i){
            myVector.push_back(i);
        }
        return myVector;
    }
    else{
        return stringToIntVector(inputString);
    }
}
vector<int> parse2AffectedIds(string inputString , Simulation* simulation){
    if(inputString.compare("all")==0){
        vector<int> myVector;
        for(int i=0; i<simulation->particleTypes.size(); ++i){
            for(int j=0; j<simulation->particleTypes.size(); ++j){
                myVector.push_back(i);
                myVector.push_back(j);
            }
        }
        return myVector;
    }
    else{
        return stringToIntIntVector(inputString);
    }
}

/// TODO!!!: do this with virtual function of the basefunction "potential" and define parsing within an own sublass-file

DiskPotential* parseDiskPotential(map<string, string> plainPotential, Simulation * simulation){
    DiskPotential* potential = new DiskPotential();
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->typeID = 1;
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    potential->radius = atof(plainPotential["radius"].c_str());
    //diskPotential.center = new double[3];
    CoordStringToDoubleArray(plainPotential["center"], potential->center);
    //diskPotential.normal = new double[3];
    CoordStringToDoubleArray(plainPotential["normal"], potential->normal);
    potential->affectedParticleTypeIds = parseAffectedIds(plainPotential["affectedParticleTypeIds"], simulation);
    potential->affectedParticleIds = parseAffectedIds(plainPotential["affectedParticleIds"], simulation);
    return potential;
}
CylinderPotential* parseCylinderPotential(map<string, string> plainPotential, Simulation* simulation){
    CylinderPotential* potential = new CylinderPotential();
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->typeID = 2;
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    potential->height = atof(plainPotential["height"].c_str());
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    potential->radius = atof(plainPotential["radius"].c_str());
    //cylinderPotential->center = new double[3];
    CoordStringToDoubleArray(plainPotential["center"], potential->center);
    //cylinderPotential->normal = new double[3];
    CoordStringToDoubleArray(plainPotential["normal"], potential->normal);
    potential->affectedParticleTypeIds = parseAffectedIds(plainPotential["affectedParticleTypeIds"], simulation);
    potential->affectedParticleIds = parseAffectedIds(plainPotential["affectedParticleIds"], simulation);
    return potential;
}
SpherePotential* parseSpherePotential(map<string, string> plainPotential, Simulation* simulation){
    SpherePotential* potential = new SpherePotential();
    potential->typeID =3;
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    CoordStringToDoubleArray(plainPotential["center"], potential->center);
    potential->radius = atof(plainPotential["radius"].c_str());
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    potential->affectedParticleTypeIds = parseAffectedIds(plainPotential["affectedParticleTypeIds"], simulation);
    potential->affectedParticleIds = parseAffectedIds(plainPotential["affectedParticleIds"], simulation);
    return potential;
}
BoxPotential* parseBoxPotential(map<string, string> plainPotential, Simulation* simulation){
    BoxPotential* potential = new BoxPotential();
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->typeID = 4;
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    CoordStringToDoubleArray(plainPotential["origin"], potential->origin);
    CoordStringToDoubleArray(plainPotential["extension"], potential->extension);
    potential->affectedParticleTypeIds = parseAffectedIds(plainPotential["affectedParticleTypeIds"], simulation);
    potential->affectedParticleIds = parseAffectedIds(plainPotential["affectedParticleIds"], simulation);
    return potential;
}

HarmonicPotential* parseHarmonicPotential(map<string, string> plainPotential, Simulation* simulation){
    HarmonicPotential * potential = new HarmonicPotential();
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->typeID = 1;
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    if(potential->subtype.compare("spring")==0)
        potential->subtypeID = 3;
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    potential->affectedParticleTypeIdPairs=parse2AffectedIds(plainPotential["affectedParticleTypeIdPairs"], simulation);
    potential->affectedParticleIdPairs=parse2AffectedIds(plainPotential["affectedParticleIdPairs"], simulation);
    return potential;
}
WeakInteractionHarmonicPotential* parseWeakInteractionHarmonicPotential(map<string, string> plainPotential, Simulation* simulation){
    WeakInteractionHarmonicPotential * potential = new WeakInteractionHarmonicPotential();
    potential->name = plainPotential["name"];
    potential->type = plainPotential["type"];
    potential->typeID = 2;
    potential->subtype = plainPotential["subtype"];
    if(potential->subtype.compare("attractive")==0)
        potential->subtypeID = 1;
    if(potential->subtype.compare("repulsive")==0)
        potential->subtypeID = 2;
    potential->forceConst = atof(plainPotential["forceConst"].c_str());
    potential->length = atof(plainPotential["length"].c_str());
    potential->depth = atof(plainPotential["depth"].c_str());
    potential->affectedParticleTypeIdPairs=parse2AffectedIds(plainPotential["affectedParticleTypeIdPairs"], simulation);
    potential->affectedParticleIdPairs=parse2AffectedIds(plainPotential["affectedParticleIdPairs"], simulation);
    return potential;
}

int Simulation::parse(){

    /// parse particle type collision radii matrices
    for(int i=0; i<particleTypes.size(); ++i){
        particleTypes[i].radiiMatrix = new double[particleTypes.size()];
        for(int j=0; j<particleTypes.size(); ++j){
            particleTypes[i].radiiMatrix[j]=0;
        }
        for(int j=0; j<particleTypes[i].collisionRadiiMap.size(); ++j){
            typedef map<string, double>::iterator MapIterator;
            for(MapIterator iterator = particleTypes[i].collisionRadiiMap.begin(); iterator != particleTypes[i].collisionRadiiMap.end(); ++iterator) {
                if(iterator->first.compare("default")==0){
                    particleTypes[i].defaultRadius = iterator->second;
                }
                else if(iterator->first.compare("all")==0){
                    particleTypes[i].defaultRadius = iterator->second;
                }
                else{
                    particleTypes[i].radiiMatrix[atoi(iterator->first.c_str())] = iterator->second;
                }
            }
        }
    }

    /// parse potentials
    for(int i=0; i<plainPotentials.size(); ++i){
        /// order one potentials
        if(plainPotentials[i]["type"].compare("DISK")==0){
            orderOnePotentials.push_back(parseDiskPotential(plainPotentials[i], this));
        }
        else if(plainPotentials[i]["type"].compare("CYLINDER")==0){
            orderOnePotentials.push_back(parseCylinderPotential(plainPotentials[i], this));
        }
        else if(plainPotentials[i]["type"].compare("SPHERE")==0){
            orderOnePotentials.push_back(parseSpherePotential(plainPotentials[i], this));
        }
        else if(plainPotentials[i]["type"].compare("BOX")==0 || plainPotentials[i]["type"].compare("CUBE")==0){
            orderOnePotentials.push_back(parseBoxPotential(plainPotentials[i], this));
        }
        /// order two potentials
        else if(plainPotentials[i]["type"].compare("HARMONIC")==0){
            orderTwoPotentials.push_back(parseHarmonicPotential(plainPotentials[i], this));
        }
        else if(plainPotentials[i]["type"].compare("WEAK_INTERACTION_HARMONIC")==0){
            orderTwoPotentials.push_back(parseWeakInteractionHarmonicPotential(plainPotentials[i], this));
        }
    }
    /// parse group potentials
    for(int g=0; g<groups.size(); ++g){
        for(int i=0; i<groups[g].plainPotentials.size(); ++i){
            /// order two potentials
            if(groups[g].plainPotentials[i]["type"].compare("HARMONIC")==0){
                groups[g].potentials.push_back(parseHarmonicPotential(groups[g].plainPotentials[i], this));
            }
            else if(groups[g].plainPotentials[i]["type"].compare("WEAK_INTERACTION_HARMONIC")==0){
                groups[g].potentials.push_back(parseWeakInteractionHarmonicPotential(groups[g].plainPotentials[i], this));
            }
        }
    }

    RDFMatrix = new double[particleTypes.size()*particleTypes.size()*numberOfRDFBins];

    numberOfParticlesPerType = vector<int>(particleTypes.size(), 0);
    for(int i=0; i<numberParticles; ++i){
        ++numberOfParticlesPerType[types[i]];
    }

    /*cout << "particle count per type: " << endl;
    for(int i=0; i<particleTypes.size(); ++i){
        cout << numberOfParticlesPerType[i] << " ";
    }
    cout << endl;*/

    return 0;
}

int Simulation::readXML(string param_particles, string tplgy_potentials, string param_groups, string tplgy_coordinates, string tplgy_groups){
    XMLReader xmlReader = XMLReader();

    if(testmode)
        cout << "read particle parameter"<< endl;
    char *cstr = new char[param_particles.length() + 1];
    strcpy(cstr, param_particles.c_str());
    xmlReader.readParticleParameters(this, cstr);

    if(testmode)
        cout << "read potential topology"<< endl;
    cstr = new char[tplgy_potentials.length() + 1];
    strcpy(cstr, tplgy_potentials.c_str());
    xmlReader.readPotentialParameters(this, cstr);

    if(testmode)
        cout << "read group parameter"<< endl;
    cstr = new char[param_groups.length() + 1];
    strcpy(cstr, param_groups.c_str());
    xmlReader.readGroupParameters(this, cstr);

    if(testmode)
        cout << "read coordinate topology" << endl;
    CoordReader coordReader = CoordReader();
    coordReader.readCoords(this, tplgy_coordinates);

    if(testmode)
        cout << "read group topology" << endl;
    GroupReader groupReader = GroupReader();
    groupReader.readGroups(this, tplgy_groups);

    if(testmode){
        cout << "groups: \t\t" << groups.size() << endl;
        cout << "particle types: \t" <<  particleTypes.size() << endl;
        cout << "potentials: \t\t" << plainPotentials.size() << endl;
    }

    if(testmode){
        cout <<"number of particles: " << numberParticles << endl;
    }

    if(testmode && groups.size()>=1){
        cout <<"number of individual groups[0]: " << groups[0].individualGroups.size() << endl;
    }

    return 0;
}
int Simulation::readGlobalXML(string param_global){
    XMLReader xmlReader = XMLReader();

    if(testmode)
        cout << "read global topology" << endl;
    char* cstr = new char[param_global.length() + 1];
    strcpy(cstr, param_global.c_str());
    xmlReader.readGlobalParameters(this, cstr);
    return 0;
}


int Simulation::simulate(){
    if(cudaSimulation->simulate()!=0){
        return 1;
    }
    return 0;
}

int Simulation::simulate(int x){
    for(int i=0; i<x; ++i){
        if(cudaSimulation->simulate()!=0){
            return 1;
        }
    }
    return 0;
}
