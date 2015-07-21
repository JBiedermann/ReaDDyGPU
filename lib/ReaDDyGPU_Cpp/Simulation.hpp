
#include <string>
#include <vector>
#include <sstream>
#include <map>


using namespace std;

class CudaSimulation;

struct Coord{
public:
    double x;
    double y;
    double z;
    Coord(double,double,double);
};

// the Particle Class

class ParticleType{
public:
    string name;
    int ID;
    double D;
    int numberOfDummyParticles;
    double * radiiMatrix;
    double defaultRadius;
    map<string, double> collisionRadiiMap;
};

// Potentials

class Potential{
public:
    string name;
    string type;
    int typeID;
    string subtype;
    int subtypeID;
};

class CudaOrderOnePotential;

class OrderOnePotential : public Potential{
public:
    vector<int> affectedParticleTypeIds;
    vector<int> affectedParticleIds;
};
class DiskPotential : public OrderOnePotential{     /// 1
public:
    double forceConst;
    double center[3];
    double normal[3];
    double radius;
};
class CylinderPotential : public OrderOnePotential{ /// 2
public:
    double forceConst;
    double center[3];
    double normal[3];
    double radius;
    double height;
};
class SpherePotential : public OrderOnePotential{   /// 3
public:
    double forceConst;
    double center[3];
    double radius;
};
class BoxPotential : public OrderOnePotential{     /// 4
public:
    double forceConst;
    double origin[3];
    double extension[3];
};

class OrderTwoPotential : public Potential{
public:
    vector<int> affectedParticleTypeIdPairs;
    vector<int> affectedParticleIdPairs;
};
class HarmonicPotential : public OrderTwoPotential{
public:
    double forceConst;
};
class WeakInteractionHarmonicPotential : public OrderTwoPotential{
public:
    double forceConst;
    double length;
    double depth;
};
class WeakInteractionPiecewiseHarmonicPotential : public OrderTwoPotential{
public:
};
/*
class GroupPotential : public Potential{
public:
    string name;
    string type;
    string subtype;
    double forceConstant;
    string affectedInternalIdPairsString;
    vector<int[2]>affectedInternalIdPairs;
};
*/
// For Groups

class BuildingBlock{
public:
    int internalID;
    string type;
    double templateCoord[3];
};

class Group{
public:
    int ID;
    string type;
    double templateOrigin[3];
    double templateNormal[3];
    string templateOriginString;
    string templateNormalString;
    vector<BuildingBlock> buildingBlocks;
    //vector<GroupPotential> potentials;
    vector<OrderTwoPotential*> potentials;
    int maxNumberOfGroupAssignmentsPerParticle;
    //vector<map<string, string> > plainPotentials;
    vector< vector< int > > individualGroups;
    vector<map<string, string> > plainPotentials;

    Group();
};

// The Simulation class

class Simulation{
public:
    Simulation();

    float* coords;
    int numberParticles;
    int* types;
    int* ReaDDyNumber;

    string name;
    vector<ParticleType> particleTypes;
    vector<int>numberOfParticlesPerType;
    vector<map<string, string> > plainPotentials;
    vector<OrderOnePotential*> orderOnePotentials;
    vector<OrderTwoPotential*> orderTwoPotentials;
    vector<Group> groups;
    //vector<GroupPotential*> groupPotentials;
    bool testmode;
    int totalNumberOfFrames;
    double stepSizeInPs;
    double stepSizeInPs_reactions;
    int stepsPerFrame;
    double  temperature;
    double boltzmann;
    float  friction;
    double* periodicBoundaries;
    double* latticeBounds;
    double maxCutoff;
    int RDFrequired;
    double* RDFMatrix;
    int numberOfRDFFrames;
    int numberOfRDFBins;
    int cudaDevice;

    int readXML(string param_particles, string tplgy_potentials, string param_groups, string tplgy_coordinates, string tplgy_groups);
    int readGlobalXML(string);
    int parse();
    int simulate();
    int simulate(int x);
    int normalizeRDF();

    CudaSimulation *cudaSimulation;

};
