 

class OutputHandler{
public:
    int openOutputFile(char *){return 1;};
    int closeOutputFile(){return 1;};
    int writeFrame(int){return 1;};

    char * fileName;
    Simulation * simulation;
    FILE * myOfile;
};

class PDBTrajectoryOutputHandler : public OutputHandler
{
public:
    PDBTrajectoryOutputHandler(Simulation *);
    int openOutputFile(char *);
    int closeOutputFile();
    int writeFrame(int);
};

class XMLTrajectoryOutputHandler : public OutputHandler
{
public:
    XMLTrajectoryOutputHandler(Simulation *);
    int openOutputFile(char *);
    int closeOutputFile();
    int writeFrame(int);
};

class RDFOutputHandler : public OutputHandler
{
public:
    RDFOutputHandler(Simulation *);
    int openOutputFile(char *);
    int closeOutputFile();
    int writeFrame(int);
    int writeOutput();
};
