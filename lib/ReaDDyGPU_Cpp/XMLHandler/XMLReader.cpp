
#include "ReaDDyGPU.hpp"
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLString.hpp>
#include <string>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <map>
using namespace std;
using namespace xercesc;


int XMLReader::readGlobalParameters(Simulation* simulation, char* fileName){

    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Error during initialization! :\n";
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return 1;
    }

    char* xmlFile = fileName;
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
    //parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
    //parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);   // optional

    GlobalParametersHandler* handler = new GlobalParametersHandler(simulation);
    parser->setContentHandler(handler);
    parser->setErrorHandler(handler);

    try {
        parser->parse(xmlFile);
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (const SAXParseException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (...) {
        cout << "Unexpected Exception \n" ;
        return -1;
    }

    delete parser;
    delete handler;

    simulation->stepsPerFrame=round(simulation->stepSizeInPs_reactions/simulation->stepSizeInPs);

    return 0;
}


int XMLReader::readPotentialParameters(Simulation* simulation, char* fileName){

    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Error during initialization! :\n";
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return 1;
    }

    char* xmlFile = fileName;
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
    //parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
    //parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);   // optional

    PotentialParametersHandler* handler = new PotentialParametersHandler(simulation);
    parser->setContentHandler(handler);
    parser->setErrorHandler(handler);

    try {
        parser->parse(xmlFile);
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (const SAXParseException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (...) {
        cout << "Unexpected Exception \n" ;
        return -1;
    }

    delete parser;
    delete handler;
    return 0;
}


int XMLReader::readParticleParameters(Simulation *simulation, char* fileName){

    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Error during initialization! :\n";
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return 1;
    }

    char* xmlFile = fileName;
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
    //parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
    //parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);   // optional

    ParticleParametersHandler* handler = new ParticleParametersHandler(simulation);
    parser->setContentHandler(handler);
    parser->setErrorHandler(handler);

    try {
        parser->parse(xmlFile);
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (const SAXParseException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (...) {
        cout << "Unexpected Exception \n" ;
        return -1;
    }

    delete parser;
    delete handler;
    return 0;
}

int XMLReader::readGroupParameters(Simulation *simulation, char* fileName){

    try {
        XMLPlatformUtils::Initialize();
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Error during initialization! :\n";
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return 1;
    }

    char* xmlFile = fileName;
    SAX2XMLReader* parser = XMLReaderFactory::createXMLReader();
    //parser->setFeature(XMLUni::fgSAX2CoreValidation, true);
    //parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, true);   // optional

    GroupParametersHandler* handler = new GroupParametersHandler(simulation);
    parser->setContentHandler(handler);
    parser->setErrorHandler(handler);

    try {
        parser->parse(xmlFile);
    }
    catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (const SAXParseException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        cout << "Exception message is: \n"
             << message << "\n";
        XMLString::release(&message);
        return -1;
    }
    catch (...) {
        cout << "Unexpected Exception \n" ;
        return -1;
    }

    delete parser;
    delete handler;
    return 0;
}
