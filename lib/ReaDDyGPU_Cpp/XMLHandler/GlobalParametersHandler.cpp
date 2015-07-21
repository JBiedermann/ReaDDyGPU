
#include "ReaDDyGPU.hpp"
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <map>

using namespace std;

// ---------------------------------------------------------------------------
//  GlobalParametersHandler: Constructors and Destructor
// ---------------------------------------------------------------------------
GlobalParametersHandler::GlobalParametersHandler(Simulation *simulation) :

    fAttrCount(0)
    , fCharacterCount(0)
    , fElementCount(0)
    , fSpaceCount(0)
    , fSawErrors(false)
{
    this->simulation = simulation;
}




GlobalParametersHandler::~GlobalParametersHandler()
{
}

// ---------------------------------------------------------------------------
//  GlobalParametersHandler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void GlobalParametersHandler::startElement(const XMLCh* const uri
                                   , const XMLCh* const localName
                                   , const XMLCh* const qname
                                   , const Attributes& atts)
{
    //fElementCount++;
    //fAttrCount += attrs.getLength();
    //cout << StrX(localname) <<endl;

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();

}


void GlobalParametersHandler::endElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname){

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();

    string tempCharBuffer = charBuffer.str();
    while(tempCharBuffer.find_first_of(' ')!=string::npos){
        tempCharBuffer.erase(tempCharBuffer.find_first_of(' '),1);
    }
    while(tempCharBuffer.find_first_of('\t')!=string::npos){
        tempCharBuffer.erase(tempCharBuffer.find_first_of('\t'),1);
    }
    while(tempCharBuffer.find_first_of('\n')!=string::npos){
        tempCharBuffer.erase(tempCharBuffer.find_first_of('\n'),1);
    }
    charBuffer.str(tempCharBuffer);


    if (0==sLocalName.compare("dt")) {
        simulation->stepSizeInPs_reactions = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("dtOpenMM")) {
        simulation->stepSizeInPs = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("dt_dynamic")) {
        simulation->stepSizeInPs = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("testmode")) {
        if(atoi(charBuffer.str().c_str())==1){
            simulation->testmode=true;
        }
        else{
            simulation->testmode=false;
        }
    }
    if (0==sLocalName.compare("dt_dyn")) {
        simulation->stepSizeInPs = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("nSimulationSteps")) {
        simulation->totalNumberOfFrames = atoi(charBuffer.str().c_str());
    }/*
    if (0==sLocalName.compare("nFrames")) {
        simulation->totalNumberOfFrames = atoi(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("nStepsPerFrame")) {
        simulation->stepsPerFrame = atoi(charBuffer.str().c_str());
    }*/
    if (0==sLocalName.compare("cudaDeviceIndex")) {
        simulation->cudaDevice = atoi(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("cudaDevice")) {
        simulation->cudaDevice = atoi(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("T")) {
        simulation->temperature = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("Kb")) {
        simulation->boltzmann = atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("latticeBounds")) {
        //<latticeBounds>[[-100,100];[-100,100];[-50,150]]</latticeBounds>
        //simulation->latticeBounds = atof(charBuffer.str().c_str());
        simulation->latticeBounds = new double[6];
        string buffer = charBuffer.str();
        buffer.erase(0,2);
        //cout << buffer << endl;
        simulation->latticeBounds[0]=atof(buffer.substr(0, buffer.find_first_of(',')).c_str());
        //cout << simulation->latticeBounds[0] << endl;
        buffer.erase(0,buffer.find_first_of(',')+1);
        //cout << buffer << endl;
        simulation->latticeBounds[1]=atof(buffer.substr(0, buffer.find_first_of(']')).c_str());
        //cout << simulation->latticeBounds[1] << endl;
        buffer.erase(0,buffer.find_first_of('[')+1);
        //cout << buffer << endl;
        simulation->latticeBounds[2]=atof(buffer.substr(0, buffer.find_first_of(',')).c_str());
        //cout << simulation->latticeBounds[2] << endl;
        buffer.erase(0,buffer.find_first_of(',')+1);
        //cout << buffer << endl;
        simulation->latticeBounds[3]=atof(buffer.substr(0, buffer.find_first_of(']')).c_str());
        //cout << simulation->latticeBounds[3] << endl;
        buffer.erase(0,buffer.find_first_of('[')+1);
        //cout << buffer << endl;
        simulation->latticeBounds[4]=atof(buffer.substr(0, buffer.find_first_of(',')).c_str());
        //cout << simulation->latticeBounds[4] << endl;
        buffer.erase(0,buffer.find_first_of(',')+1);
        //cout << buffer << endl;
        simulation->latticeBounds[5]=atof(buffer.substr(0, buffer.find_first_of(']')).c_str());
        //cout << simulation->latticeBounds[5] << endl;
    }

    charBuffer.str("");
}

/**
  * Receive notification of character data inside an element.
  *
  * <p>By default, do nothing.  Application writers may override this
  * method to take specific actions for each chunk of character data
  * (such as adding the data to a node or buffer, or printing it to
  * a file).</p>
  *
  * @param chars The characters.
  * @param length The number of characters to use from the
  *               character array.
  * @exception SAXException Any SAX exception, possibly
  *            wrapping another exception.
  * @see DocumentHandler#characters
  */
void GlobalParametersHandler::characters(  const   XMLCh* const   chars
                                    , const XMLSize_t length)
{
    fCharacterCount += length;
    //cout << "should:" << StrX(chars) << endl;
    charBuffer << StrX(chars) ;
    //strcpy(charBuffer, (char*)chars);
    /*for(int i=0; i<length; ++i){
        charBuffer.push_back(((char*)chars)[i]);
    }*/
}


void GlobalParametersHandler::ignorableWhitespace( const   XMLCh* const /* chars */
										    , const XMLSize_t length)
{
    fSpaceCount += length;
}

void GlobalParametersHandler::startDocument()
{
    fAttrCount = 0;
    fCharacterCount = 0;
    fElementCount = 0;
    fSpaceCount = 0;
}


// ---------------------------------------------------------------------------
//  GlobalParametersHandler: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void GlobalParametersHandler::error(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GlobalParametersHandler::fatalError(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GlobalParametersHandler::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GlobalParametersHandler::resetErrors()
{
    fSawErrors = false;
}
