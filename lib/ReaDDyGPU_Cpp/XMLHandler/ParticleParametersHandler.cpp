
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
//  ParticleParametersHandler: Constructors and Destructor
// ---------------------------------------------------------------------------
ParticleParametersHandler::ParticleParametersHandler(Simulation *simulation) :

    fAttrCount(0)
    , fCharacterCount(0)
    , fElementCount(0)
    , fSpaceCount(0)
    , fSawErrors(false)
{
    this->simulation = simulation;
    this->first=true;
    collisionRadiusParsing = false;
}

ParticleParametersHandler::~ParticleParametersHandler()
{
}

// ---------------------------------------------------------------------------
//  ParticleParametersHandler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
/*void ParticleParametersHandler::startElement(const XMLCh* const uri
                                   , const XMLCh* const localname
                                   , const XMLCh* const qname
                                   , const Attributes& attrs)
{
    fElementCount++;
    fAttrCount += attrs.getLength();
    cout << StrX(localname) <<endl;
}*/


void ParticleParametersHandler::startElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname, const Attributes& atts){

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();
    //cout << "..." << sLocalName<< endl;

    //accumulator.setLength(0);

    if (0==sLocalName.compare("particle")) {
        particleType = ParticleType();
    }
    // RADII
    if (0==sLocalName.compare("collisionRadiusMap")) {
        collisionRadii = vector<vector<double> >();
    }
    if (0==sLocalName.compare("collisionRadius")) {
        collisionRadiusPartnerType = "";
        collisionRadiusValue = 0;
        collisionRadiusParsing = true;
    }

}

/*int split(string in, string front, string end, char split){
    for(int i=0; i<in.length(); ++i){
        if(in.at(i)==split){
            //size_t copy (char* s, size_t len, size_t pos = 0) const;
            char pFront[20];
            in.copy(pFront, (size_t)i-1, (size_t)0);
            front=string(pFront);
            char pEnd[20];
            in.copy(pEnd, (size_t)in.length()-i-1, (size_t)i+1);
            end=string(pEnd);
            cout << front << "." << end << endl;
            return 0;
        }
    }
    return 1;
}*/

void ParticleParametersHandler::endElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname){

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();

    //cout << StrX(uri) << " " << StrX(localName) << " " << StrX(qname) << endl;
    //cout << "1 " << charBuffer.str() <<endl;
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
    //cout << "2 " << tempCharBuffer << endl;
    //cout << charBuffer.str() <<endl;

    if (0==sLocalName.compare("particle")) {
        particleType.ID=simulation->particleTypes.size();

        //particleType.radiiMatrix=collisionRadii;
        particleType.collisionRadiiMap=collisionRadiiMap;

        simulation->particleTypes.push_back(particleType);
    }
    if (0==sLocalName.compare("type")) {
        particleType.name=charBuffer.str();
    }
    if (0==sLocalName.compare("diffusionConstant")) {
        particleType.D=atof(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("numberOfDummyParticles")) {
        particleType.numberOfDummyParticles=atoi(charBuffer.str().c_str());
    }
    if (0==sLocalName.compare("partnerType")) {
        if (collisionRadiusParsing) {
            collisionRadiusPartnerType = charBuffer.str();
        }
    }
    if (0==sLocalName.compare("radius")) {
        if (collisionRadiusParsing) {
            collisionRadiusValue = atof(charBuffer.str().c_str());
        }
    }
    if (0==sLocalName.compare("collisionRadius")) {
        if (collisionRadiusParsing) {

                vector<string> collisionPartners = vector<string>();
                int last=0;
                int i;
                for(i=0; i<collisionRadiusPartnerType.length(); ++i){
                    if(collisionRadiusPartnerType.at(i)==','){
                        //cout << collisionRadiusPartnerType.substr(last, i-last) << endl;
                        collisionPartners.push_back(collisionRadiusPartnerType.substr(last, i-last));
                        last=i+1;
                    }
                }
                //cout << collisionRadiusPartnerType.substr(last, i-last) << endl;
                collisionPartners.push_back(collisionRadiusPartnerType.substr(last, i-last));
                for(i=0; i<collisionPartners.size(); ++i){
                    //cout << "collision radius for " << collisionPartners[i] << " = " << collisionRadiusValue <<endl;
                    collisionRadiiMap[collisionPartners[i]]=collisionRadiusValue;
                }

        }
        collisionRadiusParsing = false;
    }
    charBuffer.str("");
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////
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
void ParticleParametersHandler::characters(  const   XMLCh* const   chars
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

void ParticleParametersHandler::ignorableWhitespace( const   XMLCh* const /* chars */
										    , const XMLSize_t length)
{
    fSpaceCount += length;
}

void ParticleParametersHandler::startDocument()
{
    fAttrCount = 0;
    fCharacterCount = 0;
    fElementCount = 0;
    fSpaceCount = 0;
}


// ---------------------------------------------------------------------------
//  ParticleParametersHandler: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void ParticleParametersHandler::error(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void ParticleParametersHandler::fatalError(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void ParticleParametersHandler::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void ParticleParametersHandler::resetErrors()
{
    fSawErrors = false;
}
