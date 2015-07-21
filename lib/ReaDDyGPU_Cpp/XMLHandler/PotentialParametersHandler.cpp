
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
//  PotentialParametersHandler: Constructors and Destructor
// ---------------------------------------------------------------------------
PotentialParametersHandler::PotentialParametersHandler(Simulation *simulation) :

    fAttrCount(0)
    , fCharacterCount(0)
    , fElementCount(0)
    , fSpaceCount(0)
    , fSawErrors(false)
{
    this->simulation = simulation;
}




PotentialParametersHandler::~PotentialParametersHandler()
{
}

// ---------------------------------------------------------------------------
//  PotentialParametersHandler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void PotentialParametersHandler::startElement(const XMLCh* const uri
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

    if (0==sLocalName.compare("pot")) {
        potentials = map<string, string>();
        for(int i=0; i<atts.getLength(); ++i){
            stringstream s1;
            s1 << StrX(atts.getLocalName(i));
            string lName=s1.str();

            stringstream s2;
            s2 << StrX(atts.getValue(i));
            string value=s2.str();

            potentials[lName]=value;
            //potentials[lName]=atof(atts.getValue(i));

            //cout <<lName<< ": " << value<< endl;
        }
    }


}


void PotentialParametersHandler::endElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname){

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();

    if (0==sLocalName.compare("pot")) {
        simulation->plainPotentials.push_back(potentials);
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
void PotentialParametersHandler::characters(  const   XMLCh* const   chars
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


void PotentialParametersHandler::ignorableWhitespace( const   XMLCh* const /* chars */
										    , const XMLSize_t length)
{
    fSpaceCount += length;
}

void PotentialParametersHandler::startDocument()
{
    fAttrCount = 0;
    fCharacterCount = 0;
    fElementCount = 0;
    fSpaceCount = 0;
}


// ---------------------------------------------------------------------------
//  PotentialParametersHandler: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void PotentialParametersHandler::error(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void PotentialParametersHandler::fatalError(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void PotentialParametersHandler::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void PotentialParametersHandler::resetErrors()
{
    fSawErrors = false;
}
