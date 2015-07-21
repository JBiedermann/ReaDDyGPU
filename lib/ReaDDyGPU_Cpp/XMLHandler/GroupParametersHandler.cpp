
#include "ReaDDyGPU.hpp"
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>

using namespace std;

// ---------------------------------------------------------------------------
//  GroupParametersHandler: Constructors and Destructor
// ---------------------------------------------------------------------------
GroupParametersHandler::GroupParametersHandler(Simulation *simulation) :

    fAttrCount(0)
    , fCharacterCount(0)
    , fElementCount(0)
    , fSpaceCount(0)
    , fSawErrors(false)
{
    this->simulation=simulation;
    //first = true;
    //nowBuildingBlocksParsing = false;
}

GroupParametersHandler::~GroupParametersHandler()
{
}


int CoordStringToDoubleArray(string in, double out[3]);


// ---------------------------------------------------------------------------
//  GroupParametersHandler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void GroupParametersHandler::startElement(const XMLCh* const uri
                                   , const XMLCh* const localName
                                   , const XMLCh* const qname
                                   , const Attributes& atts)
{

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();


    if (0==sLocalName.compare("particleGroup")) {
        groupData = Group();
    }

    if (0==sLocalName.compare("pot")) {
        map<string, string> groupPotential = map<string, string>();
        //involvedPotentials = vector<map<string, string> >();
        int nAtts = atts.getLength();
        for (int i = 0; i < nAtts; i++) {
            stringstream s1;
            s1 << StrX(atts.getLocalName(i));
            string lName=s1.str();

            stringstream s2;
            s2 << StrX(atts.getValue(i));
            string value=s2.str();

            groupPotential[lName]=value;

            /*stringstream s1;
            s1 << StrX(atts.getLocalName(i));
            string lName=s1.str();

            stringstream s2;
            s2 << StrX(atts.getValue(i));
            string value=s2.str();

            if (0==lName.compare("name")) {
                groupPotential.name=value;
            }
            if (0==lName.compare("type")) {
                groupPotential.type=value;
            }
            if (0==lName.compare("subtype")) {
                groupPotential.subtype=value;
            }
            if (0==lName.compare("forceConstant")) {
                groupPotential.forceConstant=atof(value.c_str());
            }
            if (0==lName.compare("affectedInternalIdPairs")) {
                groupPotential.affectedInternalIdPairsString=value;
            }*/
        }
        groupData.plainPotentials.push_back(groupPotential);
    }

    if (0==sLocalName.compare("particle")) {
        BuildingBlock buildingBlock;
        int nAtts = atts.getLength();
        for (int i = 0; i < nAtts; i++) {

            stringstream s1;
            s1 << StrX(atts.getLocalName(i));
            string lName=s1.str();

            stringstream s2;
            s2 << StrX(atts.getValue(i));
            string value=s2.str();

            if (0==lName.compare("internalIds")) {
                buildingBlock.internalID=atoi(value.c_str());
            }
            if (0==lName.compare("type")) {
                buildingBlock.type=value;
            }
            if (0==lName.compare("templateCs")) {
                CoordStringToDoubleArray(value, buildingBlock.templateCoord);
            }
        }
        groupData.buildingBlocks.push_back(buildingBlock);
    }


    if (0==sLocalName.compare("buildingBlocks")) {

    }
    if (0==sLocalName.compare("potentials")) {

    }

}


void GroupParametersHandler::endElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname){

    stringstream ss;
    ss << StrX(localName);
    string sLocalName=ss.str();
    string accumulator=charBuffer.str();
    //cout << sLocalName << endl;

    int i=0;
    while(accumulator.at(i)==' ' || accumulator.at(i)=='\t' || accumulator.at(i)=='\n' || accumulator.at(i)=='\r'|| accumulator.at(i)==10){
        accumulator.erase(i,1);
        //cout <<"." <<  accumulator<< "." << endl;
        if(accumulator.length()==0){
            break;
        }
    }
/*    i=accumulator.length()-1;
    while(accumulator.at(i)==' ' || accumulator.at(i)=='\t' || accumulator.at(i)=='\n' || accumulator.at(i)=='\r'){
        accumulator.erase(i,1);
        //cout <<"." <<  accumulator<< "." << endl;
        --i;
    }*/

    if (0==sLocalName.compare("type")) {
        groupData.type=accumulator;
        //cout << groupData.type << endl;
        groupData.ID=simulation->groups.size();
    }
    if (0==sLocalName.compare("templateOrigin")) {
        CoordStringToDoubleArray(accumulator, groupData.templateOrigin);
    }

    if (0==sLocalName.compare("templateNormal")) {
        CoordStringToDoubleArray(accumulator, groupData.templateNormal);
    }

    if (0==sLocalName.compare("maxNumberOfGroupAssignmentsPerParticle")) {
        groupData.maxNumberOfGroupAssignmentsPerParticle=atoi(accumulator.c_str());
    }

    if (0==sLocalName.compare("buildingBlocks")) {
    }

    if (0==sLocalName.compare("potentials")) {
    }

    if (0==sLocalName.compare("particleGroup")) {
        simulation->groups.push_back(groupData);
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
void GroupParametersHandler::characters(  const   XMLCh* const   chars
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


void GroupParametersHandler::ignorableWhitespace( const   XMLCh* const /* chars */
										    , const XMLSize_t length)
{
    fSpaceCount += length;
}

void GroupParametersHandler::startDocument()
{
    fAttrCount = 0;
    fCharacterCount = 0;
    fElementCount = 0;
    fSpaceCount = 0;
}


// ---------------------------------------------------------------------------
//  GroupParametersHandler: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void GroupParametersHandler::error(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GroupParametersHandler::fatalError(const SAXParseException& e)
{
    fSawErrors = true;
    XERCES_STD_QUALIFIER cerr << "\nFatal Error at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GroupParametersHandler::warning(const SAXParseException& e)
{
    XERCES_STD_QUALIFIER cerr << "\nWarning at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << XERCES_STD_QUALIFIER endl;
}

void GroupParametersHandler::resetErrors()
{
    fSawErrors = false;
}
