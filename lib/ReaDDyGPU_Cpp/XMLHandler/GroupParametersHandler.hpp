
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>

XERCES_CPP_NAMESPACE_USE

class GroupParametersHandler : public DefaultHandler
{
public:
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    GroupParametersHandler(Simulation *simulation);
    ~GroupParametersHandler();

    Simulation *simulation;
    stringstream charBuffer;
    //int currentParticleTypeId = -1;

    //bool nowBuildingBlocksParsing;
    //string currentBuildingBlockParticleOrGroup;
    //string currentInternalIds;
    //string currentBuildingBlockTypeName;
    //string currentBuildingBlockTemplateC;
    //string currentBuildingBlockTemplateCoords;
    //ArrayList<String[]> buildingBlocks;
    //bool first;
    //vector<string[]> buildingBlocksString;
    //vector<BuildingBlocks> buildingBlocks;
    //vector<map<string, string> > involvedPotentials;
    //vector<GroupPotentials> groupPotentials;
    //GroupPotential groupPotential;
    Group groupData;
    /*GroupData groupData;
    // list to cumulate the parsed particle data
    // and that is returned finally
    ArrayList<IGroupData> groupDataList = new ArrayList();
    IParamGroupsFileData paramGroupsFileData = null;
    ArrayList<HashMap<String, String>> involvedPotentials = null;*/


    // -----------------------------------------------------------------------
    //  Getter methods
    // -----------------------------------------------------------------------
    XMLSize_t getElementCount() const
    {
        return fElementCount;
    }

    XMLSize_t getAttrCount() const
    {
        return fAttrCount;
    }

    XMLSize_t getCharacterCount() const
    {
        return fCharacterCount;
    }

    bool getSawErrors() const
    {
        return fSawErrors;
    }

    XMLSize_t getSpaceCount() const
    {
        return fSpaceCount;
    }


    // -----------------------------------------------------------------------
    //  Handlers for the SAX ContentHandler interface
    // -----------------------------------------------------------------------
    void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs);
    void endElement(const XMLCh* const uri, const XMLCh* const localName, const XMLCh* const qname);
    void characters(const XMLCh* const chars, const XMLSize_t length);
    void ignorableWhitespace(const XMLCh* const chars, const XMLSize_t length);
    void startDocument();


    // -----------------------------------------------------------------------
    //  Handlers for the SAX ErrorHandler interface
    // -----------------------------------------------------------------------
	void warning(const SAXParseException& exc);
    void error(const SAXParseException& exc);
    void fatalError(const SAXParseException& exc);
    void resetErrors();


private:
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  fAttrCount
    //  fCharacterCount
    //  fElementCount
    //  fSpaceCount
    //      These are just counters that are run upwards based on the input
    //      from the document handlers.
    //
    //  fSawErrors
    //      This is set by the error handlers, and is queryable later to
    //      see if any errors occured.
    // -----------------------------------------------------------------------
    XMLSize_t       fAttrCount;
    XMLSize_t       fCharacterCount;
    XMLSize_t       fElementCount;
    XMLSize_t       fSpaceCount;
    bool            fSawErrors;
};
