
#include <xercesc/util/PlatformUtils.hpp>
#include <stdlib.h>
#include <string>
#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif
#include "CudaKernels/CudaSimulation.cuh"
#include "Simulation.hpp"
#include "XMLHandler/GroupParametersHandler.hpp"
#include "XMLHandler/PotentialParametersHandler.hpp"
#include "XMLHandler/ParticleParametersHandler.hpp"
#include "XMLHandler/XMLReader.hpp"
#include "XMLHandler/CoordReader.hpp"
#include "XMLHandler/GroupReader.hpp"
#include "XMLHandler/GlobalParametersHandler.hpp"
#include "outputHandler/outputHandler.hpp"
#include "AuxillaryFunctions.hpp"
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>


class StrX
{
public :
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    StrX(const XMLCh* const toTranscode)
    {
        // Call the private transcoding method
        fLocalForm = XMLString::transcode(toTranscode);
    }

    ~StrX()
    {
        XMLString::release(&fLocalForm);
    }

    // -----------------------------------------------------------------------
    //  Getter methods
    // -----------------------------------------------------------------------
    const char* localForm() const
    {
        return fLocalForm;
    }

private :
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  fLocalForm
    //      This is the local code page form of the string.
    // -----------------------------------------------------------------------
    char*   fLocalForm;
};

inline XERCES_STD_QUALIFIER ostream& operator<<(XERCES_STD_QUALIFIER ostream& target, const StrX& toDump)
{
    target << toDump.localForm();
    return target;
}
