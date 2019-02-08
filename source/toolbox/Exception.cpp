/* ========================================================================= *
 *                                                                           *
 *                                 SMPL++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - SMPL: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original SMPL model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  CLASS Exception IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include "toolbox/Exception.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/**Exception
 * 
 * Brief
 * ----------
 * 
 *      Constructor to create an exception instance.
 * 
 * Arguments
 * ----------
 * 
 *      @module: - const string -
 *          Name of the module that have just broken.
 *      
 *      @error: - const string -
 *          Error prompt to be dumped into the standard error port.
 * 
 *      @function: - const string -
 *          Name of the function which throws an exception.
 * 
 *      @file: - const string -
 *          File of the source code where an exception is thrown.
 * 
 *      @line: - const int -
 *          Line in file of the source code where an exception is thrown.
 * 
 * Return
 * ----------
 * 
 * 
 */
Exception::Exception(const std::string module, const std::string error,
        const std::string function, const std::string file, const int line)
        noexcept(true) :
    __module(module),
    __error(error),
    __function(function),
    __file(file),
    __line(line)
{
    __stream << __module << " Error: ";
    __stream << __error << std::endl;
    __stream << "Broken Function: " << __function << std::endl;
    __stream << "Broken File: " << __file << std::endl;
    __stream << "Broken Line: " << __line << std::endl;

    __message = __stream.str();
}

/**Exception (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 *      
 *      @exception: - const Exception & -
 *          The Exception instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
Exception::Exception(const Exception &exception) noexcept(true)
{
    *this = exception;
}

/**~Exception
 * 
 * Brief
 * ----------
 * 
 *      Deconstructor.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
Exception::~Exception() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment operator is used to copy an exception message.
 * 
 * Arguments
 * ----------
 * 
 *      @exception: - const Exception & -
 *          The Exception instance to be copied with.
 * 
 * Return
 * ----------
 *      @*this: - Exception & -
 *          Current instance.
 * 
 */
Exception &Exception::operator=(const Exception &exception) noexcept(true)
{
    __module = exception.__module;
    __error = exception.__error;
    __function = exception.__function;
    __file = exception.__file;
    __line = exception.__line;

    __stream << __module << " Error: ";
    __stream << __error << std::endl;
    __stream << "Broken Function: " << __function << std::endl;
    __stream << "Broken File: " << __file << std::endl;
    __stream << "Broken Line: " << __line;

    __message = __stream.str();

    return *this;
}

/**what
 * 
 * Brief
 * ----------
 * 
 *      Get what the exception stream is.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @stream: - const char * -
 *          Exception stream containing all error messages.
 * 
 */
const char *Exception::what() const noexcept(true)
{
    return __message.c_str();
}


//=============================================================================
} // namespace SMPL
//=============================================================================
