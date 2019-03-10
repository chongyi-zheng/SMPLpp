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
//  CLASS Exception DECLARATIONS
//
//=============================================================================

#ifndef EXCEPTION_H
#define EXCEPTION_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <exception>
#include <string>
#include <sstream>
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================

#ifndef smpl_error
#define smpl_error(module, error) Exception(module, error, \
    __func__, __FILE__, __LINE__)
#endif

//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/** 
 * DESCRIPTIONS:
 * 
 *      Exception to be thrown when a SMPL module works incorrectly.
 * 
 *      This class is a tool for the whole system.
 * 
 * INHERITANCE:
 * 
 *      - std::exception: <public>
 * 
 * ATTRIBUTES:
 *
 *      - m__module: <private>
 *          Name of the module that have just broken.
 *
 *      - m__error: <private>
 *          Error prompt to be dumped into the standard error port.
 *
 *      - m__function: <private>
 *          Name of the function which throws an exception.
 * 
 *      - m__file: <private>
 *          File of the source code where an exception is thrown.
 *
 * 
 *      - m__line: <private>
 *          Line in file of the source code where an exception is thrown.
 * 
 *      - m__stream: <private>
 *          A string stream contains all the error message.
 * 
 *      - m__message: <private>
 *          All the error message
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor
 *      %
 *      - Exception: <public>
 *          Constructor to create an exception instance.
 *
 *      - Exception: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~Exception: <public> <virtual/top>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment operator is used to copy an exception message.
 *      %%
 * 
 *      %
 *          Exception propagation
 *      %
 *      - what: <public> <virtual/down>
 *          Get what the exception stream is.
 *      %
 * 
 */

class Exception final : public std::exception
{

private: // PRIVATE ATTRIBUTES

    std::string m__module;
    std::string m__error;
    std::string m__function;
    std::string m__file;
    int m__line;

    std::stringstream m__stream;
    std::string m__message;

protected: // PROTECTED ATTRIBUTES


public: // PUBLIC ATTRIBUTES


private: // PRIVATE METHODS


protected: // PROTECTED METHODS


public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    Exception(const std::string module, const std::string error,
        const std::string function, const std::string file, const int line)
        noexcept(true);
    Exception(const Exception &exception) noexcept(true);
    ~Exception() noexcept(true) final;

    // %% Operators %%
    Exception &operator=(const Exception &exception) noexcept(true);
    
    // %% Exception propagation %%
    const char *what() const noexcept(true) final;

};

//=============================================================================
} // namespace SMPL
//=============================================================================
#endif // EXCEPTION_H
//=============================================================================
