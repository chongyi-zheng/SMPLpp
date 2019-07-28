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
//  CLASS TorchEx IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include "toolbox/TorchEx.hpp"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**TorchEx
 * 
 * Brief
 * ----------
 * 
 *      Default constructor
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
TorchEx::TorchEx() noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the constructor!");
}

/**TorchEx (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @TorchEx: - const TorchEx & -
 *          The extra torch libraries to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
TorchEx::TorchEx(const TorchEx &TorchEx) noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the copy constructor!");
}

/**~TorchEx
 * 
 * Brief
 * ----------
 * 
 *      Destructor.
 * 
 * Argument:
 * ---------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
TorchEx::~TorchEx() noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the Destructor!");
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy an <TorchEx> instantiation, but this
 *          one will never be used as it is a private method.
 * 
 * Arguments
 * ----------
 * 
 *      @TorchEx: - const TorchEx & -
 *          The extra torch libraries to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - TorchEx & -
 *          Current instantiation.
 * 
 */
TorchEx &TorchEx::operator=(const TorchEx &TorchEx) noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot not call the assignment operator!");
}

//=============================================================================
} // namespace smpl
//=============================================================================
