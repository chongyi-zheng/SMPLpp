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
//  CLASS XtensorEx IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <typeinfo>
//----------
#include "toolbox/XtensorEx.hpp"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**XtensorEX
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
XtensorEx::XtensorEx() noexcept(false)
{
    throw smpl_error("XtensorEx", "Cannot call the constructor!");
}

/**XtensorEx (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @xtensorEx: - const XtensorEx & -
 *          The extra xtensor libraries to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
XtensorEx::XtensorEx(const XtensorEx &xtensorEx) noexcept(false)
{
    throw smpl_error("XtensorEx", "Cannot call the copy constructor!");
}

/**~XtensorEx
 * 
 * Brief
 * ----------
 * 
 *      Deconstructor.
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
XtensorEx::~XtensorEx() noexcept(false)
{
    throw smpl_error("XtensorEx", "Cannot call the deconstructor!");
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy an extra xtensor instantiation, but this
 *          one will never be used as it is a private method.
 * 
 * Arguments
 * ----------
 * 
 *      @xtensorEx: - const XtensorEx & -
 *          The extra xtensor libraries to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - Xtensor & -
 *          Current instantiation.
 * 
 */
XtensorEx &XtensorEx::operator=(const XtensorEx &xtensorEx) noexcept(false)
{
    throw smpl_error("XtensorEx", "Cannot not call the assignment operator!");
}

//=============================================================================
} // namespace smpl
//=============================================================================
