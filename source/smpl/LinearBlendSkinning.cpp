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
//  CLASS LinearBlendSkinning IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <xtensor/xbuilder.hpp>
#include <xtensor-blas/xlinalg.hpp>
//----------
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/XtensorEx.hpp"
#include "smpl/LinearBlendSkinning.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**LinearBlendSkinning
 * 
 * Brief
 * ----------
 * 
 *      Default constructor.
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
LinearBlendSkinning::LinearBlendSkinning() noexcept(true) :
    __restShape(),
    __transformation(),
    __weights(),
    __posedVert()
{
}

/**LinearBlendSkinning (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize weights for linear blend skinning.
 * 
 * Arguments
 * ----------
 * 
 *      weights: - xarray -
 *          Weights for linear blend skinning, (6890, 24).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
LinearBlendSkinning::LinearBlendSkinning(xt::xarray<double> weights) 
    noexcept(false)
{
    if (weights.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, JOINT_NUM})) {
        __weights = weights;
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to initialize linear blend weights!");
    }
}

/**LinearBlendSkinning (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @linearBlendSkinning: - const LinearBlendSkinning & -
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
LinearBlendSkinning::LinearBlendSkinning(
    const LinearBlendSkinning &linearBlendSkinning) noexcept(false) :
    __posedVert()
{
    try {
        *this = linearBlendSkinning;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~LinearBlendSkinning
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
LinearBlendSkinning::~LinearBlendSkinning() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <LinearBlendSkinning> instantiation.
 * 
 * Arguments
 * ----------
 * 
 *      @linearBlendSkinning: - xarray -
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - LinearBlendSkinning & -
 *          Current instantiation.
 * 
 */
LinearBlendSkinning &LinearBlendSkinning::operator=(
    const LinearBlendSkinning &linearBlendSkinning) noexcept(false)
{
    //
    // hard copy
    //
    if (linearBlendSkinning.__restShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __restShape = linearBlendSkinning.__restShape;
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to copy deformed shape in rest pose!");
    }

    if (linearBlendSkinning.__transformation.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        __transformation = linearBlendSkinning.__transformation;
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to copy world transformation!");
    }

    if (linearBlendSkinning.__weights.shape() ==
        xt::xarray<double>::shape_type(VERTEX_NUM, JOINT_NUM)) {
        __weights = linearBlendSkinning.__weights;
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to copy linear blend weights!");
    }

    //
    // soft copy
    //
    if (linearBlendSkinning.__posedVert.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __posedVert = linearBlendSkinning.__posedVert;
    }

    return *this;
}


/**setRestShape
 * 
 * Brief
 * ----------
 * 
 *      Set the deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 *      @restShape: - xarray -
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setRestShape(
    xt::xarray<double> restShape) noexcept(false)
{
    if (restShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __restShape = restShape;
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to set deformed shape in rest pose!");
    }

    return;
}

/**setWeight
 * 
 * Brief
 * ----------
 * 
 *      Set the weights for linear blend skinning.
 * 
 * Arguments
 * ----------
 * 
 *      weights: - xarray -
 *          Weights for linear blend skinning, (6890, 24).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setWeight(xt::xarray<double> weights) noexcept(false)
{
    if (weights.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, JOINT_NUM})) {
        __weights = weights;
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to set linear blend weights!");
    }

    return;
}

/**setTransformation
 * 
 * Brief
 * ----------
 * 
 *      Set the world transformation.
 * 
 * Arguments
 * ----------
 * 
 *      @transformation: - xarray -
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setTransformation(
    xt::xarray<double> transformation) noexcept(false)
{
    if (transformation.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        __transformation = transformation;
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to set world transformation!");
    }

    return;
}

/**getVertex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex locations of the new pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @vertices: - xarray -
 *          Vertex locations of the new pose, (N, 6890, 3).
 * 
 */
xt::xarray<double> LinearBlendSkinning::getVertex() noexcept(false)
{
    xt::xarray<double> vertices;
    if (__posedVert.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        vertices = __posedVert;
    }
    else {
        throw smpl_error("LinearBlendSknning", 
            "Failed to get vertices of new pose!");
    }

    return vertices;
}

/**skinning
 * 
 * Brief
 * ----------
 * 
 *      Do all the skinning stuffs.
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
void LinearBlendSkinning::skinning() noexcept(false)
{
    //
    // Cartesian coordinates to homogeneous coordinates
    //
    xt::xarray<double> restShapeHomo;
    try {
        restShapeHomo = cart2homo(__restShape);// (N, 6890, 4)
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // linear blend skinning
    //
    xt::xarray<double> coefficients = xt::linalg::tensordot(
        __weights, __transformation, {1}, {1});// (N, 6890, 4, 4)
    coefficients = xt::transpose(coefficients, {1, 0, 2, 3});// (N, 6890, 4, 4)
    restShapeHomo = xt::expand_dims(restShapeHomo, 3);// (N, 6890, 4, 1)
    xt::xarray<double> verticesHomo = XtensorEx::matmul(
        coefficients, restShapeHomo);// (N, 6890, 4, 4)
    verticesHomo = xt::squeeze(verticesHomo, 3);// (N, 6890, 4)

    //
    // homogeneous coordinates to Cartesian coordinates
    //
    try {
        __posedVert = homo2cart(verticesHomo);
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

/**cart2homo
 * 
 * Brief
 * ----------
 * 
 *      Convert Cartesian coordinates to homogeneous coordinates.
 * 
 * Argument
 * ----------
 * 
 *      @cart: - xarray -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 *      @homo: - xarray -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 */
xt::xarray<double> LinearBlendSkinning::cart2homo(xt::xarray<double> cart) 
    noexcept(false)
{
    if (cart.shape() !=
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        throw smpl_error("LinearBlendSkinning",
            "Cannot convert Cartesian coordinates to homogeneous one!");
    }

    xt::xarray<double> ones = xt::ones<double>(
        {BATCH_SIZE_RAW, VERTEX_NUM_RAW, 1});// (N, 6890, 1)
    xt::xarray<double> homo = xt::concatenate(
        xt::xtuple(cart, ones), 2);// (N, 6890, 4)

    return homo;
}

/**homo2cart
 * 
 * Brief
 * ----------
 * 
 *      Convert Cartesian coordinates to homogeneous coordinates.
 * 
 * Argument
 * ----------
 * 
 *      @homo: - xarray -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 * Return
 * ----------
 * 
 *      @cart: - xarray -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 */
xt::xarray<double> LinearBlendSkinning::homo2cart(xt::xarray<double> homo) 
    noexcept(false)
{
    if (homo.shape() !=
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 4})) {
        throw smpl_error("LinearBlendSkinning",
            "Cannot convert homogeneous coordinates to Cartesian ones!");
    }

    xt::xarray<double> homoW = xt::view(homo,
        xt::all(), xt::all(), 3);// (N, 6890)
    homoW = xt::expand_dims(homoW, 2);// (N, 6890, 1)
    xt::xarray<double> homoUnit = homo / homoW;// (N, 6890, 4)
    xt::xarray<double> cart = xt::view(homoUnit, 
        xt::all(), xt::all(), xt::range(0, 3));// (N, 6890, 3)
    
    return cart;
}

//=============================================================================
} // namespace smpl
//=============================================================================
