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
//  CLASS JointRegression IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
//----------
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "smpl/JointRegression.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**JointRegression
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
JointRegression::JointRegression() noexcept(true) :
    __restShape(),
    __shapeBlendShape(),
    __poseBlendShape(),
    __templateRestShape(),
    __joints(),
    __jointRegressor()
{
}

/**JointRegression (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize joint regressor and template shape.
 * 
 * Arguments
 * ----------
 * 
 *      @jointRegressor: - xarray -
 *          The joint coefficients, (24, 6890).
 * 
 *      @templateRestShape: - xarray -
 *          The template shape in rest pose, (6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
JointRegression::JointRegression(xt::xarray<double> jointRegressor,
    xt::xarray<double> templateRestShape) noexcept(false) :
    __restShape(),
    __shapeBlendShape(),
    __poseBlendShape(),
    __joints()
{
    if (jointRegressor.shape() == 
        xt::xarray<double>::shape_type({JOINT_NUM, VERTEX_NUM})) {
        __jointRegressor = jointRegressor;
    }
    else {
        throw smpl_error("JointRegression", 
            "Failed to initialize joint regressor!");
    }

    if (templateRestShape.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, 3})) {
        __templateRestShape = templateRestShape;
    }
    else {
        throw smpl_error("JointRegression", 
            "Failed to initialize template shape!");
    }
}

/**JointRegression (overload)
 *  
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @jointRegression: - xarray -
 *          The <JointRegression> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
JointRegression::JointRegression(const JointRegression &jointRegression)
    noexcept(false) :
    __restShape(),
    __joints()
{
    try {
        *this = jointRegression;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~JointRegression
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
JointRegression::~JointRegression() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <JointRegression> instantiation.
 * 
 * Arguments
 * ----------
 * 
 *      @jointRegression: - xarray -
 *          The <JointRegression> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - xarray -
 *          Current instantiation.
 * 
 */
JointRegression &JointRegression::operator=(
    const JointRegression &jointRegression) noexcept(false)
{
    //
    // hard copy
    //
    if (jointRegression.__shapeBlendShape.shape()
        == xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __shapeBlendShape = jointRegression.__shapeBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy shape blend shape!");
    }

    if (jointRegression.__poseBlendShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __poseBlendShape = jointRegression.__poseBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy pose blend shape!");
    }

    if (jointRegression.__templateRestShape.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, 3})) {
        __templateRestShape = jointRegression.__templateRestShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy template shape!");
    }

    if (jointRegression.__jointRegressor.shape() == 
        xt::xarray<double>::shape_type({JOINT_NUM, VERTEX_NUM})) {
        __jointRegressor = jointRegression.__jointRegressor;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy joint regressor!");
    }

    //
    // soft copy
    //
    if (jointRegression.__restShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __restShape = jointRegression.__restShape;
    }

    if (jointRegression.__joints.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __joints = jointRegression.__joints;
    }
}

/**setShapeBlendShape
 * 
 * Brief
 * ----------
 * 
 *      Set shape blend shape.
 * 
 * 
 * Arguments
 * ----------
 * 
 *      @shapeBlendShape: - xarray -
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setShapeBlendShape(xt::xarray<double> shapeBlendShape)
    noexcept(false)
{
    if (shapeBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __shapeBlendShape = shapeBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set shape blend shape!");
    }

    return;
}

/**setPoseBlendShape
 * 
 * Brief
 * ----------
 * 
 *      Set pose blend shape.
 * 
 * Arguments
 * ----------
 * 
 *      @poseBlendShape: - xarray -
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setPoseBlendShape(xt::xarray<double> poseBlendShape)
    noexcept(false)
{
    if (poseBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __poseBlendShape = poseBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set pose blend shape!");        
    }

    return;
}

/**setTemplateRestShape.
 * 
 * Brief
 * ----------
 * 
 *      Set template shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 *      @templateRestShape: - xarray -
 *          Template shape in rest pose, (6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setTemplateRestShape(
    xt::xarray<double> templateRestShape) noexcept(false)
{
    if (templateRestShape.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, 3})) {
        __templateRestShape = templateRestShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set template shape!");
    }

    return;
}

/**setJointRegressor
 * 
 * Brief
 * ----------
 * 
 *      Set the joint coefficients.
 * 
 * Arguments
 * ----------
 * 
 *      @jointRegressor: - xarray -
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setJointRegressor(xt::xarray<double> jointRegressor)
    noexcept(false)
{
    if (jointRegressor.shape() ==
        xt::xarray<double>::shape_type({JOINT_NUM, VERTEX_NUM})) {
        __jointRegressor = jointRegressor;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set joint regressor!");
    }
}

/**getRestShape
 * 
 * Brief
 * ----------
 * 
 *      Get the deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @restShape: - xarray -
 *          Deformed shape in rest pose, (N, 6890, 3).     
 * 
 */
xt::xarray<double> JointRegression::getRestShape() noexcept(false)
{
    xt::xarray<double> restShape;

    if (__restShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        restShape = __restShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to get rest shape!");
    }

    return restShape;
}

/**getJoint
 * 
 * Brief
 * ----------
 * 
 *      Get global joint locations.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @joints: - xarray -
 *          Joint locations of the deformed shape, (N, 24, 3).
 * 
 */
xt::xarray<double> JointRegression::getJoint() noexcept(false)
{
    xt::xarray<double> joints;

    if (__joints.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        joints = __joints;
    }
    else {
        throw smpl_error("JointRegression", "Failed to get joints!");
    }

    return joints;
}

/**regress
 * 
 * Brief
 * ----------
 * 
 *      Outside wrapper to encapsulate joint regression process.
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
void JointRegression::regress() noexcept(false)
{
    //
    // linearly combine shapes
    //
    try {
        linearCombine();
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // joint regression
    //
    try {
        jointRegress();
    }
    catch(std::exception &e) {
        throw;
    }
}

/**linearCombine
 * 
 * Brief
 * ----------
 * 
 *          Linearly combining pose-dependent shape and shape-dependent shape
 *          with template shape.
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
void JointRegression::linearCombine() noexcept(false)
{
    if (__shapeBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})
        && __poseBlendShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})
        && __templateRestShape.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3})) {
        __restShape = __templateRestShape + __shapeBlendShape + 
            __poseBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Cannot linearly combine shapes!");
    }

    return;
}

/**jointRegress
 * 
 * Brief
 * ----------
 * 
 *      Regress the rest shape for new pose into joints.
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
void JointRegression::jointRegress() noexcept(false)
{
    if (__shapeBlendShape.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})
        && __templateRestShape.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3})) {
        xt::xarray<double> shape = __templateRestShape + 
            __shapeBlendShape;// (N, 6890, 3)
        __joints = xt::linalg::tensordot(shape, __jointRegressor, 
            {1}, {1});// (N, 3, 24)
        __joints = xt::transpose(__joints, {0, 2, 1});// (N, 24, 3)
    }
    else {
        throw smpl_error("JointRegression", 
            "Cannot regress vertices to joints!");
    }

    return;
}

//=============================================================================
} // namespace smpl
//=============================================================================
