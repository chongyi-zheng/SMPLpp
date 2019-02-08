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
//  CLASS BlendShape IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//-----------
#include <xtensor/xnorm.hpp>
//-----------
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/XtensorEx.hpp"
#include "smpl/BlendShape.h"
//-----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**BlendShape
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
BlendShape::BlendShape() noexcept(true) :
    __beta(),
    __shapeBlendBasis(),
    __shapeBlendShape(),
    __theta(),
    __restTheta(),
    __poseRot(),
    __restPoseRot(),
    __poseBlendBasis(),
    __poseBlendShape()
{
}

/**BlendShape (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize a BlendShape module.
 * 
 * Arguments
 * ----------
 * 
 *      @shapeBlendBasis: - xarray -
 *          Basis of the shape-dependent shape space, (6890, 3, 10).
 * 
 *      @poseBlendBasis: - xarray -
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 * Return
 * ----------
 * 
 * 
 */
BlendShape::BlendShape(xt::xarray<double> shapeBlendBasis,
        xt::xarray<double> poseBlendBasis) noexcept(false) :
    __beta(),
    __shapeBlendBasis(),
    __shapeBlendShape(),
    __theta(),
    __restTheta(),
    __poseRot(),
    __restPoseRot(),
    __poseBlendBasis(),
    __poseBlendShape()
{
    if (shapeBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        __shapeBlendBasis = shapeBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", 
            "Failed to initialize shape blend basis!");
    }

    if (poseBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        __poseBlendBasis = poseBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", 
            "Failed to initialize pose blend basis!");
    }
}

/**BlendShape (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @blendShape: - const BlendShape & -
 *          The <BlendShape> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
BlendShape::BlendShape(const BlendShape &blendShape) noexcept(false)
{
    try{
        *this = blendShape;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~BlendShape
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
BlendShape::~BlendShape() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy an BlendShape module.
 * 
 * Arguments
 * ----------
 * 
 *      @blendShape: - const BlendShape & -
 *          The <BlendShape> instance to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - BlendShape & -
 *          Current instance of the BlendShape class.
 * 
 */
BlendShape &BlendShape::operator=(const BlendShape &blendShape) noexcept(false)
{
    //
    //  hard copy
    //
    if (blendShape.__shapeBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        __shapeBlendBasis = blendShape.__shapeBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy shape blend basis!");
    }

    if (blendShape.__poseBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        __poseBlendBasis = blendShape.__poseBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy pose blend basis!");
    }

    if (blendShape.__beta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        __beta = blendShape.__beta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy beta!");
    }

    if (blendShape.__theta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __theta = blendShape.__theta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy theta!");
    }

    if (blendShape.__restTheta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __restTheta = blendShape.__restTheta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy theta of rest pose!");
    }

    //
    //  soft copy
    //
    if (blendShape.__shapeBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __shapeBlendShape = blendShape.__shapeBlendShape;
    }

    if (blendShape.__poseBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        __poseBlendShape = blendShape.__poseBlendShape;
    }

    if (blendShape.__poseRot.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        __poseRot = blendShape.__poseRot;
    }

    if (blendShape.__restPoseRot.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        __restPoseRot = blendShape.__restPoseRot;
    }

    return *this;
}

/**setBeta
 * 
 * Bridef:
 * ----------
 * 
 *      Set shape coefficient vector.
 * 
 * Arguments
 * ----------
 * 
 *      @beta: - xarray -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 * Return
 * ----------
 * 
 */
void BlendShape::setBeta(xt::xarray<double> beta) noexcept(false)
{
    if (beta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        __beta = beta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set beta!");
    }

    return;
}

/**setShapeBlendBasis
 * 
 * Brief
 * ----------
 * 
 *      Set shape blend basis.
 * 
 * Arguments
 * ----------
 * 
 *      @shapeBlendBasis: - xarray -
 *          Basis of the shape-dependent shape space, (6890, 3, 10).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setShapeBlendBasis(xt::xarray<double> shapeBlendBasis)
    noexcept(false)
{
    if (shapeBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        __shapeBlendBasis = shapeBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set shape blend basis!");
    }

    return;
}

/**setTheta
 * 
 * Brief
 * ----------
 * 
 *      Set new pose in axis-angle representation.
 * 
 * Arguments
 * ----------
 * 
 *      @theta: - xarray -
 *          Batch of pose axis-angle representations, (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setTheta(xt::xarray<double> theta) noexcept(false)
{
    if (theta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __theta = theta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set theta!");
    }

    return;
}

/**setRestTheta
 * 
 * Brief
 * ----------
 * 
 *      Set rest pose rotations in axis-angle representation.
 * 
 * Arguments
 * ----------
 * 
 *      @restTheta: - xarray -
 *          Batch of rest pose axis-angle representations, (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setRestTheta(xt::xarray<double> restTheta)
    noexcept(true)
{
    if (restTheta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __restTheta = restTheta;
    }

    return; 
}

/**setPoseBlendBasis
 * 
 * Brief
 * ----------
 * 
 *      Set pose blend basis.
 * 
 * Arguments
 * ----------
 * 
 *      @poseBlendBasis:
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setPoseBlendBasis(xt::xarray<double> poseBlendBasis)
    noexcept(false)
{
    if (poseBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        __poseBlendBasis = poseBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set pose blend basis!");
    }

    return;
}

/**getShapeBlendShape
 * 
 * Brief
 * ----------
 *      Get shape blend shape.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 *      @shapeBlendShape: - xarray -
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 */
xt::xarray<double> BlendShape::getShapeBlendShape() noexcept(false)
{
    xt::xarray<double> shapeBlendShape;

    if (__shapeBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        shapeBlendShape = __shapeBlendShape;
    }
    else {
        throw smpl_error("BlendShape", "Failed to get shape blend shape!");
    }

    return shapeBlendShape;
}

/**getPoseRotation
 * 
 * Brief
 * ----------
 * 
 *      Get pose rotation matrix.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @poseRotation: - xarray -
 *          Rotation with respect to pose axis-angles representations,
 *          (N, 24, 3, 3).
 * 
 */
xt::xarray<double> BlendShape::getPoseRotation() noexcept(false)
{
    xt::xarray<double> poseRotation;

    if (__poseRot.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        poseRotation = __poseRot;
    }
    else {
        throw smpl_error("BlendShape", "Failed to get pose rotation!");
    }

    return poseRotation;
}

/**getRestPoseRotation
 * 
 * Brief
 * ----------
 * 
 *      Get rest pose rotation matrix.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @restPoseRotation: - xarray -
 *          Pose rotation of rest pose, (N, 24, 3, 3).
 * 
 */
xt::xarray<double> BlendShape::getRestPoseRotation() noexcept(false)
{
    xt::xarray<double> restPoseRotation;

    if (__restPoseRot.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        restPoseRotation = __restPoseRot;
    }
    else {
        throw smpl_error("BlendShape", "Failed to get rest pose rotation!");
    }

    return restPoseRotation;
}

/**getPoseBlendShape
 * 
 * Brief
 * ----------
 *      Get pose blend shape.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @poseBlendShape: - xarray -
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 */
xt::xarray<double> BlendShape::getPoseBlendShape() noexcept(false)
{
    xt::xarray<double> poseBlendShape;

    if (__poseBlendShape.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 3})) {
        poseBlendShape = __poseBlendShape;
    }
    else {
        throw smpl_error("BlendShape", "Failed to get pose blend hape!");
    }

    return poseBlendShape;
}

/**blend
 * 
 * Brief
 * ----------
 * 
 *      Outside monitor to generate blend shape.
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
void BlendShape::blend() noexcept(false)
{
    //
    // pose blend
    //
    try {
        poseBlend();// save result in <__poseBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // shape blend
    //
    try {
        shapeBlend();// save result in <__shapeBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }
}

/**shapeBlend
 * 
 * Brief
 * ----------
 * 
 *      Generate shape blend shape.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 * Notes
 * ----------
 * 
 *      This function implements equation (8) in the paper.
 * 
 */
void BlendShape::shapeBlend() noexcept(false)
{
    if (__shapeBlendBasis.shape() ==
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, SHAPE_BASIS_DIM})
        && __beta.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        __shapeBlendShape = 
            xt::linalg::tensordot(__beta, __shapeBlendBasis, {1}, {2});
            // (N, 6890, 3)
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend shape-dependented shape!");
    }

    return;
}

/**poseBlend
 * 
 * Brief
 * ----------
 * 
 *      Generate pose blend shape.
 * 
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 * Notes
 * ----------
 * 
 *      This function implements equation (9) in the paper.
 * 
 */
void BlendShape::poseBlend() noexcept(false)
{
    //
    // pose rotation and rest pose rotation
    //
    if (__theta.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        try {
            __poseRot = rodrigues(__theta);// (N, 24, 3, 3) 
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend pose-dependented shape!");;
    }

    if (__restTheta.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        try {
            __restPoseRot = rodrigues(__restTheta);// (N, 24, 3, 3)            
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {
        __restPoseRot = xt::eye({BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
    }

    //
    //  pose blend coefficients
    //
    xt::xarray<double> poseBlendCoeffs;
    try {
        poseBlendCoeffs = linRotMin();
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // pose blend
    //
    if (__poseBlendBasis.shape() == 
        xt::xarray<double>::shape_type({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        __poseBlendShape = 
            xt::linalg::tensordot(poseBlendCoeffs, __poseBlendBasis, 
                {1}, {2});// (N, 6890, 3)
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend pose-dependented shape!");;
    }

    return;
}

/**rodrigues
 * 
 * Brief
 * ----------
 * 
 *      Get arbitrary rotations in axis-angle representations using
 *      Rodrigues' formula.
 * 
 * Arguments
 * ----------
 * 
 *      @theta: - xarray -
 *          pose axis-angle representations.
 *              Batch of pose or rest pose in axis-angle representations,
 *              (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 *      @rotation: - xarray -
 *          Batch of pose or rest pose rotation, (N, 24, 3, 3).
 * 
 * Notes
 * ----------
 * 
 *      This function implements equation (1) in the paper.
 * 
 */
xt::xarray<double> BlendShape::rodrigues(xt::xarray<double> theta)
    noexcept(false)
{
    if (theta.shape() !=
        xt::xarray<double>::shape_type({BATCH_SIZE, 24, 3})) {
        throw smpl_error("BlendShape", "Cannot do arbitrary rotation!");
    }
    // std::cout << theta << std::endl;

    //
    // rotation angles and axis
    //
    xt::xarray<double> angles = xt::norm_l2(theta, {2});// (N, 24)
    angles = xt::expand_dims(angles, 2);// (N, 24, 1)
    xt::xarray<double> axes = theta / angles;// (N, 24, 3)

    //
    // skew symmetric matrices
    //
    xt::xarray<double> zero = xt::zeros<double>(
        {BATCH_SIZE_RAW, JOINT_NUM_RAW});// (N, 24)
    xt::xarray<double> tiledSkew = xt::stack(
        xt::xtuple(
            zero,
            -xt::view(axes, xt::all(), xt::all(), 2),
            xt::view(axes, xt::all(), xt::all(), 1),

            xt::view(axes, xt::all(), xt::all(), 2),
            zero,
            -xt::view(axes, xt::all(), xt::all(), 0),

            -xt::view(axes, xt::all(), xt::all(), 1),
            xt::view(axes, xt::all(), xt::all(), 0),
            zero
        ), 2); // (N, 24, 9)
    xt::xarray<double> skew = xt::reshape_view(
        tiledSkew, {BATCH_SIZE_RAW, JOINT_NUM_RAW, 3, 3});// (N, 24, 3, 3)
    xt::xarray<double> skewSq;
    try {
        skewSq = XtensorEx::matmul(skew, skew);
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // Rodrigues' formula
    //
    xt::xarray<double> eye = xt::eye(
        {BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
    xt::xarray<double> sine = xt::sin(
        xt::expand_dims(angles, 3)
    );
    sine = xt::broadcast(sine, 
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 3, 3});// (N, 24, 3, 3)
    xt::xarray<double> cosine = xt::cos(
        xt::expand_dims(angles, 3)
    );
    cosine = xt::broadcast(cosine, 
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 3, 3});// (N, 24, 3, 3)
    xt::xarray<double> rotation = eye 
        + skew * sine 
        + skewSq * (1 - cosine);// (N, 24, 3, 3)

    return rotation;
}

/**linRotMin
 * 
 * Brief
 * ----------
 * 
 *      Eliminate the influence of rest pose on pose blend shape and
 *      generate pose blend coefficients (linear rotation minimization).
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @poseBlendCoeffs: - xarray -
 *          Pose blend coefficients for combining pose blend basis.
 * 
 */
xt::xarray<double> BlendShape::linRotMin() noexcept(false)
{
    //
    // unroll rotations
    //
    xt::xarray<double> unPoseRot, unRestPoseRot;
    try {
        unPoseRot = unroll(__poseRot);// (N, 216)
        unRestPoseRot = unroll(__restPoseRot);// (N, 216)
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // truncate rotations
    //
    unPoseRot = xt::view(unPoseRot, 
        xt::all(), xt::range(9, xt::placeholders::_));// (N, 207)
    unRestPoseRot = xt::view(unRestPoseRot,
        xt::all(), xt::range(9, xt::placeholders::_));// (N, 207)

    //
    // pose blend coefficients
    //
    xt::xarray<double> poseBlendCoeffs = unPoseRot - unRestPoseRot;// (N, 207)

    return poseBlendCoeffs;
}

/**unroll
 * 
 * Brief
 * ----------
 * 
 * 
 * Arguments
 * ----------
 * 
 *      @rotation: - xarray -
 *          Pose or rest pose rotation to be unrolled, (N, 24, 3, 3).
 * 
 * Return
 * ----------
 * 
 *      @unRotation: - xarray -
 *          Unrolled pose or rest pose rotation, (N, 216).
 * 
 * 
 */
xt::xarray<double> BlendShape::unroll(xt::xarray<double> rotation) 
    noexcept(false)
{
    if (rotation.shape() !=
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        throw smpl_error("BlendShape", "Cannot unroll a rotation!");
    }

    xt::xarray<double> unRotation(rotation);
    unRotation.reshape({BATCH_SIZE, JOINT_NUM * 3 * 3});

    return unRotation;
}

//=============================================================================
} // namespace SMPL
//=============================================================================
