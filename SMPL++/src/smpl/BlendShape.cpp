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
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include "toolbox/Exception.h"
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
    m__device(torch::kCPU),
    m__beta(),
    m__shapeBlendBasis(),
    m__shapeBlendShape(),
    m__theta(),
    m__restTheta(),
    m__poseRot(),
    m__restPoseRot(),
    m__poseBlendBasis(),
    m__poseBlendShape()
{
}

/**BlendShape (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize shape blend basis, pose blend basis, and
 *      torch device.
 * 
 * Arguments
 * ----------
 * 
 *      @shapeBlendBasis: - Tensor -
 *          Basis of the shape-dependent shape space, (6890, 3, 10).
 * 
 *      @poseBlendBasis: - Tensor -
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 *      @device: - Device -
 *          Torch device to run the module, CPUs or GPUs.
 * 
 * Return
 * ----------
 * 
 * 
 */
BlendShape::BlendShape(torch::Tensor &shapeBlendBasis,
        torch::Tensor &poseBlendBasis, torch::Device &device) noexcept(false) :
    m__device(torch::kCPU),
    m__beta(),
    m__shapeBlendBasis(),
    m__shapeBlendShape(),
    m__theta(),
    m__restTheta(),
    m__poseRot(),
    m__restPoseRot(),
    m__poseBlendBasis(),
    m__poseBlendShape()
{
    if (m__device.has_index()) {
        m__device = m__device;
    }
    else {
        throw smpl_error("BlendShape", "Failed to fetch device index!");
    }

    if (shapeBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = shapeBlendBasis.clone().to(m__device);
    }
    else {
        throw smpl_error("BlendShape", 
            "Failed to initialize shape blend basis!");
    }

    if (poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        m__poseBlendBasis = poseBlendBasis.clone().to(m__device);
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
BlendShape::BlendShape(const BlendShape &blendShape) noexcept(false) :
    m__device(torch::kCPU)
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
 *      Destructor.
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
    if (blendShape.m__device.has_index()) {
        m__device = blendShape.m__device;
    }
    else {
        throw smpl_error("BlendShape", "Failed to fetch device index!");
    }

    if (blendShape.m__shapeBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = blendShape.m__shapeBlendBasis.clone().to(
            m__device);
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy shape blend basis!");
    }

    if (blendShape.m__poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        m__poseBlendBasis = blendShape.m__poseBlendBasis.clone().to(m__device);
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy pose blend basis!");
    }

    if (blendShape.m__beta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        m__beta = blendShape.m__beta.clone().to(m__device);
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy beta!");
    }

    if (blendShape.m__theta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        m__theta = blendShape.m__theta.clone().to(m__device);
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy theta!");
    }

    if (blendShape.m__restTheta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        m__restTheta = blendShape.m__restTheta.clone().to(m__device);
    }
    else {
        throw smpl_error("BlendShape", "Failed to copy theta of rest pose!");
    }

    //
    //  soft copy
    //
    if (blendShape.m__shapeBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__shapeBlendShape = blendShape.m__shapeBlendShape.clone().to(
            m__device);
    }

    if (blendShape.m__poseBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__poseBlendShape = blendShape.m__poseBlendShape.clone().to(m__device);
    }

    if (blendShape.m__poseRot.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        m__poseRot = blendShape.m__poseRot.clone().to(m__device);
    }

    if (blendShape.m__restPoseRot.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        m__restPoseRot = blendShape.m__restPoseRot.clone().to(m__device);
    }

    return *this;
}

/**setDevice
 * 
 * Brief
 * ----------
 * 
 *      Set the torch device.
 * 
 * Arguments
 * ----------
 * 
 *      @device: - const Device & -
 *          The torch device to be used.
 * 
 * Return
 * ----------
 * 
 */
void BlendShape::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("BlendShape", "Failed to fetch device index!");
    }

    return;
}

/**setBeta
 * 
 * Brief
 * ----------
 * 
 *      Set shape coefficient vector.
 * 
 * Arguments
 * ----------
 * 
 *      @beta: - const Tensor & -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 * Return
 * ----------
 * 
 */
void BlendShape::setBeta(const torch::Tensor &beta) noexcept(false)
{
    if (beta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        m__beta = beta.clone().to(m__device);
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
 *      @shapeBlendBasis: - const Tensor & -
 *          Basis of the shape-dependent shape space, (6890, 3, 10).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setShapeBlendBasis(const torch::Tensor &shapeBlendBasis)
    noexcept(false)
{
    if (shapeBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = shapeBlendBasis.clone().to(m__device);
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
 *      @theta: - const Tensor & -
 *          Batch of pose axis-angle representations, (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setTheta(const torch::Tensor &theta) noexcept(false)
{
    if (theta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        m__theta = theta.clone().to(m__device);
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
 *      @restTheta: - const Tensor & -
 *          Batch of rest pose axis-angle representations, (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setRestTheta(const torch::Tensor &restTheta)
    noexcept(true)
{
    if (restTheta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        m__restTheta = restTheta.clone().to(m__device);
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
 *      @poseBlendBasis: - const Tensor & -
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 * Return
 * ----------
 * 
 * 
 */
void BlendShape::setPoseBlendBasis(const torch::Tensor &poseBlendBasis)
    noexcept(false)
{
    if (poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        m__poseBlendBasis = poseBlendBasis.clone().to(m__device);
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
 *      @shapeBlendShape: - Tensor -
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 */
torch::Tensor BlendShape::getShapeBlendShape() noexcept(false)
{
    torch::Tensor shapeBlendShape;

    if (m__shapeBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        shapeBlendShape = m__shapeBlendShape.clone().to(m__device);
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
 *      @poseRotation: - Tensor -
 *          Rotation with respect to pose axis-angles representations,
 *          (N, 24, 3, 3).
 * 
 */
torch::Tensor BlendShape::getPoseRotation() noexcept(false)
{
    torch::Tensor poseRotation;

    if (m__poseRot.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        poseRotation = m__poseRot.clone().to(m__device);
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
 *      @restPoseRotation: - Tensor -
 *          Pose rotation of rest pose, (N, 24, 3, 3).
 * 
 */
torch::Tensor BlendShape::getRestPoseRotation() noexcept(false)
{
    torch::Tensor restPoseRotation;

    if (m__restPoseRot.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        restPoseRotation = m__restPoseRot.clone().to(m__device);
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
 *      @poseBlendShape: - Tensor -
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 */
torch::Tensor BlendShape::getPoseBlendShape() noexcept(false)
{
    torch::Tensor poseBlendShape;

    if (m__poseBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        poseBlendShape = m__poseBlendShape.clone().to(m__device);
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
        poseBlend();// save result in <m__poseBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // shape blend
    //
    try {
        shapeBlend();// save result in <m__shapeBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }

    return;
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
    if (m__shapeBlendBasis.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})
        && m__beta.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})) {
        m__shapeBlendShape = 
            torch::tensordot(m__beta, m__shapeBlendBasis, {1}, {2});// (N, 6890, 3)
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
    if (m__theta.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        try {
            m__poseRot = rodrigues(m__theta);// (N, 24, 3, 3) 
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend pose-dependented shape!");;
    }

    if (!m__restTheta.is_same(torch::Tensor())
        && m__restTheta.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        try {
            m__restPoseRot = rodrigues(m__restTheta);// (N, 24, 3, 3)            
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {
        m__restPoseRot = torch::eye(3, m__device);// (3, 3)
        m__restPoseRot = m__restPoseRot.expand(
            {BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
    }

    //
    //  pose blend coefficients
    //
    torch::Tensor poseBlendCoeffs;
    try {
        poseBlendCoeffs = linRotMin();
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // pose blend
    //
    if (m__poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        m__poseBlendShape = torch::tensordot(
            poseBlendCoeffs, m__poseBlendBasis, {1}, {2});// (N, 6890, 3)
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
 *      @theta: - Tensor -
 *          pose axis-angle representations.
 *              Batch of pose or rest pose in axis-angle representations,
 *              (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 *      @rotation: - Tensor -
 *          Batch of pose or rest pose rotation, (N, 24, 3, 3).
 * 
 * Notes
 * ----------
 * 
 *      This function implements equation (1) in the paper.
 * 
 */
torch::Tensor BlendShape::rodrigues(torch::Tensor &theta)
    noexcept(false)
{
    if (theta.sizes() !=
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        throw smpl_error("BlendShape", "Cannot do arbitrary rotation!");
    }

    //
    // rotation angles and axis
    //
    torch::Tensor angles = torch::norm(theta, 2, {2}, true);// (N, 24, 1)
    torch::Tensor axes = theta / angles;// (N, 24, 3)

    //
    // skew symmetric matrices
    //
    torch::Tensor zeros = torch::zeros({BATCH_SIZE, JOINT_NUM}, 
        m__device);// (N, 24)
    torch::Tensor skew = torch::stack({
            zeros,
            -TorchEx::indexing(axes, 
                torch::IntList(), 
                torch::IntList(), 
                torch::IntList({2})
            ),
            TorchEx::indexing(axes, 
                torch::IntList(), 
                torch::IntList(), 
                torch::IntList({1})
            ),

            TorchEx::indexing(axes, 
                torch::IntList(), 
                torch::IntList(), 
                torch::IntList({2})
            ),
            zeros,
            -TorchEx::indexing(axes,
                torch::IntList(),
                torch::IntList(),
                torch::IntList({0})
            ),

            -TorchEx::indexing(axes,
                torch::IntList(),
                torch::IntList(),
                torch::IntList({1})
            ),
            TorchEx::indexing(axes,
                torch::IntList(),
                torch::IntList(),
                torch::IntList({0})
            ),
            zeros
        }, 2);// (N, 24, 9)
    skew = torch::reshape(skew, 
        {BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)

    //
    // Rodrigues' formula
    //
    torch::Tensor eye = torch::eye(3, m__device);// (3, 3)
    eye = eye.expand({BATCH_SIZE, JOINT_NUM, 3, 3});// (N, 24, 3, 3)
    torch::Tensor sine = torch::sin(
        torch::unsqueeze(angles, 3).expand({BATCH_SIZE, JOINT_NUM, 3, 3})
    );// (N, 24, 3, 3)
    torch::Tensor cosine = torch::cos(
        torch::unsqueeze(angles, 3).expand({BATCH_SIZE, JOINT_NUM, 3, 3})
    );// (N, 24, 3, 3)
    torch::Tensor rotation = eye 
        + skew * sine 
        + torch::matmul(skew, skew) * (1 - cosine);// (N, 24, 3, 3)

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
 *      @poseBlendCoeffs: - Tensor -
 *          Pose blend coefficients for combining pose blend basis, (N, 207).
 * 
 */
torch::Tensor BlendShape::linRotMin() noexcept(false)
{
    //
    // unroll rotations
    //
    torch::Tensor unPoseRot, unRestPoseRot;
    try {
        unPoseRot = unroll(m__poseRot);// (N, 216)
        unRestPoseRot = unroll(m__restPoseRot);// (N, 216)
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // truncate rotations
    //
    unPoseRot = TorchEx::indexing(unPoseRot,
        torch::IntList(),
        torch::IntList({9, unPoseRot.size(1)})
    );// (N, 207)
    unRestPoseRot = TorchEx::indexing(unRestPoseRot,
        torch::IntList(),
        torch::IntList({9, unRestPoseRot.size(1)})
    );// (N, 207)
    
    //
    // pose blend coefficients
    //
    torch::Tensor poseBlendCoeffs = unPoseRot - unRestPoseRot;// (N, 207)

    return poseBlendCoeffs;
}

/**unroll
 * 
 * Brief
 * ----------
 * 
 *      Unroll rotation matrix into vector.
 * 
 * Arguments
 * ----------
 * 
 *      @rotation: - Tensor -
 *          Pose or rest pose rotation to be unrolled, (N, 24, 3, 3).
 * 
 * Return
 * ----------
 * 
 *      @unRotation: - Tensor -
 *          Unrolled pose or rest pose rotation, (N, 216).
 * 
 * 
 */
torch::Tensor BlendShape::unroll(torch::Tensor &rotation) 
    noexcept(false)
{
    if (rotation.sizes() !=
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        throw smpl_error("BlendShape", "Cannot unroll a rotation!");
    }

    torch::Tensor unRotation = torch::reshape(rotation, 
        {BATCH_SIZE, JOINT_NUM * 3 * 3});

    return unRotation;
}

//=============================================================================
} // namespace smpl
//=============================================================================
