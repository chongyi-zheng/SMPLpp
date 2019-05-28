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
    m__device(torch::kCPU),
    m__restShape(),
    m__shapeBlendShape(),
    m__poseBlendShape(),
    m__templateRestShape(),
    m__joints(),
    m__jointRegressor()
{
}

/**JointRegression (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize joint regressor, template shape, and torch 
 *      device.
 * 
 * Arguments
 * ----------
 * 
 *      @jointRegressor: - Tensor -
 *          The joint coefficients, (24, 6890).
 * 
 *      @templateRestShape: - Tensor -
 *          The template shape in rest pose, (6890, 3).
 * 
 *      @device: - Device -
 *          Torch device to run the module, CPUs or GPUs.
 * 
 * Return
 * ----------
 * 
 * 
 */
JointRegression::JointRegression(torch::Tensor &jointRegressor,
    torch::Tensor &templateRestShape, torch::Device &device) noexcept(false) :
    m__device(torch::kCPU),
    m__restShape(),
    m__shapeBlendShape(),
    m__poseBlendShape(),
    m__joints()
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("JointRegression", "Failed to fetch device index!");
    }

    if (jointRegressor.sizes() == 
        torch::IntArrayRef({JOINT_NUM, VERTEX_NUM})) {
        m__jointRegressor = jointRegressor.clone().to(m__device);
    }
    else {
        throw smpl_error("JointRegression", 
            "Failed to initialize joint regressor!");
    }

    if (templateRestShape.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__templateRestShape = templateRestShape.clone().to(m__device);
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
 *      @jointRegression: - Tensor -
 *          The <JointRegression> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
JointRegression::JointRegression(const JointRegression &jointRegression)
    noexcept(false) :
    m__device(torch::kCPU),
    m__restShape(),
    m__joints()
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
 *      @jointRegression: - Tensor -
 *          The <JointRegression> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - Tensor -
 *          Current instantiation.
 * 
 */
JointRegression &JointRegression::operator=(
    const JointRegression &jointRegression) noexcept(false)
{
    //
    // hard copy
    //
    if (jointRegression.m__device.has_index()) {
        m__device = jointRegression.m__device;
    }
    else {
        throw smpl_error("JointRegression", "Failed to fetch device index!");
    }

    if (jointRegression.m__shapeBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__shapeBlendShape = jointRegression.m__shapeBlendShape.clone().to(
            m__device);
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy shape blend shape!");
    }

    if (jointRegression.m__poseBlendShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__poseBlendShape = jointRegression.m__poseBlendShape.clone().to(
            m__device);
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy pose blend shape!");
    }

    if (jointRegression.m__templateRestShape.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__templateRestShape = jointRegression.m__templateRestShape.clone().to(
            m__device);
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy template shape!");
    }

    if (jointRegression.m__jointRegressor.sizes() == 
        torch::IntArrayRef({JOINT_NUM, VERTEX_NUM})) {
        m__jointRegressor = jointRegression.m__jointRegressor.clone().to(
            m__device);
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy joint regressor!");
    }

    //
    // soft copy
    //
    if (jointRegression.m__restShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__restShape = jointRegression.m__restShape.clone().to(m__device);
    }

    if (jointRegression.m__joints.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        m__joints = jointRegression.m__joints.clone().to(m__device);
    }
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
void JointRegression::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("JointRegression", "Failed to fetch device index!");
    }

    return;
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
 *      @shapeBlendShape: - const Tensor & -
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setShapeBlendShape(const torch::Tensor &shapeBlendShape)
    noexcept(false)
{
    if (shapeBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__shapeBlendShape = shapeBlendShape.clone().to(m__device);
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
 *      @poseBlendShape: - const Tensor & -
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setPoseBlendShape(const torch::Tensor &poseBlendShape)
    noexcept(false)
{
    if (poseBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__poseBlendShape = poseBlendShape.clone().to(m__device);
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
 *      @templateRestShape: - const Tensor & -
 *          Template shape in rest pose, (6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setTemplateRestShape(
    const torch::Tensor &templateRestShape) noexcept(false)
{
    if (templateRestShape.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__templateRestShape = templateRestShape.clone().to(m__device);
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
 *      @jointRegressor: - const Tensor & -
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 * Return
 * ----------
 * 
 * 
 */
void JointRegression::setJointRegressor(const torch::Tensor &jointRegressor)
    noexcept(false)
{
    if (jointRegressor.sizes() ==
        torch::IntArrayRef({JOINT_NUM, VERTEX_NUM})) {
        m__jointRegressor = jointRegressor.clone().to(m__device);
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
 *      @restShape: - Tensor -
 *          Deformed shape in rest pose, (N, 6890, 3).     
 * 
 */
torch::Tensor JointRegression::getRestShape() noexcept(false)
{
    torch::Tensor restShape;

    if (m__restShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        restShape = m__restShape.clone().to(m__device);
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
 *      @joints: - Tensor -
 *          Joint locations of the deformed shape, (N, 24, 3).
 * 
 */
torch::Tensor JointRegression::getJoint() noexcept(false)
{
    torch::Tensor joints;

    if (m__joints.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {
        joints = m__joints.clone().to(m__device);
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
    if (m__shapeBlendShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})
        && m__poseBlendShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})
        && m__templateRestShape.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__restShape = m__templateRestShape 
            + m__shapeBlendShape 
            + m__poseBlendShape;
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
    if (m__shapeBlendShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})
        && m__templateRestShape.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        torch::Tensor blendShape = m__templateRestShape + 
            m__shapeBlendShape;// (N, 6890, 3)
        m__joints = torch::tensordot(blendShape, m__jointRegressor, 
            {1}, {1});// (N, 3, 24)
        m__joints = torch::transpose(m__joints, 1, 2);// (N, 24, 3)
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
