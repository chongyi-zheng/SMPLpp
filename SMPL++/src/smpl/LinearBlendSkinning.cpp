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
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/TorchEx.hpp"
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
    m__device(torch::kCPU),
    m__restShape(),
    m__transformation(),
    m__weights(),
    m__posedVert()
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
 *      @weights: - Tensor -
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      @device: - Device -
 *          Torch device to run the module, CPUs or GPUs.
 * 
 * Return
 * ----------
 * 
 * 
 */
LinearBlendSkinning::LinearBlendSkinning(torch::Tensor &weights,
    torch::Device &device) 
    noexcept(false) :
    m__device(torch::kCPU)
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to fetch device index!");
    }

    if (weights.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, JOINT_NUM})) {
        m__weights = weights.clone().to(m__device);
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
    m__device(torch::kCPU),
    m__posedVert()
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
 *      @linearBlendSkinning: - Tensor -
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
    if (linearBlendSkinning.m__device.has_index()) {
        m__device = linearBlendSkinning.m__device;
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to fetch device index!");
    }

    if (linearBlendSkinning.m__restShape.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__restShape = linearBlendSkinning.m__restShape.clone().to(m__device);
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to copy deformed shape in rest pose!");
    }

    if (linearBlendSkinning.m__transformation.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        m__transformation = linearBlendSkinning.m__transformation.clone().to(
            m__device);
    }
    else {
        throw smpl_error("LinearBlendSkinning",
            "Failed to copy world transformation!");
    }

    if (linearBlendSkinning.m__weights.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, JOINT_NUM})) {
        m__weights = linearBlendSkinning.m__weights.clone().to(m__device);
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to copy linear blend weights!");
    }

    //
    // soft copy
    //
    if (linearBlendSkinning.m__posedVert.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__posedVert = linearBlendSkinning.m__posedVert.clone().to(m__device);
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
void LinearBlendSkinning::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("LinearBlendSkinning", "Failed to fetch device index!");
    }

    return;
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
 *      @restShape: - Tensor -
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setRestShape(
    const torch::Tensor &restShape) noexcept(false)
{
    if (restShape.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        m__restShape = restShape.clone().to(m__device);
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
 *      weights: - Tensor -
 *          Weights for linear blend skinning, (6890, 24).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setWeight(
    const torch::Tensor &weights) noexcept(false)
{
    if (weights.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, JOINT_NUM})) {
        m__weights = weights.clone().to(m__device);
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
 *      @transformation: - Tensor -
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::setTransformation(
    const torch::Tensor &transformation) noexcept(false)
{
    if (transformation.sizes() ==
        torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        m__transformation = transformation.clone().to(m__device);
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
 *      @vertices: - Tensor -
 *          Vertex locations of the new pose, (N, 6890, 3).
 * 
 */
torch::Tensor LinearBlendSkinning::getVertex() noexcept(false)
{
    torch::Tensor vertices;
    if (m__posedVert.sizes() == 
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        vertices = m__posedVert.clone().to(m__device);
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
    torch::Tensor restShapeHomo;
    try {
        restShapeHomo = cart2homo(m__restShape);// (N, 6890, 4)
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // linear blend skinning
    //
    torch::Tensor coefficients = torch::tensordot(
        m__weights, m__transformation, {1}, {1});// (6890, N, 4, 4)
    coefficients = torch::transpose(coefficients, 0, 1);// (N, 6890, 4, 4)
    restShapeHomo = torch::unsqueeze(restShapeHomo, 3);// (N, 6890, 4, 1)
    torch::Tensor verticesHomo = torch::matmul(
        coefficients, restShapeHomo);// (N, 6890, 4, 1)
    verticesHomo = torch::squeeze(verticesHomo, 3);// (N, 6890, 4)

    //
    // homogeneous coordinates to Cartesian coordinates
    //
    try {
        m__posedVert = homo2cart(verticesHomo);
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
 *      @cart: - Tensor -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 *      @homo: - Tensor -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 */
torch::Tensor LinearBlendSkinning::cart2homo(torch::Tensor &cart) 
    noexcept(false)
{
    if (cart.sizes() !=
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 3})) {
        throw smpl_error("LinearBlendSkinning",
            "Cannot convert Cartesian coordinates to homogeneous one!");
    }

    torch::Tensor ones = torch::ones(
        {BATCH_SIZE, VERTEX_NUM, 1}, m__device);// (N, 6890, 1)
    torch::Tensor homo = torch::cat({cart, ones}, 2);// (N, 6890, 4)

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
 *      @homo: - Tensor -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 * Return
 * ----------
 * 
 *      @cart: - Tensor -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 */
torch::Tensor LinearBlendSkinning::homo2cart(torch::Tensor &homo) 
    noexcept(false)
{
    if (homo.sizes() !=
        torch::IntArrayRef({BATCH_SIZE, VERTEX_NUM, 4})) {
        throw smpl_error("LinearBlendSkinning",
            "Cannot convert homogeneous coordinates to Cartesian ones!");
    }

    torch::Tensor homoW = TorchEx::indexing(homo,
        torch::IntList(),
        torch::IntList(),
        torch::IntList({3}));// (N, 6890)
    homoW = torch::unsqueeze(homoW, 2);// (N, 6890, 1)
    torch::Tensor homoUnit = homo / homoW;// (N, 6890, 4)
    torch::Tensor cart = TorchEx::indexing(homoUnit,
        torch::IntList(), 
        torch::IntList(), 
        torch::IntList({0, 3}));// (N, 6890, 3)
    
    return cart;
}

//=============================================================================
} // namespace smpl
//=============================================================================
