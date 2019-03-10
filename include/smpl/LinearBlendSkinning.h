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
//  CLASS LinearBlendSkinning DECLARATIONS
//
//=============================================================================

#ifndef LINEAR_BLEND_SKINNING_H
#define LINEAR_BLEND_SKINNING_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

#include <torch/torch.h>

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/**
 * DESCRIPTIONS:
 * 
 *      Fourth of the four modules in SMPL pipeline.
 * 
 *      This class finally implement pose change of the model. We will apply
 *      the most popular skinning method - linear blend skinning - to all
 *      vertices. As its definition, linear blend skinning combines
 *      transformations of different near bones together to get the finally
 *      transformation. Indeed, this one doesn't guarantee a rigid
 *      transformation any more. Hopefully, we may use more sophistatic
 *      skinning model like dual quaternion skinning to fine tune the pose.
 * 
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - m__device: <private>
 *          Torch device to run the module, could be CPUs or GPUs.
 * 
 *      - m__restShape: <private>
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 *      - m__transformation: <private>
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 *      - m__weights: <private>
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      - m__posedVert: <private>
 *           Vertex locations of the new pose, (N, 6890, 3).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor
 *      %
 *      - LinearBlendSkinning: <public>
 *          Default constructor.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Constructor to initialize weights and torch device for linear 
 *          blend skinning.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~LinearBlendSkinning: <public>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <LinearBlendSkinning> instantiation.
 *      %%
 * 
 *      %
 *          Getter and Setter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      - setRestShape: <public>
 *          Set the deformed shape in rest pose.
 * 
 *      - setWeight: <public>
 *          Set the weights for linear blend skinning.
 * 
 *      - setTransformation: <public>
 *          Set the world transformation.
 * 
 *      - getVertex: <public>
 *          Get vertex locations of the new pose.
 * 
 *      %%
 * 
 *      %
 *          Linear Blend Skinning
 *      %
 *      - skinning: <public>
 *          Do all the skinning stuffs.
 * 
 *      - cart2homo: <private>
 *          Convert Cartesian coordinates to homogeneous coordinates.
 * 
 *      - homo2cart: <private>
 *          Convert homogeneous coordinates to Cartesian coordinates.
 *      %%
 * 
 * 
 */

class LinearBlendSkinning final
{

private: // PIRVATE ATTRIBUTES

    torch::Device m__device;

    torch::Tensor m__restShape;
    torch::Tensor m__transformation;
    torch::Tensor m__weights;
    torch::Tensor m__posedVert;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    // %% Linear Blend Skinning %%
    torch::Tensor cart2homo(torch::Tensor &cart) noexcept(false);
    torch::Tensor homo2cart(torch::Tensor &homo) noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    LinearBlendSkinning() noexcept(true);
    LinearBlendSkinning(torch::Tensor &weights, 
        torch::Device &device) noexcept(false);
    LinearBlendSkinning(const LinearBlendSkinning &linearBlendSkinning)
        noexcept(false);
    ~LinearBlendSkinning() noexcept(true);

    // %% Operator %%
    LinearBlendSkinning &operator=(
        const LinearBlendSkinning &linearBlendSkinning) noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setWeight(const torch::Tensor &weights) noexcept(false);
    void setRestShape(const torch::Tensor &restShape) noexcept(false);
    void setTransformation(
        const torch::Tensor &transformation) noexcept(false);

    torch::Tensor getVertex() noexcept(false);

    // %% Linear Blend Skinning %%
    void skinning() noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // LINEAR_BLEND_SKINNING_H
//=============================================================================
