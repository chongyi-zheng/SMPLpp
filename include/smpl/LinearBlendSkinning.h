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

#include <xtensor/xarray.hpp>

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
 *      - __restShape: <private>
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 *      - __transformation: <private>
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 *      - __weights: <private>
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      - __posedVert: <private>
 *           Vertex locations of the new pose, (N, 6890, 3).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Deconstructor
 *      %
 *      - LinearBlendSkinning: <public>
 *          Default constructor.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Constructor to initialize weights for linear blend skinning.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~LinearBlendSkinning: <public>
 *          Deconstructor.
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

    xt::xarray<double> __restShape;
    xt::xarray<double> __transformation;
    xt::xarray<double> __weights;
    xt::xarray<double> __posedVert;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    // %% Linear Blend Skinning %%
    xt::xarray<double> cart2homo(xt::xarray<double> cart) noexcept(false);
    xt::xarray<double> homo2cart(xt::xarray<double> homo) noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%
    LinearBlendSkinning() noexcept(true);
    LinearBlendSkinning(xt::xarray<double> weights) noexcept(false);
    LinearBlendSkinning(const LinearBlendSkinning &linearBlendSkinning)
        noexcept(false);
    ~LinearBlendSkinning() noexcept(true);

    // %% Operator %%
    LinearBlendSkinning &operator=(
        const LinearBlendSkinning &linearBlendSkinning) noexcept(false);

    // %% Setter and Getter %%
    void setWeight(xt::xarray<double> weights) noexcept(false);
    void setRestShape(xt::xarray<double> restShape) noexcept(false);
    void setTransformation(xt::xarray<double> transformation) noexcept(false);

    xt::xarray<double> getVertex() noexcept(false);

    // %% Linear Blend Skinning %%
    void skinning() noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // LINEAR_BLEND_SKINNING_H
//=============================================================================
