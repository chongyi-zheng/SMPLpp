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
//  CLASS BlendShape DECLARATIONS
//
//=============================================================================

#ifndef BLEND_SHAPE_H
#define BLEND_SHAPE_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <xtensor/xarray.hpp>
//----------
#include "toolbox/Exception.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/** 
 * DESCRIPTIONS:
 * 
 *      First of the four modules in SMPL pipeline.
 * 
 *      This class is used to generate shape blend shape and pose blend shape
 *      by combining parameters thetas and betas with their own basis.
 * 
 *      Formulas (1), (8), and (9) are implemented here.
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 *
 *      - __beta: <private>
 *          Batch of shape coefficient vectors, (N, 10).
 *
 *      - __shapeBlendBasis: <private>
 *          Basis of the shape-dependent shape space,
 *          (6890, 3, 10).
 * 
 *      - __shapeBlendShape: <private>
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - __theta: <private>
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 *
 *      - __restTheta: <private>
 *          Batch of rest pose in axis-angle representations, (N, 24, 3).
 *  
 *      - __poseRot: <private>
 *          Rotation with respect to pose axis-angles representations,
 *          (N, 24, 3, 3).
 *
 *      - __restPoseRot: <private>
 *          Pose rotation of rest pose, (N, 24, 3, 3).
 *
 *      - __poseBlendBasis: <private>
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 *
 *      - __poseBlendShape: <private>
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Deconstructor
 *      %
 *      - BlendShape: <public>
 *          Default constructor.
 * 
 *      - BlendShape: (overload) <public>
 *          Constructor to initialize both shape and pose basis.
 *       
 *      - BlendShape: (overload) <public>
 *          Copy constructor.
 * 
 * 
 *      - ~BlendShape: <public>
 *          Deconstructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <BlendShape> instantiation.
 *      %%
 *      
 *      %
 *          Setter and Getter
 *      %
 *      - setBeta: <public>
 *          Set shape coefficient vector.
 *
 *      - setShapeBlendBasis: <public>
 *          Set shape blend basis.
 * 
 *      - setTheta: <public>
 *          Set new pose in axis-angle representation.
 *      
 *      - setRestTheta: <public>
 *          Set rest pose rotations in axis-angle representation.
 * 
 *      - setPoseBlendBasis: <public>
 *          Set pose blend basis.
 *
 *      - getShapeBlendShape: <public>
 *          Get shape blend shape.
 *  
 *      - getPoseRotation: <public>
 *          Get pose rotation matrix.
 *
 *      - getRestPoseRotation: <public>
 *          Get rest pose rotation matrix.
 * 
 *      - getPoseBlendShape: <public>
 *          Get pose blend shape.
 * 
 *      %%
 *      
 *      %
 *          Blend Shape Wrapper
 *      %
 *      - blend: <public>
 *          Outside monitor to generate blend shape.
 *      %%
 * 
 *      %
 *          Blend Shape Generation
 *      %
 *      - shapeBlend: <private>
 *          Generate shape blend shape.
 *
 *      - poseBlend: <private>
 *          Generate pose blend shape.
 * 
 *      - rodrigues: <private>
 *          Get arbitrary rotations in axis-angle representations using
 *          Rodrigues' formula.
 *      
 *      - unroll: <private>
 *          Unroll rotation matrix into vector.
 *
 *      - linRotMin: <private>
 *          Eliminate the influence of rest pose on pose blend shape and
 *          generate pose blend coefficients (linear rotation minimization).
 *      %%
 * 
 */

class BlendShape final
{

private: // PRIVATE ATTRIBUTES

    xt::xarray<double> __beta;
    xt::xarray<double> __shapeBlendBasis;
    xt::xarray<double> __shapeBlendShape;

    xt::xarray<double> __theta;
    xt::xarray<double> __restTheta;
    xt::xarray<double> __poseRot;
    xt::xarray<double> __restPoseRot;
    xt::xarray<double> __poseBlendBasis;
    xt::xarray<double> __poseBlendShape;

protected: // PROTECTED ATTRIBUTES


public: // PUBLIC ATTRIBUTES


private: // PRIVATE METHODS

    // %% Blend Shape Generation %%
    void shapeBlend() noexcept(false);
    void poseBlend() noexcept(false);
    xt::xarray<double> rodrigues(xt::xarray<double> theta) noexcept(false);
    xt::xarray<double> unroll(xt::xarray<double> rotation) noexcept(false);
    xt::xarray<double> linRotMin() noexcept(false);

protected: // PROTECTED METHODS


public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%4
    BlendShape() noexcept(true);
    BlendShape(xt::xarray<double> shapeBlendBasis,
        xt::xarray<double> poseBlendBasis) noexcept(false);
    BlendShape(const BlendShape &blendShape) noexcept(false);
    ~BlendShape() noexcept(true);
    
    // %% Operators %%
    BlendShape &operator=(const BlendShape &blendShape) noexcept(false);

    // %% Blend Shape Wrapper %%
    void blend() noexcept(false);

    // %% Setter and Getter %%
    void setBeta(xt::xarray<double> beta) noexcept(false);
    void setShapeBlendBasis(xt::xarray<double> shapeBlendBasis)
        noexcept(false);

    void setTheta(xt::xarray<double> theta) noexcept(false);
    void setRestTheta(xt::xarray<double> restPostTheta) noexcept(true);    
    void setPoseBlendBasis(xt::xarray<double> poseBlendBasis)
        noexcept(false);

    xt::xarray<double> getShapeBlendShape() noexcept(false);
    xt::xarray<double> getPoseRotation() noexcept(false);
    xt::xarray<double> getRestPoseRotation() noexcept(false);
    xt::xarray<double> getPoseBlendShape() noexcept(false);

};

//=============================================================================
} // namespace SMPL
//=============================================================================
#endif // BLEND_SHAPE_H
//=============================================================================
