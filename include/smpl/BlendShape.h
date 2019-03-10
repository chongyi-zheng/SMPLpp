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
#include <torch/torch.h>
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
 *      - m__device: <private>
 *          Torch device to run the module, could be CPUs or GPUs.
 * 
 *      - m__beta: <private>
 *          Batch of shape coefficient vectors, (N, 10).
 *
 *      - m__shapeBlendBasis: <private>
 *          Basis of the shape-dependent shape space,
 *          (6890, 3, 10).
 * 
 *      - m__shapeBlendShape: <private>
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - m__theta: <private>
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 *
 *      - m__restTheta: <private>
 *          Batch of rest pose in axis-angle representations, (N, 24, 3).
 *  
 *      - m__poseRot: <private>
 *          Rotation with respect to pose axis-angles representations,
 *          (N, 24, 3, 3).
 *
 *      - m__restPoseRot: <private>
 *          Pose rotation of rest pose, (N, 24, 3, 3).
 *
 *      - m__poseBlendBasis: <private>
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 *
 *      - m__poseBlendShape: <private>
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor
 *      %
 *      - BlendShape: <public>
 *          Default constructor.
 * 
 *      - BlendShape: (overload) <public>
 *          Constructor to initialize shape blend basis, pose blend basis, and
 *          torch device.
 *       
 *      - BlendShape: (overload) <public>
 *          Copy constructor.
 * 
 * 
 *      - ~BlendShape: <public>
 *          Destructor.
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
 *      - setDevice: <public>
 *          Set the torch device.
 * 
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

    torch::Device m__device;

    torch::Tensor m__beta;
    torch::Tensor m__shapeBlendBasis;
    torch::Tensor m__shapeBlendShape;

    torch::Tensor m__theta;
    torch::Tensor m__restTheta;
    torch::Tensor m__poseRot;
    torch::Tensor m__restPoseRot;
    torch::Tensor m__poseBlendBasis;
    torch::Tensor m__poseBlendShape;

protected: // PROTECTED ATTRIBUTES


public: // PUBLIC ATTRIBUTES


private: // PRIVATE METHODS

    // %% Blend Shape Generation %%
    void shapeBlend() noexcept(false);
    void poseBlend() noexcept(false);
    torch::Tensor rodrigues(torch::Tensor &theta) noexcept(false);
    torch::Tensor unroll(torch::Tensor &rotation) noexcept(false);
    torch::Tensor linRotMin() noexcept(false);

protected: // PROTECTED METHODS


public: // PUBLIC METHODS

    // %% Constructor and Destructor %%4
    BlendShape() noexcept(true);
    BlendShape(torch::Tensor &shapeBlendBasis,
        torch::Tensor &poseBlendBasis, torch::Device &device) noexcept(false);
    BlendShape(const BlendShape &blendShape) noexcept(false);
    ~BlendShape() noexcept(true);
    
    // %% Operators %%
    BlendShape &operator=(const BlendShape &blendShape) noexcept(false);

    // %% Blend Shape Wrapper %%
    void blend() noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setBeta(const torch::Tensor &beta) noexcept(false);
    void setShapeBlendBasis(const torch::Tensor &shapeBlendBasis)
        noexcept(false);

    void setTheta(const torch::Tensor &theta) noexcept(false);
    void setRestTheta(const torch::Tensor &restPostTheta) noexcept(true);    
    void setPoseBlendBasis(const torch::Tensor &poseBlendBasis)
        noexcept(false);

    torch::Tensor getShapeBlendShape() noexcept(false);
    torch::Tensor getPoseRotation() noexcept(false);
    torch::Tensor getRestPoseRotation() noexcept(false);
    torch::Tensor getPoseBlendShape() noexcept(false);

};

//=============================================================================
} // namespace SMPL
//=============================================================================
#endif // BLEND_SHAPE_H
//=============================================================================
