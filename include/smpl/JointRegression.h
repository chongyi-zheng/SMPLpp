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
//  CLASS JointRegression DECLARATIONS
//
//=============================================================================

#ifndef JOINT_REGRESSION_H
#define JOINT_REGRESSION_H

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
 *      Second of the four modules in SMPL pipeline.
 * 
 *      This class will regress vertex position into joint location of the new
 *      shape with different pose deformation considered.
 * 
 *      Formulas (6) and (10) will be implemented.
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
 *      - m__shapeBlendShape: <private>
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - m__poseBlendShape: <private>
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - m__templateRestShape: <private>
 *          Template shape in rest pose, (6890, 3).
 * 
 *      - m__joints: <private>
 *          Joint locations of the deformed shape, (N, 24, 3).
 * 
 *      - m__jointRegressor: <private>
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor.
 *      %
 *      - JointRegression: <public>
 *          Default constructor.
 * 
 *      - JointRegression: (overload) <public>
 *          Constructor to initialize joint regressor, template shape, and torch
 *          device.
 * 
 *      - JointRegression: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~JointRegression: <public>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <JointRegression> instantiation.
 *      %%
 * 
 *      %
 *          Setter and Getter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      - setShapeBlendShape: <public>
 *          Set shape blend shape.
 * 
 *      - setPoseBlendShape: <public>
 *          Set pose blend shape.
 * 
 *      - setTemplateRestShape: <public>
 *          Set template shape in rest pose. We will combine this one with
 *          shape blend shape and pose blend shape to get the shape for new
 *          pose.
 * 
 *      - setJointRegressor: <public>
 *          Set the joint coefficients.
 * 
 *      - getRestShape: <public>
 *          Get the deformed shape in rest pose.
 * 
 *      - getJoint: <public>
 *          Get global joint locations.
 * 
 *      %%
 * 
 *      %
 *          Joint Regression Wrapper
 *      %
 *      - regress: <public>
 *          Outside wrapper to encapsulate joint regression process.
 *      %%
 * 
 *      %
 *          Regression
 *      %
 *      - linearCombine: <private>
 *          Linearly combining pose-dependent shape and shape-dependent shape
 *          with template shape.
 * 
 *      - jointRegress: <private>
 *          Regress the rest shape for new pose into joints.
 *      %%
 * 
 * 
 */

class JointRegression final
{

private: // PRIVATE ATTRIBUTES

    torch::Device m__device;

    torch::Tensor m__restShape;
    torch::Tensor m__shapeBlendShape;
    torch::Tensor m__poseBlendShape;
    torch::Tensor m__templateRestShape;

    torch::Tensor m__joints;
    torch::Tensor m__jointRegressor;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    void linearCombine() noexcept(false);
    void jointRegress() noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    JointRegression() noexcept(true);
    JointRegression(torch::Tensor &jointRegressor, 
        torch::Tensor &templateRestShape, 
        torch::Device &device) noexcept(false);
    JointRegression(const JointRegression &jointRegression) noexcept(false);
    ~JointRegression() noexcept(true);

    // %% Operators %%
    JointRegression &operator=(const JointRegression &jointRegression)
        noexcept(false);
    
    // %% Getter and Setter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setShapeBlendShape(const torch::Tensor &shapeBlendShape) 
        noexcept(false);
    void setPoseBlendShape(const torch::Tensor &poseBlendShape)
        noexcept(false);
    void setTemplateRestShape(const torch::Tensor &templateRestShape)
        noexcept(false);
    void setJointRegressor(const torch::Tensor &jointRegressor)
        noexcept(false);
    
    torch::Tensor getRestShape() noexcept(false);
    torch::Tensor getJoint() noexcept(false);

    // %% Joint Regression Wrapper %%
    void regress() noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // JOINT_REGRESSION_H
//=============================================================================
