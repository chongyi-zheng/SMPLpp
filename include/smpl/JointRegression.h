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
 *      - __restShape: <private>
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 *      - __shapeBlendShape: <private>
 *          Shape blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - __poseBlendShape: <private>
 *          Pose blend shape of SMPL model, (N, 6890, 3).
 * 
 *      - __templateRestShape: <private>
 *          Template shape in rest pose, (6890, 3).
 * 
 *      - __joints: <private>
 *          Joint locations of the deformed shape, (N, 24, 3).
 * 
 *      - __jointRegressor: <private>
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and deconstructor.
 *      %
 *      - JointRegression: <public>
 *          Default constructor.
 * 
 *      - JointRegression: (overload) <public>
 *          Constructor to initialize joint regressor and template shape.
 * 
 *      - JointRegression: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~JointRegression: <public>
 *          Deconstructor.
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

    xt::xarray<double> __restShape;
    xt::xarray<double> __shapeBlendShape;
    xt::xarray<double> __poseBlendShape;
    xt::xarray<double> __templateRestShape;

    xt::xarray<double> __joints;
    xt::xarray<double> __jointRegressor;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    void linearCombine() noexcept(false);
    void jointRegress() noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%
    JointRegression() noexcept(true);
    JointRegression(xt::xarray<double> jointRegressor, 
        xt::xarray<double> templateRestShape) noexcept(false);
    JointRegression(const JointRegression &jointRegression) noexcept(false);
    ~JointRegression() noexcept(true);

    // %% Operators %%
    JointRegression &operator=(const JointRegression &jointRegression)
        noexcept(false);
    
    // %% Getter and Setter %%
    void setShapeBlendShape(xt::xarray<double> shapeBlendShape) 
        noexcept(false);
    void setPoseBlendShape(xt::xarray<double> poseBlendShape)
        noexcept(false);
    void setTemplateRestShape(xt::xarray<double> templateRestShape)
        noexcept(false);
    void setJointRegressor(xt::xarray<double> jointRegressor)
        noexcept(false);
    
    xt::xarray<double> getRestShape() noexcept(false);
    xt::xarray<double> getJoint() noexcept(false);

    // %% Joint Regression Wrapper %%
    void regress() noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // JOINT_REGRESSION_H
//=============================================================================
