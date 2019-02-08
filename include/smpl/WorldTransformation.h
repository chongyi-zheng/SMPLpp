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
//  CLASS WorldTransformation DECLARATIONS
//
//=============================================================================

#ifndef WORLD_TRANSFORMATION_H
#define WORLD_TRANSFORMATION_H

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
 *      Third of the four modules in SMPL pipeline.
 * 
 *      This class will transform joints from T-pose's position into the ones 
 *      of new pose. Each transformation corresponds to an exact bone, and we
 *      will use them to linearly blend vertices of the whole body.
 * 
 *      Formulas (3) and (4) are implemented.
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - __joints: <private>
 *          Joint locations of the deformed shape after regressing, (N, 24, 3).
 * 
 *      - __poseRot: <private>
 *          Rotations with respect to new pose by axis-angles
 *          representations, (N, 24, 3, 3).
 * 
 *      - __kineTree: <private>
 *          Hierarchy relation between joints, the root is at the belly button,
 *          (2, 24).
 * 
 *      - __transformations: <private>
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Deconstructor
 *      %
 *      - WorldTransformation: <public>
 *          Default constructor.
 * 
 *      - WorldTransformation: (overload) <public>
 *          Constructor to initialize kinematicTree.
 * 
 *      - WorldTransformation: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~WorldTransformation: <public>
 *          Deconstructor.
 * 
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <WorldTransformation> instantiation.
 *      %%
 * 
 *      %
 *          Getter and Setter
 *      %
 *      - setJoint:
 *          Set joint locations of the deformed shape.
 * 
 *      - setPoseRotation:
 *          Set pose rotations by axis-angles representations.
 *          
 *      - setKinematicTree:
 *          Set kinematice tree of the body.
 * 
 *      - getTransformation:
 *          Get world transformations.
 *          
 *      %%
 * 
 *      %
 *          Transformation wrapper
 *      %
 *      - transform: <public>
 *          Outside wrapper to encapsulate world transformation process.
 *      %%
 * 
 *      %
 *          Transformations
 *      %
 *      - localTransform: <private>
 *          Local transformations with respect to each joint.
 * 
 *      - globalTransform: <private>
 *          Combine local transformation according to kinematic tree to get
 *          global transformation of each bone.
 * 
 *      - relativeTransform: <private>
 *          Eliminate rest pose's transformation from global transformation.
 *      %%
 * 
 * 
 */

class WorldTransformation final
{

private: // PIRVATE ATTRIBUTES

    xt::xarray<double> __joints;
    xt::xarray<double> __poseRot;
    xt::xarray<uint32_t> __kineTree;
    xt::xarray<double> __transformations;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    // %% Transformations %%
    xt::xarray<double> localTransform(xt::xarray<double> poseRotHomo) 
        noexcept(false);
    xt::xarray<double> globalTransform(xt::xarray<double> localTransformations)
        noexcept(false);
    void relativeTransform(xt::xarray<double> globalTransformations)
        noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%
    WorldTransformation() noexcept(true);
    WorldTransformation(xt::xarray<double> kineTree) noexcept(false);
    WorldTransformation(const WorldTransformation &worldTransformation)
        noexcept(false);
    ~WorldTransformation() noexcept(true);

    // %% Operators %%
    WorldTransformation &operator=(
        const WorldTransformation &worldTransformation) noexcept(false);

    // %% Setter and Getter %%
    void setJoint(xt::xarray<double> joints) noexcept(false);
    void setPoseRotation(xt::xarray<double> poseRot) noexcept(false);
    void setKinematicTree(xt::xarray<uint32_t> kineTree) noexcept(false);

    xt::xarray<double> getTransformation() noexcept(false);
    // %% Transformation wrapper %%
    void transform();

};

//=============================================================================
}
//=============================================================================
#endif // WORLD_TRANSFORMATION_H
//=============================================================================