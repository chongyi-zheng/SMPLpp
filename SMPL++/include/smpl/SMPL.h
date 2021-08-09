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
//  CLASS SMPL DECLARATIONS
//
//=============================================================================

#ifndef SMPL_H
#define SMPL_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <string>
//----------
#include <nlohmann/json.hpp>
#ifdef TARGET_IOS
#include <LibTorch-Lite.h>
#else
#include <torch/torch.h>
#endif // TARGET_IOS
//----------
#include "smpl/BlendShape.h"
#include "smpl/JointRegression.h"
#include "smpl/WorldTransformation.h"
#include "smpl/LinearBlendSkinning.h"
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
 *      The final system to combine all modules and make them work properly.
 * 
 *      This class is system wrapper which dose real computation. The actually
 *      working process is consistent with the SMPL pipeline.
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - m__device: <private>
 *          Torch device to run the module, could be CPUs or GPUs.
 * 
 *      - m__modelPath: <private>
 *          Path to the JSON model file.
 * 
 *      - m__vertPath: <private>
 *          Path to store the mesh OBJ file.
 * 
 *      - m__faceIndices: <private>
 *          Vertex indices of each face, (13776, 3)
 * 
 *      - m__shapeBlendBasis: <private>
 *          Basis of the shape-dependent shape space,
 *          (6890, 3, 10).
 * 
 *      - m__poseBlendBasis: <private>
 *          Basis of the pose-dependent shape space, (6890, 3, 207).
 * 
 *      - m__templateRestShape: <private>
 *          Template shape in rest pose, (6890, 3).
 * 
 *      - m__jointRegressor: <private>
 *          Joint coefficients of each vertices for regressing them to joint
 *          locations, (24, 6890).
 * 
 *      - m__kinematicTree: <private>
 *          Hierarchy relation between joints, the root is at the belly button,
 *          (2, 24).
 * 
 *      - m__weights: <private>
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      - m__model: <private>
 *          JSON object represents.
 * 
 *      - m__blender: <private>
 *          A module to generate shape blend shape and pose blend shape
 *          by combining parameters thetas and betas with their own basis.
 * 
 *      - m__regressor: <private>
 *          A module to regress vertex position into joint location of the new
 *          shape with different pose deformation considered.
 * 
 *      - m__transformer: <private>
 *          A module to transform joints from T-pose's position into the ones 
 *          of new pose.
 * 
 *      - m__skinner: <private>
 *          A module to do linear blend skinning.
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor.
 *      %
 *      - SMPL: <public>
 *          Default constructor.
 * 
 *      - SMPL: <public>
 *          Constructor to initialize model path, vertex path, and torch 
 *          device.
 * 
 *      - SMPL: <public>
 *          Copy constructor.
 * 
 *      - ~SMPL: <public>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operator
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <SMPL> instantiation.
 *      %%
 * 
 *      %
 *          Setter and Getter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      - setPath: <public>
 *          Set model path to the JSON model file.
 * 
 *      - getRestShape: <public>
 *          Get deformed shape in rest pose.
 * 
 *      - getFaceIndex: <public>
 *          Get vertex indices of each face.
 * 
 *      - getRestJoint: <public>
 *          Get joint locations of the deformed shape in rest pose.
 * 
 *      - getVertex: <public>
 *          Get vertex locations of the deformed mesh.
 *      %%
 * 
 *      %
 *          Modeling
 *      %
 *      - init: <public>
 *          Load model data stored as JSON file into current application.
 *          (Note: The loading will spend a long time because of a large
 *           JSON file.)
 * 
 *      - launch: <public>
 *          Run the model with a specific group of beta, theta, and 
 *          translation.
 * 
 *      - out: <public>
 *          Export the deformed mesh to OBJ file.
 *      %%
 * 
 */

class SMPL final
{

private: // PIRVATE ATTRIBUTES

    torch::Device m__device;

    std::string m__modelPath;
    std::string m__vertPath;
    nlohmann::json m__model;

    torch::Tensor m__faceIndices;
    torch::Tensor m__shapeBlendBasis;
    torch::Tensor m__poseBlendBasis;
    torch::Tensor m__templateRestShape;
    torch::Tensor m__jointRegressor;
    torch::Tensor m__kinematicTree;
    torch::Tensor m__weights;

    BlendShape m__blender;
    JointRegression m__regressor;
    WorldTransformation m__transformer;
    LinearBlendSkinning m__skinner;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    SMPL() noexcept(true);
    SMPL(std::string &modelPath, 
        std::string &vertPath, torch::Device &device) noexcept(false);
    SMPL(const SMPL& smpl) noexcept(false);
    ~SMPL() noexcept(true);

    // %% Operators %%
    SMPL &operator=(const SMPL& smpl) noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setModelPath(const std::string &modelPath) noexcept(false);
    void setVertPath(const std::string &vertexPath) noexcept(false);

    torch::Tensor getRestShape() noexcept(false);
    torch::Tensor getFaceIndex() noexcept(false);
    torch::Tensor getRestJoint() noexcept(false);
    torch::Tensor getVertex() noexcept(false);
    torch::Tensor getExtra() noexcept;

    // %% Modeling %%
    void init() noexcept(false);
    void launch(
        torch::Tensor &beta,
        torch::Tensor &theta,
        std::optional<torch::Tensor> &extra) noexcept(false);
    void out(int64_t index) noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // SMPL_H
//=============================================================================
