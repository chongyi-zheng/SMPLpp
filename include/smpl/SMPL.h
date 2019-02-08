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
#include <xtensor/xarray.hpp>
#include <xtensor/xjson.hpp>
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
 *      - __modelPath: <private>
 *          Path to the JSON model file.
 * 
 *      - __vertPath: <private>
 *          Path to store the mesh OBJ file.
 * 
 *      - __model: <private>
 *          JSON object represents.
 * 
 *      - __blender: <private>
 *          A module to generate shape blend shape and pose blend shape
 *          by combining parameters thetas and betas with their own basis.
 * 
 *      - __regressor: <private>
 *          A module to regress vertex position into joint location of the new
 *          shape with different pose deformation considered.
 * 
 *      - __transformer: <private>
 *          A module to transform joints from T-pose's position into the ones 
 *          of new pose.
 * 
 *      - __skinner: <private>
 *          A module to do linear blend skinning.
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Deconstructor.
 *      %
 *      - SMPL: <public>
 *          Default constructor.
 * 
 *      - SMPL: <public>
 *          Constructor to initialize model path.
 * 
 *      - SMPL: <public>
 *          Copy constructor.
 * 
 *      - ~SMPL: <public>
 *          Deconstructor.
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

    std::string __modelPath;
    std::string __vertPath;
    nlohmann::json __model;

    BlendShape __blender;
    JointRegression __regressor;
    WorldTransformation __transformer;
    LinearBlendSkinning __skinner;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%
    SMPL() noexcept(true);
    SMPL(std::string modelPath, std::string vertPath) noexcept(false);
    SMPL(const SMPL& smpl) noexcept(false);
    ~SMPL() noexcept(true);

    // %% Operators %%
    SMPL &operator=(const SMPL& smpl) noexcept(false);

    // %% Setter and Getter %%
    void setModelPath(std::string modelPath) noexcept(false);
    void setVertPath(std::string vertexPath) noexcept(false);

    xt::xarray<double> getRestShape() noexcept(false);
    xt::xarray<uint32_t> getFaceIndex() noexcept(false);
    xt::xarray<double> getRestJoint() noexcept(false);
    xt::xarray<double> getVertex() noexcept(false);

    // %% Modeling %%
    void init() noexcept(false);
    void launch(
        xt::xarray<double> beta,
        xt::xarray<double> theta) noexcept(false);
    void out(size_t) noexcept(false);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // SMPL_H
//=============================================================================
