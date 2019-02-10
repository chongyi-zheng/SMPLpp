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

#ifndef TESTER_H
#define TESTER_H

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
 *      Test module aims to test other modules to ensure correctness and
 *      robustness.
 * 
 *      This class is a tool for the whole system.
 * 
 * INHERITANCE:
 * 
 * 
 * ATTRIBUTES:
 * 
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Deconstructor
 *      %
 *      - Tester: <public>
 *          Default constructor.
 * 
 *      - Tester: <public>
 *          Constructor to initialize torch device.
 * 
 *      - Tester: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~Tester: <public>
 *          ~Deconstructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment operator is used to copy an tester.
 * 
 *      %%
 * 
 *      %
 *          Setter and Getter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      %
 *          Singleton pattern
 *      %
 *      - singleton: <public>
 *          Test the <Singleton> module.
 *      %%
 * 
 *      %
 *          File reading
 *      %
 *      - import: <public>
 *          Test file reading module.
 *      %%
 * 
 *      %
 *          Shape blend shape and pose blend shape
 *      %
 *      - blendShape: <publc>
 *          Test the <BlendShape> module.
 *      %%
 * 
 *      %
 *          Joint regressor
 *      %
 *      - jointRegression: <public>
 *          Test the <JointRegression> module.
 *      %%
 * 
 *      %
 *          Linear blend shape
 *      %
 *      - linearBlendSkinng: <public>
 *          Test the <LinearBlendSkinng> module.
 *      %%
 * 
 *      %
 *          World transformation
 *      %
 *      - worldTransformation: <public>
 *          Test the <WorldTransformation> module.
 *      %%
 * 
 */

class Tester final
{

private: // PRIVATE ATTRIBUTES

    torch::Device m__device;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Deconstructor %%
    Tester() noexcept(true);
    Tester(const Tester &tester) noexcept(true);
    ~Tester() noexcept(true);

    // %% Operators %%
    Tester &operator=(const Tester &tester) noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);

    // %% Singleton pattern %%
    void singleton() noexcept(true);

    // %% File reading %%
    void import() noexcept(true);

    // %% Shape blend shape and pose blend shape %%
    void blendShape() noexcept(true);

    // %% Joint regressor %%
    void jointRegression() noexcept(true);

    // %% Linear blend shape %%
    void linearBlendSkinning() noexcept(true);

    // %% World transformation %%
    void worldTransformation() noexcept(true);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // TESTER_H
//=============================================================================
