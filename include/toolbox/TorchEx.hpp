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
//  CLASS TorchEx DECLARATIONS AND IMPLEMENTATIONS
//
//=============================================================================

#ifndef TORCH_EXTRA_H
#define TORCH_EXTRA_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
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
 *      Extra functions to of TORCH library to be used in this system.
 * 
 *      This class will be based on the existing TORCH library to ensure
 *      correctness, efficiency, and robustness. This one is also a tool.
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor
 *      %
 *      - TorchEx: <private>
 *          Default constructor.
 * 
 *      - TorchEx: (overload) <private>
 *          Copy constructor.
 * 
 *      - ~TorchEx: <private>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <private>
 *          Assignment is used to copy an <TorchEx> instantiation, but this
 *          one will never be used as it is a private method.
 *      %%
 * 
 *      %
 *          Extended Libraries
 *      %
 *      - indexing_impl: <private> [static]
 *          Indexing a tensor recursively and get a slice of it accordingly.
 *      
 *      - indexing_impl: (overload) <private> [static]
 *          Base case for indexing a tensor.
 * 
 *      - indexing: <publice> [static]
 *          Wrapper to encapsulate indexing process.
 * 
 *      %%
 * 
 */

class TorchEx final
{

private: // PRIVATE ATTRIBUTES

    // %% Constructor and Destructor %%
    TorchEx() noexcept(false);
    TorchEx(const TorchEx &TorchEx) noexcept(false);
    ~TorchEx() noexcept(false);

    // %% Operators %%
    TorchEx &operator=(const TorchEx &TorchEx) noexcept(false);

    // %% Extended Libraries %%
    template <class Array, class ... Rest>
    static
    torch::Tensor indexing_impl(torch::Tensor &self, 
        int64_t layer, Array index, Rest ... indices) noexcept(false);

    template <class Array>
    static
    torch::Tensor indexing_impl(torch::Tensor &self, 
        int64_t layer, Array index) noexcept(false);

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE ATTRIBUTES

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

    // %% Extended Libraries %%
    template <class ... Arrays>
    static
    torch::Tensor indexing(torch::Tensor &self,
        Arrays ... indices) noexcept(false);
};

//===== CLASS IMPLEMENTATIONS =================================================

/**indexing
 * 
 * Brief
 * ----------
 * 
 *      Wrapper to encapsulate indexing process
 * 
 * Arguments
 * ----------
 * 
 *      @self: - Tensor -
 *          A tensor to be indexed.
 * 
 *      @indices: - Arrays (variadic template) -
 *          An integer list pack to specify slices of each dimension.
 * 
 * Return
 * ----------
 * 
 *      @out: - Tensor -
 *          Result of recursive indexing.
 * 
 * Note
 * ---------
 * 
 *      With this helper function, we can index a tensor as a numpy array:
 *          
 *          tensor[:, 0] <=> TensorEx::indexing(tensor, 
 *                              torch::IntList(), 
 *                              torch::IntList({0}))
 * 
 *          tensor[:, 0:5] <=> TensorEx::indexing(tensor, 
 *                              torch::IntList(), 
 *                              torch::IntList(0, 5))
 * 
 *          tensor[0:2:10] <=> TensorEx::indexing(tensor, 
 *                              torch::IntList(0, 2, 10));
 * 
 *          tensor[1:, 0] <=> TensorEx::indexing(tensor, 
 *                              torch::IntList({1, tensor.size(0)}), 
 *                              torch::IntList({0}))
 * 
 */
template <class ... Arrays>
torch::Tensor TorchEx::indexing(torch::Tensor &self, 
    Arrays ... indices) noexcept(false)
{
    torch::Tensor out;
    try {
        out = indexing_impl(self, 0, indices ...);
    }
    catch(std::exception &e) {
        throw;
    }

    return out;
}


/**indexing_impl
 * 
 * Brief
 * ----------
 * 
 *      Indexing a tensor recursively and get a slice of it accordingly.
 * 
 * Arguments
 * ----------
 * 
 *      @self: - Tensor -
 *          A tensor to be indexed.
 * 
 *      @layer: - int64_t -
 *          Current tensor dimension.
 * 
 *      @index: - Array (template) -
 *          An integer list to specify a slice from the leftmost dimension.
 * 
 *      @indices: - Rest (variadic template) -
 *          An integer list pack to specify slices of other dimensions.
 * 
 * Return
 * ----------
 * 
 *      @out: - Tensor -
 *          Result of recursive indexing.
 * 
 */
template <class Array, class ... Rest>
torch::Tensor TorchEx::indexing_impl(torch::Tensor &self, 
    int64_t layer, Array index, Rest ... indices) noexcept(false)
{
    if (std::is_same<Array, torch::IntList>::value) {

        torch::Tensor slice;
        torch::Tensor out;
        switch (index.size()) {
            case 0:
                slice = self.slice(layer);
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 1:
                slice = self.slice(layer,
                    *(index.begin()),
                    *(index.begin()) + 1).squeeze(layer);
                out = indexing_impl(slice, layer, indices ...);
                break;
            case 2:
                slice = self.slice(layer,
                    *(index.begin()),
                    *(index.begin() + 1));
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 3:
                slice = self.slice(layer, 
                    *(index.begin()),
                    *(index.begin() + 1),
                    *(index.begin() + 2));
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            default:
                throw smpl_error("TorchEx", 
                        "Invalid integer list for recursive indexing!");
                out = torch::Tensor();
                break;
        }
        return out;
    }
    else {
        throw smpl_error("TorchEx", 
            "Integer list type mismatch in recursion!");
    }

    return torch::Tensor();
}

/**indexing_impl (overload)
 * 
 * Brief
 * ----------
 * 
 *      Base case for indexing a tensor.
 * 
 * Arguments
 * ----------
 * 
 *      @self: - Tensor -
 *          A tensor to be indexed.
 * 
 *      @layer: - int64_t -
 *          Current tensor dimension.
 * 
 *      @index: - Array (template) -
 *          An integer list to specify a slice from the leftmost dimension.
 *
 * Return
 * ----------
 * 
 *      @out: - Tensor -
 *          Result of basic indexing.
 * 
 */
template <class Array>
torch::Tensor TorchEx::indexing_impl(torch::Tensor &self, 
    int64_t layer, Array index) noexcept(false)
{
    if (std::is_same<Array, torch::IntList>::value) {
        
        torch::Tensor out;
        switch (index.size()) {
            case 0:
                out = self.slice(layer);
                break;
            case 1:
                out = self.slice(layer, 
                        *(index.begin()), 
                        *(index.begin()) + 1).squeeze(layer);
                break;
            case 2:
                out = self.slice(layer, 
                        *(index.begin()), 
                        *(index.begin() + 1));
                break;
            case 3:
                out = self.slice(layer, 
                        *(index.begin()), 
                        *(index.begin() + 1), 
                        *(index.begin() + 2));
                break;
            default:
                throw smpl_error("TorchEx", 
                    "Invalid integer list for basic indexing!");
                out = torch::Tensor();
                break;
        }
        return out;
    }
    else {
        throw smpl_error("TorchEx", 
            "Integer list type mismatch in base!");
    }

    return torch::Tensor();
}

//=============================================================================
}
//=============================================================================
#endif // TORCH_EXTRA_H
//=============================================================================
