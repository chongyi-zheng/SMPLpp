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
//  CLASS XtensorEx DECLARATIONS AND IMPLEMENTATIONS
//
//=============================================================================

#ifndef XTENSOR_EXTRA_H
#define XTENSOR_EXTRA_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <stddef.h>
#include <typeinfo>
#include <stdexcept>
//----------
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
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
 *      Extra functions to of XTENSOR library to be used in this system.
 * 
 *      This class will be based on the existing XTENSOR library to ensure
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
 *          Constructor and Deconstructor
 *      %
 *      - XtensorEx: <private>
 *          Default constructor.
 * 
 *      - XtensorEx: (overload) <private>
 *          Copy constructor.
 * 
 *      - ~XtensorEx: <private>
 *          Deconstructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <private>
 *          Assignment is used to copy an extra xtensor instantiation, but this
 *          one will never be used as it is a private method.
 *      %%
 * 
 *      %
 *          Extended Libraries
 *      %
 *      - matmul: <public> [static]
 *          Use array dot to implement matrix multiplication. (np.matmul)
 * 
 *      - printShape: <public> [static]
 *          Print the shape of a array or tensor.
 * 
 *      - array2tuple_impl: <private> [static]
 *          Convert a fixed size xarray array to xtuple.
 * 
 *      - array2tuple: <public> [static]
 *          Wrapper to convert a fixed size xarray array to xtuple
 *      %%
 * 
 */

class XtensorEx final
{

private: // PRIVATE ATTRIBUTES

    // %% Constructor and Deconstructor %%
    XtensorEx() noexcept(false);
    XtensorEx(const XtensorEx &xtensorEx) noexcept(false);
    ~XtensorEx() noexcept(false);

    // %% Operators %%
    XtensorEx &operator=(const XtensorEx &xtensorEx) noexcept(false);

    // %% Extended Libraries %%
    template <class Array, size_t ... Is>
    static auto array2tuple_impl(const Array &array, 
        std::index_sequence<Is ...>) noexcept(true);

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE ATTRIBUTES

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

    // %% Extended Libraries %%
    template <class T>
    static xt::xarray<T> matmul(xt::xarray<T> a, xt::xarray<T> b)
    noexcept(false);

    template <class T>
    static void printShape(xt::xarray<T> array) noexcept(false);

    template <class T, size_t N>
    static auto array2tuple(const std::array<T, N>& array) noexcept(false);

};

//===== CLASS IMPLEMENTATIONS =================================================

/**matmul
 * 
 * Brief
 * ----------
 * 
 *      Use array dot to implement matrix multiplication. (np.matmul)
 * 
 * Arguments
 * ----------
 * 
 *      @a: - xarray -
 *          First multi-dimensional array.
 * 
 *      @b: - xarray -
 *          Second multi-dimensional array.
 * 
 * Return
 * ----------
 * 
 *      @out: - xarray -
 *          Result of matrix multiplication.
 * 
 * Notes
 * ----------
 * 
 *      Here we need to check the template type first to make sure valid array.
 * 
 *      If both arguments are 2-D they are multiplied like conventional 
 *      matrices.
 * 
 *      If either argument is N-D, N > 2, it is treated as a stack of 
 *      matrices residing in the last two indexes and broadcast accordingly.
 * 
 *      If the first argument is 1-D, it is promoted to a matrix by prepending 
 *      a 1 to its dimensions. After matrix multiplication the prepended 1 is 
 *      removed.
 * 
 *      If the second argument is 1-D, it is promoted to a matrix by appending 
 *      a 1 to its dimensions. After matrix multiplication the appended 1 is 
 *      removed.
 * 
 * Sources:
 * ----------
 * 
 *      numpy documentation, numpy.matmul.
 *      https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html.
 *      
 */
template <class T>
xt::xarray<T> XtensorEx::matmul(xt::xarray<T> a, xt::xarray<T> b)
    noexcept(false)
{
    if (abs(a.shape().size() - b.shape().size()) > 1) {
        throw smpl_error("XtensorEx", "Shape mismatching!");
    }

    if (typeid(T).hash_code() != typeid(int).hash_code()
        && typeid(T).hash_code() != typeid(size_t).hash_code()
        && typeid(T).hash_code() != typeid(float).hash_code()
        && typeid(T).hash_code() != typeid(double).hash_code()) {
        throw smpl_error("XtensorEx", "Element type mismatching!");
    }

    //
    // filter vector
    //
    bool aflag = true;
    bool bflag = true;
    if (a.shape().size() < b.shape().size()) {
        aflag = false;
    }
    if (b.shape().size() < a.shape().size()) {
        bflag = false;
    }

    typename xt::xarray<T>::shape_type shape = 
        aflag ? a.shape() : b.shape();// (..., M, M)
    typename xt::xarray<T>::shape_type outShape = 
        (!aflag) ? a.shape() : b.shape();// (..., M) or (..., M, M)
    xt::xarray<T> out(outShape);
    //
    // Both arguments are 2-D, end the recursion.
    //
    //      a - (M, M) | (M)    | (M, M)
    //      b - (M, M) | (M, M) | (M)
    // 
    if (shape.size() == 2) {

        if (aflag && bflag || bflag) {
            xt::xarray<T> matrix, vector;

            for (size_t col = 0; col < *(shape.end() - 1); col++) {
                matrix = xt::view(a, xt::all(), xt::all());
                vector = xt::view(b, xt::all(), col);
                xt::view(out, xt::all(), col) = xt::linalg::dot(matrix, vector);
            }
        }
        else {
            out = xt::linalg::dot(a, b);
        }
    }
    //
    // Both arguments are N-D, go into deeper layer.
    //
    //      a - (..., M, M) | (..., M)    | (..., M, M)
    //      b - (..., M, M) | (..., M, M) | (..., M)
    //
    else {
        xt::xarray<T> aSlice, bSlice;
        for (size_t i = 0; i < shape[0]; i++) {
            aSlice = xt::view(a, i);
            bSlice = xt::view(b, i);
            xt::view(out, i) = matmul(aSlice, bSlice);
        }
    }

    return out;
}

/**printShape
 * 
 * Brief
 * ----------
 * 
 *      Print the shape of a array or tensor.
 * 
 * Arguments
 * ----------
 * 
 *      @array: - xarray -
 *          We will print the shape of this array.
 * 
 * Return
 * ----------
 * 
 * 
 * Notes
 * ----------
 * 
 *      Here we need to check the template type first to make sure valid array.
 * 
 */
template <class T>
void XtensorEx::printShape(xt::xarray<T> array) noexcept(false)
{
    if (typeid(T).hash_code() != typeid(int).hash_code()
        && typeid(T).hash_code() != typeid(uint32_t).hash_code()
        && typeid(T).hash_code() != typeid(size_t).hash_code()
        && typeid(T).hash_code() != typeid(float).hash_code()
        && typeid(T).hash_code() != typeid(double).hash_code()) {
        throw smpl_error("XtensorEx", "Element type mismatching!");
    }

    auto shape = array.shape();
    std::cout << "(" << shape[0];
    for (auto i = 1; i < shape.size(); i++) {
        std::cout << ", " << shape[i];
    }
    std::cout << ")" << std::endl;

    return;
}

/**array2tuple_impl
 * 
 * Brief
 * ----------
 * 
 *      Convert a fixed size xarray array to xtuple.
 * 
 * Arguments
 * ----------
 * 
 *      @array: - const Array& -
 *          An array of xarray to be converted.
 * 
 *      @...Is: - index_sequence -
 *          A non-type (size_t) parameter pack representing the integer 
 *          sequence.
 *
 * Return
 * ----------
 * 
 *      @tuple: - xtuple -
 *          Tuple to store elements of the array.
 * 
 */
template <class Array, size_t ... Is>
auto XtensorEx::array2tuple_impl(
    const Array& array, std::index_sequence<Is ...>) noexcept(true)
{
    return std::make_tuple(array[Is] ...);;
}

/**array2tuple
 * 
 * Brief
 * ----------
 * 
 *      Wrapper to convert a fixed size xarray array to xtuple.
 * 
 * Arguments
 * ----------
 * 
 *      @array: - const Array& -
 *          An array of xarray to be converted.
 * 
 *
 * Return
 * ----------
 * 
 *      @tuple: - xtuple -
 *          Tuple to store elements of the array.
 * 
 */
template <class T, size_t N>
auto XtensorEx::array2tuple(const std::array<T, N>& array) noexcept(false)
{
    if (std::is_same<T, xt::xarray<double>>::value) {
        auto tuple = array2tuple_impl(array, std::make_index_sequence<N>{});
        return tuple;
    }
    else {
        throw smpl_error("XtensorEx", "Array type mismatching!");
    }
}

//=============================================================================
}
//=============================================================================
#endif // XTENSOR_EXTRA_H
//=============================================================================
