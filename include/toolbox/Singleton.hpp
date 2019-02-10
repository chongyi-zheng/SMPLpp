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
//  CLASS Singleton DECLARATIONS AND IMPLEMENTATIONS
//
//=============================================================================

#ifndef SINGLETON_HPP
#define SINGLETON_HPP

//===== INCLUDES ==============================================================

//----------
#include <toolbox/Exception.h>
//----------

//===== EXTERNAL MACROS =======================================================


//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/**
 * DESCRIPTION:
 * 
 *      Template class restricts the instance of other class to one. 
 * 
 *      This is a tool for the whole system
 * 
 * INHERITANCE:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - mm__singleton: <private>
 *          The only instantiation.
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and deconstructor
 *      %
 *      - Singleton: <private>
 *          Default constructor.
 * 
 *      - Singleton: (overload) <private>
 *          Copy constructor.
 * 
 *      - ~Singleton: <private>
 *          Deconstructor.
 *      %%
 *      
 *      %
 *          Operators
 *      %
 *      - operator=: <private>
 *          Assignment is used to copy an singleton, but this one will never be
 *          called.
 *      %%
 * 
 *      %
 *          Instantiation wrappers
 *      %
 *      - get:
 *          Get the only instantiation.
 * 
 *      - destroy:
 *          Destroy the only instantiation.
 *      %%
 * 
 */

template <class T>
class Singleton final
{

private: // PRIVATE ATTRIBUTES

    static T *mm__singleton;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    // %% Constructor and deconstructor %%
    Singleton() noexcept(false);
    Singleton(const Singleton<T> &singleton) noexcept(false);
    ~Singleton() noexcept(false);

    // %% Opeartors %%
    Singleton<T> &operator=(const Singleton<T> &singleton) noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Instantiation wrappers %%
    static T *get() noexcept(true);
    static void destroy() noexcept(true);

};

//===== INTERNAL AFTERWARDS DECLARATIONS ======================================

template <class T>
T *Singleton<T>::mm__singleton = nullptr;

//===== CLASS IMPLEMENTATIONS =================================================

/**Singleton
 * 
 * Brief
 * ----------
 * 
 *      Default constructor.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
template <class T>
Singleton<T>::Singleton() noexcept(false)
{
    throw smpl_error("Singleton", "Cannot call the constructor!");
}

/**Singleton (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @singleton: - const Singleton & -
 *          The singleton instantiation to copy with.
 *          
 * Return
 * ----------
 * 
 * 
 */
template <class T>
Singleton<T>::Singleton(const Singleton<T> &singleton) noexcept(false)
{
    throw smpl_error("Singleton", "Cannot call the copy constructor!");    
}

/**~Singleton
 * 
 * Brief
 * ----------
 * 
 *      Deconstructor.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
template <class T>
Singleton<T>::~Singleton() noexcept(false)
{
    throw smpl_error("Singleton", "Cannot call the deconstructor!");
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy an singleton, but this one will never be
 *      called.
 * 
 * Arguments
 * ----------
 * 
 *      @singleton: - const Singleton & -
 *          The singleton instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - Singleton & -
 *          Current instantiation.
 * 
 * 
 */
template <class T>
Singleton<T> &Singleton<T>::operator=(const Singleton<T> &singleton) 
    noexcept(false)
{
    throw smpl_error("Singleton", "Cannot call the assignment operator!");
}

/**get
 * 
 * Brief
 * ----------
 * 
 *      Get the only instantiation.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @mm__singleton:
 *          The only instantiation.
 * 
 */
template <class T>
T *Singleton<T>::get() noexcept(true)
{
    if (!mm__singleton) {
        mm__singleton = new T;
    }

    return mm__singleton;
}

/**destroy
 * 
 * Brief
 * ----------
 * 
 *      Destroy the only instantiation.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
template <class T>
void Singleton<T>::destroy() noexcept(true)
{
    if (mm__singleton) {
        delete mm__singleton;
    }

    return;
}

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // SINGLETON_H
//=============================================================================
