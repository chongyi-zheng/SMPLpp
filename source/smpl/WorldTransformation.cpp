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
//  CLASS WorldTransformation IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <xtensor/xview.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor-blas/xlinalg.hpp>
//----------
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/XtensorEx.hpp"
#include "smpl/WorldTransformation.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**WorldTransformation
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
WorldTransformation::WorldTransformation() noexcept(true) :
    __joints(),
    __poseRot(),
    __kineTree(),
    __transformations()
{
}

/**WorldTransformation (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize kinematic tree.
 * 
 * Arguments
 * ----------
 * 
 *      @kineTree: - xarray -
 *          Hierarchical relation between joints, the root is at the belly button,
 *          (2, 24).
 * 
 * Return
 * ----------
 * 
 * 
 */
WorldTransformation::WorldTransformation(xt::xarray<double> kineTree) 
    noexcept(false) :
    __joints(),
    __poseRot(),
    __transformations()
{
    if (kineTree.shape() == 
        xt::xarray<double>::shape_type({2, JOINT_NUM})) {
        __kineTree = kineTree;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to initialize kinematic tree!");
    }
}

/**WorldTransformation (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @worldTransformation: - xarray -
 *          The <WorldTransformation> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
WorldTransformation::WorldTransformation(
    const WorldTransformation &worldTransformation) noexcept(false) :
    __transformations()
{
    try {
        *this = worldTransformation;
    }
    catch(std::exception& e) {
        throw;
    }
    
}

/**~WorldTransformation
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
WorldTransformation::~WorldTransformation() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <WorldTransformation> instantiation..
 * 
 * Arguments
 * ----------
 * 
 *      @worldTransformation: - xarray -
 *          The <WorldTransformation> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @*this: - xarray -
 *          Currrent instantiation.
 * 
 */
WorldTransformation &WorldTransformation::operator=(
        const WorldTransformation &worldTransformation) noexcept(false)
{
    //
    // hard copy
    //
    if (worldTransformation.__joints.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __joints = worldTransformation.__joints;
    }
    else {
        throw smpl_error("WorldTransformation", "Failed to copy joints");
    }

    if (worldTransformation.__poseRot.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        __poseRot = worldTransformation.__poseRot;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to copy pose rotations");
    }

    if (worldTransformation.__kineTree.shape() ==
        xt::xarray<double>::shape_type({2, JOINT_NUM})) {
        __kineTree = worldTransformation.__kineTree;
    }
    else {
        throw smpl_error("WorldTransformation",
            "Failed to copy kinematic tree!");
    }

    //
    // soft copy
    //
    if (worldTransformation.__transformations.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, VERTEX_NUM, 4, 4})) {
        __transformations = worldTransformation.__transformations;
    }

    return *this;
}

/**setJoint
 * 
 * Brief
 * ----------
 * 
 *      Set joint locations of the deformed shape.
 * 
 * Arguments
 * ----------
 * 
 *      @joints: - xarray -
 *          Joint locations of the deformed shape after regressing, (N, 24, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void WorldTransformation::setJoint(xt::xarray<double> joints) noexcept(false)
{
    if (joints.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})) {
        __joints = joints;
    }
    else {
        throw smpl_error("WorldTransformation", "Failed to set joints");
    }

    return;
}

/**setPoseRotation
 * 
 * Brief
 * ----------
 * 
 *      Set pose rotations by axis-angles representations.
 * 
 * Arguments
 * ----------
 * 
 *      @poseRot: - xarray -
 *          Rotations with respect to new pose by axis-angles
 *          representations, (N, 24, 3, 3).
 * 
 * Return
 * ----------
 * 
 * 
 */
void WorldTransformation::setPoseRotation(xt::xarray<double> poseRot) 
    noexcept(false)
{
    if (poseRot.shape() ==
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3, 3})) {
        __poseRot = poseRot;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to set pose rotations");
    }

    return;
}

/**setJoint
 * 
 * Brief
 * ----------
 * 
 *      Set kinematice tree of the body.
 * 
 * Arguments
 * ----------
 * 
 *      @kineTree: - xarray -
 *          Hierarchical relation between joints, the root is at the belly button,
 *          (2, 24).
 * 
 * Return
 * ----------
 * 
 * 
 */
void WorldTransformation::setKinematicTree(xt::xarray<uint32_t> kineTree) 
    noexcept(false)
{
    if (kineTree.shape() == 
        xt::xarray<uint32_t>::shape_type({2, JOINT_NUM})) {
        __kineTree = kineTree;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to set kinematic tree!");
    }
    
    return;
}

/**getTransformation
 * 
 * Brief
 * ----------
 * 
 *      Get world transformations.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @transformation: - xarray -
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 */
xt::xarray<double> WorldTransformation::getTransformation() noexcept(false)
{
    xt::xarray<double> transformation;

    if (__transformations.shape() == 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        transformation = __transformations;
    }

    return transformation;
}

/**transform
 * 
 * Brief
 * ----------
 * 
 *      Outside wrapper to encapsulate world transformation process.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 * Notes
 * ----------
 * 
 *      This function implements equation (4) in the paper.
 * 
 *      The formula (3) are also considered here, but we include the initial
 *      T-pose inside the transformation matrix such that world transformations
 *      of theta star are omitted.
 * 
 */
void WorldTransformation::transform() noexcept(false)
{
    //
    // rotations in homogeneous coordinates
    //
    auto zeros = xt::zeros<double>(
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 1, 3});// (N, 24, 1, 3)
    xt::xarray<double> poseRotHomo = xt::concatenate(
        xt::xtuple(__poseRot, zeros), 2);// (N, 24, 4, 3)

    //
    // local transformation
    //
    xt::xarray<double> localTransformations;
    try {
        localTransformations = localTransform(poseRotHomo);
    }
    catch(std::exception &e) {
        throw;
    }
    
    //
    // global transformation
    //
    xt::xarray<double> globalTransformations;
    try {
        globalTransformations = globalTransform(localTransformations);
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // relative transformation
    //
    try {
        relativeTransform(globalTransformations);
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

/**localTransform
 * 
 * Brief
 * ----------
 * 
 *      Local transformations with respect to each joint.
 * 
 * Arguments
 * ----------
 * 
 *      @poseRotHomo: - xarray -
 *          Relative rotation denoted by axis-angles representations in
 *          homogeneous coordinates,, (N, 24, 4, 3).
 * 
 * Return
 * ----------
 * 
 *      @localTransformations: - xarray -
 *          Local transformations in each joint local coordinate but expressed
 *          by global coordinates, (N, 24, 4, 4).
 * 
 * Notes
 * ----------
 *      Let
 *          ti - local translation of joint i
 *          ji - global location of joint i
 * 
 *      Then we have,
 *          t0 = j0, t1 = j1 - j0, t2 = j2 - j0,
 *          t3 = j3 - j0, t4 = j4 - j1, t5 = j5 - j2,
 *          t6 = j6 - j3, t7 = j7 - j4, t8 = j8 - j5,
 *          t9 = j9 - j6, t10 = j10 - j7, t11 = j11 - j8,
 *          t12 = j12 - j9, t13 = j13 - j9, t14 = j14 - j9,
 *          t15 = j15 - j12, t16 = j16 - j13, t17 = j17 - j14,
 *          t18 = j18 - j15, t19 = j19 - j17, t20 = j20 - j18,
 *          t21 = j21 - j19, t22 = j22 - j20, t23 = j23 - j21.
 * 
 */
xt::xarray<double> WorldTransformation::localTransform(
    xt::xarray<double> poseRotHomo) noexcept(false)
{
    if (poseRotHomo.shape() !=
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 3})) {
        throw smpl_error("WorldTransformation", 
            "Cannot transform bones locally!");
    }

    std::array<xt::xarray<double>, 24> translations;
    translations[0] = xt::view(__joints, xt::all(), 0, xt::all());

    size_t ancestor;
    xt::xarray<double> translation;
    for (size_t i = 1; i < JOINT_NUM; i++) {
        ancestor = (size_t)__kineTree(0, i);
        translation = xt::view(__joints, xt::all(), i, xt::all()) -
            xt::view(__joints, xt::all(), ancestor, xt::all());// (N, 3)
        translations[i] = translation;// [i, (N, 3)]
    }
    auto tuple = XtensorEx::array2tuple(translations);

    xt::xarray<double> localTranslations = xt::stack(
        std::move(tuple), 1);// (N, 24, 3)
    localTranslations = xt::expand_dims(localTranslations, 3);// (N, 24, 3, 1)

    xt::xarray<double> ones = xt::ones<double>(
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 1, 1});// (N, 24, 1, 1)
    xt::xarray<double> localTransformationsHomo = xt::concatenate(
        xt::xtuple(localTranslations, ones), 2);// (N, 24, 4, 1)
    xt::xarray<double> localTransformations = xt::concatenate(
        xt::xtuple(poseRotHomo, localTransformationsHomo), 3);// (N, 24, 4, 4)

    return localTransformations;
}

/**globalTransform
 * 
 * Brief
 * ----------
 * 
 * 
 * Arguments
 * ----------
 * 
 *      @localTransformations: - xarray -
 *          Local transformations in each joint local coordinate but expressed
 *          by global coordinates, (N, 24, 4, 4).
 * 
 * Return
 * ----------
 * 
 *      @globalTransformations: - xarray -
 *          Global transformation of each bone, (N, 24, 4, 4).
 * 
 * Notes
 * ----------
 * 
 *      Let,
 *          Ai - absolute rotation of joint i
 *          Qi - relative rotation of joint i
 * 
 *      Then we have,
 *          A0 = Q0, A1 = A0Q1 = Q0Q1,
 *          A2 = A0Q2 = Q0Q2, A3 = A0Q3 = Q0Q3,
 *          A4 = A1Q4 = Q0Q1Q4, A5 = A2Q5 = Q0Q2Q5,
 *          A6 = A3Q6 = Q0Q3Q6, A7 = A4Q7 = Q0Q1Q4Q7,
 *          A8 = A5Q8 = Q0Q2Q5Q8, A9 = A6Q9 = Q0Q3Q6Q9,
 *          A10 = A7Q10 = Q0Q1Q4Q7Q10, A11 = A8Q11 = Q0Q2Q5Q8Q11,
 *          A12 = A9Q12 = Q0Q3Q6Q9Q12, A13 = A9Q13 = Q0Q3Q6Q9Q13,
 *          A14 = A9Q14 = Q0Q3Q6Q9Q14, A15 = A12Q15 = Q0Q3Q6Q9Q12Q15,
 *          A16 = A13Q16 = Q0Q3Q6Q9Q13Q16,
 *          A17 = A14Q17 = Q0Q3Q6Q9Q14Q17,
 *          A18 = A16Q18 = Q0Q3Q6Q8Q13Q16Q18,
 *          A19 = A17Q19 = Q0Q3Q6Q9Q14Q17Q19,
 *          A20 = A18Q20 = Q0Q3Q6Q8Q13Q16Q18Q20,
 *          A21 = A19Q21 = Q0Q3Q6Q9Q14Q17Q19Q21,
 *          A22 = A20Q22 = Q0Q3Q6Q8Q13Q16Q18Q20Q22,
 *          A23 = A21Q23 = Q0Q3Q6Q9Q14Q17Q19Q21Q23.
 * 
 */
xt::xarray<double> WorldTransformation::globalTransform(
    xt::xarray<double> localTransformations) noexcept(false)
{
    if (localTransformations.shape() != 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        throw smpl_error("WorldTransformations", 
            "Cannot transform bones globally!");
    }

    std::array<xt::xarray<double>, 24> transformations;
    transformations[0] = xt::view(localTransformations, 
        xt::all(), 0, xt::all(), xt::all());// [0, (N, 4, 4)]

    size_t ancestor;

    xt::xarray<double> transformation;
    xt::xarray<double> globalSlice, localSlice;
    for (size_t i = 1; i < JOINT_NUM; i++) {
        ancestor = (size_t)__kineTree(0, i);
        globalSlice = transformations[ancestor];// (N, 4, 4)
        localSlice = xt::view(localTransformations, 
            xt::all(), i, xt::all(), xt::all());// (N, 4, 4)

        try {
            transformation = XtensorEx::matmul(
                globalSlice, localSlice);// (N, 4, 4)
        }
        catch(std::exception &e) {
            throw;
        }
        
        transformations[i] = transformation;// [i, (N, 4, 4)]
    }
    auto tuple = XtensorEx::array2tuple(transformations);

    xt::xarray<double> globalTransformations = xt::stack(
        std::move(tuple), 1);// (N, 24, 4, 4)

    return globalTransformations;
}

/**relativeTransform
 * 
 * Brief
 * ----------
 * 
 *      Eliminate rest pose's transformation from global transformation.
 * 
 * Arguments
 * ----------
 * 
 *      @globalTransformations: - xarray -
 *          Global transformation of each bone, (N, 24, 4, 4).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 * Notes
 * ----------
 * 
 *      Let,
 *          ei - eliminated vector of joint i
 *          ji - global location of joint i
 * 
 *      Then we have,
 *          e0 = Q0j0, e1 = A1j1 = Q0Q1j1,
 *          e2 = A2j2 = Q0Q2j2, e3 = A3j3 = Q0Q3j3,
 *          e4 = A4j4 = Q0Q1Q4j4, e5 = A5j5 = Q0Q2Q5j5,
 *          e6 = A6j6 = Q0Q3Q6j6, e7 = A7j7 = Q0Q1Q4Q7j7,
 *          e8 = A8j8 = Q0Q2Q5Q8j8, e9 = A9j9 = Q0Q3Q6Q9j9,
 *          e10 = A10j10 = Q0Q1Q4Q7Q10j10, e11 = A11j11 = Q0Q2Q5Q8Q11j11,
 *          e12 = A12j12 = Q0Q3Q6Q9Q12j12, e13 = A13j13 = Q0Q3Q6Q9Q13j13,
 *          e14 = A14j14 = Q0Q3Q6Q9Q14j14,
 *          e15 = A15j15 = Q0Q3Q6Q9Q12Q15j15,
 *          e16 = A16j16 = Q0Q3Q6Q9Q13Q16j16,
 *          e17 = A17j17 = Q0Q3Q6Q9Q14Q17j17,
 *          e18 = A18j18 = Q0Q3Q6Q8Q13Q16Q18j18,
 *          e19 = A19j19 = Q0Q3Q6Q9Q14Q17Q19j19,
 *          e20 = A20j20 = Q0Q3Q6Q8Q13Q16Q18Q20j20,
 *          e21 = A21j21 = Q0Q3Q6Q9Q14Q17Q19Q21j21,
 *          e22 = A22j22 = Q0Q3Q6Q8Q13Q16Q18Q20Q22j22,
 *          e23 = A23j23 = Q0Q3Q6Q9Q14Q17Q19Q21Q23j23.
 * 
 */
void WorldTransformation::relativeTransform(
    xt::xarray<double> globalTransformations) noexcept(false)
{
    if (globalTransformations.shape() != 
        xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 4, 4})) {
        throw smpl_error("WorldTransformation", 
            "Cannot transform bones relatively!");
    }

    xt::xarray<double> globalSlice = xt::view(
        globalTransformations, 
        xt::all(), 
        xt::all(), 
        xt::range(0, 3), 
        xt::range(0, 3)
    );// (N, 24, 3, 3)
    xt::xarray<double> augJoints = xt::expand_dims(
        __joints, 3);// (N, 24, 3, 1)
    xt::xarray<double> eliminated = XtensorEx::matmul(
        globalSlice, augJoints);// (N, 24, 3, 1)
    xt::xarray<double> zeros = xt::zeros<double>(
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 1, 1});// (N, 24, 1, 1)
    xt::xarray<double> eliminatedHomo = xt::concatenate(
        xt::xtuple(eliminated, zeros), 2);// (N, 24, 4, 1)
    zeros = xt::zeros<double>(
        {BATCH_SIZE_RAW, JOINT_NUM_RAW, 4, 3});// (N, 24, 4, 3)
    eliminatedHomo = xt::concatenate(
        xt::xtuple(zeros, eliminatedHomo), 3);// (N, 24, 4, 4)
    xt::xarray<double> relativeTransformtions = 
        globalTransformations - eliminatedHomo;// (N, 24, 4, 4)

    __transformations = relativeTransformtions;

    return;
}

//=============================================================================
} // namespace smpl
//=============================================================================
