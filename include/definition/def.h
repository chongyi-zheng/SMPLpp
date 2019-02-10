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
//  UNIVERSAL VARIABLE DECLARATIONS
//
//=============================================================================

#ifndef DEF_H
#define DEF_H

//===== EXTERNAL MACROS =======================================================

// #ifndef BATCH_SIZE
// #define BATCH_SIZE (const size_t) smpl::batch_size
// #endif // BATCH_SIZE

// #ifndef VERTEX_NUM
// #define VERTEX_NUM (const size_t) smpl::vertex_num
// #endif // VERTEX_NUM

// #ifndef JOINT_NUM
// #define JOINT_NUM (const size_t) smpl::joint_num
// #endif // JOINT_NUM

// #ifndef SHAPE_BASIS_DIM
// #define SHAPE_BASIS_DIM (const size_t) smpl::shape_basis_dim
// #endif // SHAPE_BASIS_DIM

// #ifndef POSE_BASIS_DIM
// #define POSE_BASIS_DIM (const size_t) smpl::pose_basis_dim
// #endif // POSE_BASIS_DIM

// #ifndef FACE_INDEX_NUM
// #define FACE_INDEX_NUM (const size_t) smpl::face_index_num
// #endif // FACE_INDEX_NUM

#ifndef BATCH_SIZE
#define BATCH_SIZE smpl::batch_size
#endif // BATCH_SIZE

#ifndef VERTEX_NUM
#define VERTEX_NUM smpl::vertex_num
#endif // VERTEX_NUM

#ifndef JOINT_NUM
#define JOINT_NUM smpl::joint_num
#endif // JOINT_NUM

#ifndef SHAPE_BASIS_DIM
#define SHAPE_BASIS_DIM smpl::shape_basis_dim
#endif // SHAPE_BASIS_DIM

#ifndef POSE_BASIS_DIM
#define POSE_BASIS_DIM smpl::pose_basis_dim
#endif // POSE_BASIS_DIM

#ifndef FACE_INDEX_NUM
#define FACE_INDEX_NUM smpl::face_index_num
#endif // FACE_INDEX_NUM

//===== INCLUDES ==============================================================

#include <stdlib.h>

//===== EXTERNAL DECLARATIONS =================================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL DECLARATIONS =================================================

extern int64_t batch_size;// 256
extern int64_t vertex_num;// 6890
extern const int64_t joint_num;// 24
extern const int64_t shape_basis_dim;// 10
extern const int64_t pose_basis_dim;// 207
extern const int64_t face_index_num;// 13776

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // DEF_H
//=============================================================================
