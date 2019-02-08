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

#ifndef BATCH_SIZE
#define BATCH_SIZE (const size_t) smpl::batch_size
#endif // BATCH_SIZE

#ifndef VERTEX_NUM
#define VERTEX_NUM (const size_t) smpl::vertex_num
#endif // VERTEX_NUM

#ifndef JOINT_NUM
#define JOINT_NUM (const size_t) smpl::joint_num
#endif // JOINT_NUM

#ifndef SHAPE_BASIS_DIM
#define SHAPE_BASIS_DIM (const size_t) smpl::shape_basis_dim
#endif // SHAPE_BASIS_DIM

#ifndef POSE_BASIS_DIM
#define POSE_BASIS_DIM (const size_t) smpl::pose_basis_dim
#endif // POSE_BASIS_DIM

#ifndef FACE_INDEX_NUM
#define FACE_INDEX_NUM (const size_t) smpl::face_index_num
#endif // FACE_INDEX_NUM

#ifndef BATCH_SIZE_RAW
#define BATCH_SIZE_RAW smpl::batch_size
#endif // BATCH_SIZE_RAW

#ifndef VERTEX_NUM_RAW
#define VERTEX_NUM_RAW smpl::vertex_num
#endif // VERTEX_NUM_RAW

#ifndef JOINT_NUM_RAW
#define JOINT_NUM_RAW smpl::joint_num
#endif // JOINT_NUM_RAW

#ifndef SHAPE_BASIS_DIM_RAW
#define SHAPE_BASIS_DIM_RAW smpl::shape_basis_dim
#endif // SHAPE_BASIS_DIM_RAW

#ifndef POSE_BASIS_DIM_RAW
#define POSE_BASIS_DIM_RAW smpl::pose_basis_dim
#endif // POSE_BASIS_DIM_RAW

#ifndef FACE_INDEX_NUM_RAW
#define FACE_INDEX_NUM_RAW smpl::face_index_num
#endif // FACE_INDEX_NUM_RAW

//===== INCLUDES ==============================================================

#include <stddef.h>

//===== EXTERNAL DECLARATIONS =================================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL DECLARATIONS =================================================

extern int batch_size;// 256
extern int vertex_num;// 6890
extern const int joint_num;// 24
extern const int shape_basis_dim;// 10
extern const int pose_basis_dim;// 207
extern const int face_index_num;// 13776

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // DEF_H
//=============================================================================
