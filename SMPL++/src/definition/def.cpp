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
//  UNIVERSAL VARIABLE DEFINITIONS
//
//=============================================================================

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

#include "definition/def.h"

//===== EXTERNAL DECLARATIONS =================================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL DEFINITIONS ==================================================

int64_t batch_size = 1;// 1
int64_t vertex_num = 6890;// 6890
const int64_t joint_num = 24;// 24
const int64_t shape_basis_dim = 10;// 10
const int64_t pose_basis_dim = 207;// 207
const int64_t face_index_num = 13776;// 13776

//=============================================================================
} // namespace smpl
//=============================================================================
