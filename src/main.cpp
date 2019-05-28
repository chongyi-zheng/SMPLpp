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
//  APPLICATION ENTRANCE
//
//=============================================================================

//===== MACROS ================================================================

#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>

//===== INCLUDES ==============================================================

//----------
#include <chrono>
//----------
#include <torch/torch.h>
//----------
#include "definition/def.h"
#include "toolbox/Singleton.hpp"
#include "toolbox/Tester.h"
#include "smpl/SMPL.h"
//----------

//===== FORWARD DECLARATIONS ==================================================


//===== NAMESPACE =============================================================

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

//===== MAIN FUNCTION =========================================================

int main(int argc, char const *argv[])
{
	torch::Device cuda(torch::kCUDA);
	cuda.set_index(0);

	// smpl::Tester tester;
	// tester.setDevice(cuda);

	// tester.singleton();
	// tester.blendShape();
	// tester.jointRegression();
	// tester.worldTransformation();
	// tester.linearBlendSkinning();
	// tester.import();

	std::string modelPath = "../data/smpl_female.json";
	std::string outputPath = "../out/vertices.obj";

	torch::Tensor beta = 0.03 * torch::rand(
		{BATCH_SIZE, SHAPE_BASIS_DIM});// (N, 10)
	torch::Tensor theta = 0.2 * torch::rand(
		{BATCH_SIZE, JOINT_NUM, 3});// (N, 24, 3)

	torch::Tensor vertices;

	try {
		SINGLE_SMPL::get()->setDevice(cuda);
		SINGLE_SMPL::get()->setModelPath(modelPath);

		auto begin = clk::now();
		SINGLE_SMPL::get()->init();
		auto end = clk::now();
		auto duration = std::chrono::duration_cast<ms>(end - begin);
		std::cout << "Time duration to load SMPL: " 
			<< (double)duration.count() / 1000 << " s" << std::endl;

		const int64_t LOOPS = 100;
		duration = std::chrono::duration_cast<ms>(end - end);// reset duration
		for (int64_t i = 0; i < LOOPS; i++) {
			begin = clk::now();
			SINGLE_SMPL::get()->launch(beta, theta);
			end = clk::now();
			duration += std::chrono::duration_cast<ms>(end - begin);
		}
		std::cout << "Time duration to run SMPL: " 
			<< (double)duration.count() / LOOPS << " ms" << std::endl;

		vertices = SINGLE_SMPL::get()->getVertex();
	}
	catch(std::exception &e) {
		std::cerr << e.what() << std::endl;
	}

	try {
		SINGLE_SMPL::get()->setVertPath(outputPath);
		SINGLE_SMPL::get()->out(0);
	}
	catch(std::exception &e) {
		std::cerr << e.what() << std::endl;
	}

	SINGLE_SMPL::destroy();

    return 0;
}


//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
