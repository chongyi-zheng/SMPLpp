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
#include "definition/def.h"
#include "toolbox/Singleton.hpp"
#include "smpl/SMPL.h"
//----------
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
//----------
#include <torch/torch.h>

//===== FORWARD DECLARATIONS ==================================================


//===== NAMESPACE =============================================================

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

//===== MAIN FUNCTION =========================================================

int main(int argc, char const *argv[])
{
	// smpl::Tester tester;
	
	// tester.singleton();
	// tester.blendShape();
	// tester.jointRegression();
	// tester.worldTransformation();
	// tester.linearBlendSkinning();
	// tester.import();

	// torch::Device device(torch::kCUDA);
	// torch::Tensor tensor1 = torch::rand({2, 3, 3}, device);
	// torch::Tensor ones = torch::ones({3, 3}, device);

	// int64_t idx[1] = {0};
	// torch::Tensor idxs1 = torch::from_blob(idx, {1}, torch::kLong).to(device);
	// std::cout << tensor1.slice(0, 0, 1) << std::endl;
	// std::cout << tensor1.slice(0, 0, 1).squeeze(0).slice(0, 0, 1) << std::endl;
	// std::cout << tensor1.slice(0, 0, 1).squeeze(0).slice(0, 0, 1).squeeze(0).slice(0, 0, 1) << std::endl;
	// tensor1.slice(0, 0, 1).squeeze(0).slice(0, 0, 1).squeeze(0).slice(0, 0, 1) = 1.0;
	// std::cout << tensor1 << std::endl;
	// torch::Tensor slice = tensor1.index({idxs1});
	// std::cout << slice << std::endl;
	// slice = ones;
	// std::cout << slice << std::endl;
	// std::cout << tensor1 << std::endl;


	// int data[2] = {1, 2};
	// auto shape = tensor1.sizes();
	// std::cout << shape << std::endl;
	// xt::xarray<float> arr = xt::adapt((float *)tensor1.to(torch::kCPU).data_ptr(), {3, 3});
	// std::cout << arr << std::endl;

	std::string modelPath = "../data/smpl_female.json";
	std::string outputPath = "../out/vertices.obj";

	xt::random::seed(0);
	xt::xarray<double> beta = 0.03 * xt::random::rand<double>(
		{BATCH_SIZE_RAW, SHAPE_BASIS_DIM_RAW});// (N, 10)
	xt::xarray<double> theta = 0.2 * xt::random::rand<double>(
		{BATCH_SIZE_RAW, JOINT_NUM_RAW, 3});// (N, 24, 3)

	// torch::Tensor beta = 0.03 * torch::rand(
	// 	{BATCH_SIZE_RAW, SHAPE_BASIS_DIM_RAW});// (N, 10)
	// torch::Tensor theta = 0.2 * torch::rand(
	// 	{BATCH_SIZE_RAW, JOINT_NUM_RAW, 3});// (N, 24, 3)

	// xt::xarray<double> beta = {
	// 	{
	// 		0.02401774, 0.00700565, 0.02410612, 0.02072617, 0.01097957, 
	// 		0.00256078, 0.01286519, 0.01526721, 0.02432266, 0.02775677
	// 	}
	// };// (1, 10)
	// xt::xarray<double> theta = {
	// 	{
	// 		{0.05840822, 0.0252302 , 0.19246724},
	// 		{0.13628861, 0.00449609, 0.16012484},
	// 		{0.14277523, 0.18742742, 0.00637526},
	// 		{0.04304032, 0.03896925, 0.07446333},
	// 		{0.18767841, 0.17126958, 0.06172466},
	// 		{0.08524866, 0.15391154, 0.12304257},
	// 		{0.07985594, 0.11248441, 0.05429952},
	// 		{0.03933822, 0.12213914, 0.11393107},
	// 		{0.11872013, 0.16267809, 0.06804414},
	// 		{0.11540472, 0.15481414, 0.02670733},
	// 		{0.09190882, 0.06792627, 0.07202206},
	// 		{0.03766591, 0.04241826, 0.13449904},
	// 		{0.17323557, 0.01047467, 0.16031764},
	// 		{0.19483602, 0.08792233, 0.03359821},
	// 		{0.0721314 , 0.10985111, 0.17533249},
	// 		{0.03509212, 0.06860487, 0.11946439},
	// 		{0.06895616, 0.05280851, 0.12040593},
	// 		{0.16917599, 0.1998228 , 0.12220075},
	// 		{0.13888666, 0.14233694, 0.18931696},
	// 		{0.06320079, 0.01907374, 0.06447894},
	// 		{0.14576798, 0.02359065, 0.147211  },
	// 		{0.19115252, 0.02040591, 0.14838246},
	// 		{0.02997318, 0.12616056, 0.03841205},
	// 		{0.02513433, 0.00916928, 0.09003969}
	// 	}
	// };// (1, 24, 3)


	xt::xarray<double> vertices;

	try {
		SINGLE_SMPL::get()->setModelPath(modelPath);

		auto begin = clk::now();
		SINGLE_SMPL::get()->init();
		auto end = clk::now();
		auto duration = std::chrono::duration_cast<ms>(end - begin);
		std::cout << "Time duration to load SMPL: " 
			<< (double)duration.count() / 1000 << "s" << std::endl;

		begin = clk::now();
		SINGLE_SMPL::get()->launch(beta, theta);
		end = clk::now();
		duration = std::chrono::duration_cast<ms>(end - begin);
		std::cout << "Time duration to run SMPL: " 
			<< (double)duration.count() / 1000 << "s" << std::endl;

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

	// auto data = xt::load_npz("../data/smpl_female.npz");
	// xt::xarray<double> templateRestShape = 
	// 	data["vertices_template"].cast<double>();
	// xt::xarray<double> faceIndices = 
	// 	data["face_indices"].cast<double>();
	// xt::xarray<double> jointRegressor = 
	// 	data["joint_regressor"].cast<double>();
	// xt::xarray<uint32_t> kinematicTree = 
	// 	data["kinematic_tree"].cast<uint32_t>();
	// xt::xarray<double> weights = 
	// 	data["weights"].cast<double>();
	// xt::xarray<double> poseBlendBasis = 
	// 	data["pose_blend_shapes"].cast<double>();
	// xt::xarray<double> shapeBlendBasis = 
	// 	data["shape_blend_shapes"].cast<double>();
	
	// std::cout << kinematicTree << std::endl;

    return 0;
}


//===== CLEAN AFTERWARD =======================================================

#undef SINGLE_SMPL

//=============================================================================
