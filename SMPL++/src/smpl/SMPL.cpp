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
//  CLASS SMPL IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <fstream>
#include <experimental/filesystem>
//----------
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xjson.hpp>
//----------
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include "smpl/SMPL.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

namespace {
const torch::Tensor kEmptyTensor = torch::Tensor();
}

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**SMPL
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
SMPL::SMPL() noexcept(true) :
    m__device(torch::kCPU),
    m__modelPath(),
    m__vertPath(),
    m__faceIndices(),
    m__shapeBlendBasis(),
    m__poseBlendBasis(),
    m__templateRestShape(),
    m__jointRegressor(),
    m__kinematicTree(),
    m__weights(),
    m__model(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
}

/**SMPL (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize model path.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::SMPL(std::string &modelPath, 
    std::string &vertPath, torch::Device &device) noexcept(false) :
    m__device(torch::kCPU),
    m__model(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = modelPath;
        m__vertPath = vertPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }
}

/**SMPL (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const SMPL& -
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL::SMPL(const SMPL& smpl) noexcept(false) :
    m__device(torch::kCPU)
{
    try {
        *this = smpl;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~SMPL
 * 
 * Brief
 * ----------
 * 
 *      Destructor
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
SMPL::~SMPL() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <SMPL> instantiation.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const SMPL& -
 *          The <SMPL> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @this*: - SMPL & -
 *          Current instantiation.
 * 
 */
SMPL &SMPL::operator=(const SMPL& smpl) noexcept(false)
{
    //
    // hard copy
    //
    if (smpl.m__device.has_index()) {
        m__device = smpl.m__device;
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    std::experimental::filesystem::path path(smpl.m__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = smpl.m__modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to copy model path!");
    }

    try {
        m__vertPath = smpl.m__vertPath;

        m__model = smpl.m__model;
        m__blender = smpl.m__blender;
        m__regressor = smpl.m__regressor;
        m__transformer = smpl.m__transformer;
        m__skinner = smpl.m__skinner;
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // soft copy
    //
    if (smpl.m__faceIndices.sizes() ==
        torch::IntArrayRef({FACE_INDEX_NUM, 3})) {
        m__faceIndices = smpl.m__faceIndices.clone().to(m__device);
    }

    if (smpl.m__shapeBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = smpl.m__shapeBlendBasis.clone().to(
            m__device);
    }

    if (smpl.m__poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, POSE_BASIS_DIM})) {
        m__poseBlendBasis = smpl.m__poseBlendBasis.clone().to(m__device);
    }

    if (smpl.m__jointRegressor.sizes() == 
        torch::IntArrayRef({JOINT_NUM, VERTEX_NUM})) {
        m__jointRegressor = smpl.m__jointRegressor.clone().to(m__device);
    }

    if (smpl.m__templateRestShape.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__templateRestShape = smpl.m__templateRestShape.clone().to(
            m__device);
    }

    if (smpl.m__kinematicTree.sizes() ==
        torch::IntArrayRef({2, JOINT_NUM})) {
        m__kinematicTree = smpl.m__kinematicTree.clone().to(m__device);
    }

    if (smpl.m__weights.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, JOINT_NUM})) {
        m__weights = smpl.m__weights.clone().to(m__device);
    }

    return *this;
}

/**setDevice
 * 
 * Brief
 * ----------
 * 
 *      Set the torch device.
 * 
 * Arguments
 * ----------
 * 
 *      @device: - const Device & -
 *          The torch device to be used.
 * 
 * Return
 * ----------
 * 
 */
void SMPL::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
        m__blender.setDevice(device);
        m__regressor.setDevice(device);
        m__transformer.setDevice(device);
        m__skinner.setDevice(device);
    }
    else {
        throw smpl_error("SMPL", "Failed to fetch device index!");
    }

    return;
}

/**setModelPath
 * 
 * Brief
 * ----------
 * 
 *      Set model path to the JSON model file.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::setModelPath(const std::string &modelPath) noexcept(false)
{
    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }

    return;
}

/**setVertPath
 * 
 * Brief
 * ----------
 * 
 *      Set path for exporting the mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @vertexPath: - string -
 *          Vertex path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::setVertPath(const std::string &vertexPath) noexcept(false)
{
    m__vertPath = vertexPath;

    return;
}

/**getRestShape
 * 
 * Brief
 * ----------
 * 
 *      Get deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @restShape: - Tensor -
 *          Deformed shape in rest pose, (N, 6890, 3)
 * 
 */
torch::Tensor SMPL::getRestShape() noexcept(false)
{
    torch::Tensor restShape;
    
    try {
        restShape = m__regressor.getRestShape().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return restShape;
}

/**getFaceIndex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex indices of each face.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @faceIndices: - Tensor -
 *          Vertex indices of each face (triangles), (13776, 3).
 * 
 */
torch::Tensor SMPL::getFaceIndex() noexcept(false)
{
    torch::Tensor faceIndices;
    if (m__faceIndices.sizes() ==
        torch::IntArrayRef(
            {FACE_INDEX_NUM, 3})) {
        faceIndices = m__faceIndices.clone().to(m__device);
    }
    else {
        throw smpl_error("SMPL", "Failed to get face indices!");
    }

    return faceIndices;
}

/**getRestJoint
 * 
 * Brief
 * ----------
 * 
 *      Get joint locations of the deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @joints: - Tensor -
 *          Joint locations of the deformed mesh in rest pose, (N, 24, 3).
 * 
 */
torch::Tensor SMPL::getRestJoint() noexcept(false)
{
    torch::Tensor joints;
    
    try {
        joints = m__regressor.getJoint().clone().to(m__device);
    }
    catch (std::exception &e) {
        throw;
    }

    return joints;
}

/**getVertex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex locations of the deformed mesh.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @vertices: - Tensor -
 *          Vertex locations of the deformed mesh, (N, 6890, 3).
 * 
 */
torch::Tensor SMPL::getVertex() noexcept(false)
{
    torch::Tensor vertices;

    try {
        vertices = m__skinner.getVertex().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return vertices;
}

torch::Tensor SMPL::getExtra() noexcept {
    return m__skinner.getExtra().clone().to(m__device);
}

/**init
 * 
 * Brief
 * ----------
 * 
 *          Load model data stored as JSON file into current application.
 *          (Note: The loading will spend a long time because of a large
 *           JSON file.)
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
void SMPL::init() noexcept(false)
{
    std::experimental::filesystem::path path(m__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        std::ifstream file(path);
        file >> m__model;

        //
        // data loading
        //
        // face indices
        xt::xarray<int32_t> faceIndices;
        xt::from_json(m__model["face_indices"], faceIndices);
        m__faceIndices = torch::from_blob(faceIndices.data(),
            {FACE_INDEX_NUM, 3}, torch::kInt32).clone().to(
                m__device);

        // blender
        xt::xarray<float> shapeBlendBasis;
        xt::xarray<float> poseBlendBasis;
        xt::from_json(m__model["shape_blend_shapes"], shapeBlendBasis);
        xt::from_json(m__model["pose_blend_shapes"], poseBlendBasis);
        m__shapeBlendBasis = torch::from_blob(shapeBlendBasis.data(),
            {VERTEX_NUM, 3, SHAPE_BASIS_DIM}).to(m__device);// (6890, 3, 10)
        m__poseBlendBasis = torch::from_blob(poseBlendBasis.data(),
            {VERTEX_NUM, 3, POSE_BASIS_DIM}).to(m__device);// (6890, 3, 207)

        // regressor
        xt::xarray<float> templateRestShape;
        xt::xarray<float> jointRegressor;
        xt::from_json(m__model["vertices_template"], templateRestShape);
        xt::from_json(m__model["joint_regressor"], jointRegressor);
        m__templateRestShape = torch::from_blob(templateRestShape.data(),
            {VERTEX_NUM, 3}).to(m__device);// (6890, 3)
        m__jointRegressor = torch::from_blob(jointRegressor.data(),
            {JOINT_NUM, VERTEX_NUM}).to(m__device);// (24, 6890)

        // transformer
        xt::xarray<int64_t> kinematicTree;
        xt::from_json(m__model["kinematic_tree"], kinematicTree);
        m__kinematicTree = torch::from_blob(kinematicTree.data(),
            {2, JOINT_NUM}, torch::kInt64).to(m__device);// (2, 24)

        // skinner
        xt::xarray<float> weights;
        xt::from_json(m__model["weights"], weights);
        m__weights = torch::from_blob(weights.data(),
            {VERTEX_NUM, JOINT_NUM}).to(m__device);// (6890, 24)
    }
    else {
        throw smpl_error("SMPL", "Cannot initialize a SMPL model!");
    }

    return;
}

/**launch
 * 
 * Brief
 * ----------
 * 
 *          Run the model with a specific group of beta, theta, and 
 *          translation.
 * 
 * Arguments
 * ----------
 * 
 *      @beta: - Tensor -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 *      @theta: - Tensor -
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 * 
 *      @translation: - Tensor -
 *          Batch of global translation vectors, (N, 3).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::launch(
    torch::Tensor &beta, 
    torch::Tensor &theta,
    std::optional<torch::Tensor> &extra) noexcept(false)
{
    if (m__model.is_null()
        && beta.sizes() !=
            torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})
        && theta.sizes() != 
            torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})
        ) {
        throw smpl_error("SMPL", "Cannot launch a SMPL model!");
    }

    try {
        //
        // blend shapes
        //
        m__blender.setBeta(beta);
        m__blender.setTheta(theta);
        m__blender.setShapeBlendBasis(m__shapeBlendBasis);
        m__blender.setPoseBlendBasis(m__poseBlendBasis);

        m__blender.blend();

        torch::Tensor shapeBlendShape = m__blender.getShapeBlendShape();
        torch::Tensor poseBlendShape = m__blender.getPoseBlendShape();
        torch::Tensor poseRotation = m__blender.getPoseRotation();

        //
        // regress joints
        //
        m__regressor.setTemplateRestShape(m__templateRestShape);
        m__regressor.setJointRegressor(m__jointRegressor);
        m__regressor.setShapeBlendShape(shapeBlendShape);
        m__regressor.setPoseBlendShape(poseBlendShape);

        m__regressor.regress();

        torch::Tensor restShape = m__regressor.getRestShape();
        torch::Tensor joints = m__regressor.getJoint();

        //
        // transform
        //
        m__transformer.setKinematicTree(m__kinematicTree);
        m__transformer.setJoint(joints);
        m__transformer.setPoseRotation(poseRotation);

        m__transformer.transform();

        torch::Tensor transformation = m__transformer.getTransformation();

        //
        // skinning
        //

        m__skinner.setWeight(m__weights);
        m__skinner.setRestShape(restShape);
        m__skinner.setTransformation(transformation);
        if(extra.has_value()) {
            m__skinner.setExtra(*extra + shapeBlendShape + poseBlendShape);
        } else {
            m__skinner.setExtra(kEmptyTensor);
        }

        m__skinner.skinning();
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

/**out
 * 
 * Brief
 * ----------
 * 
 *      Export the deformed mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @index: - size_t -
 *          A mesh in the batch to be exported.
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::out(int64_t index) noexcept(false)
{
    torch::Tensor vertices = 
        m__skinner.getVertex().clone().to(m__device);// (N, 6890, 3)

    if (vertices.sizes() ==
            torch::IntArrayRef(
                {BATCH_SIZE, VERTEX_NUM, 3})
        && m__faceIndices.sizes() ==
            torch::IntArrayRef(
                {FACE_INDEX_NUM, 3})
        ) {
        std::ofstream file(m__vertPath);

        torch::Tensor slice_ = TorchEx::indexing(vertices,
            torch::IntList({index}));// (6890, 3)
        xt::xarray<float> slice = xt::adapt(
            (float *)slice_.to(torch::kCPU).data_ptr(),
            xt::xarray<float>::shape_type({(const size_t)VERTEX_NUM, 3})
        );
        
        xt::xarray<int32_t> faceIndices;
        faceIndices = xt::adapt(
            (int32_t *)m__faceIndices.to(torch::kCPU).data_ptr(),
            xt::xarray<int32_t>::shape_type(
                {(const size_t)FACE_INDEX_NUM, 3})
        );

        for (int64_t i = 0; i < VERTEX_NUM; i++) {
            file << 'v' << ' '
                << slice(i, 0) << ' '
                << slice(i, 1) << ' ' 
                << slice(i, 2) << '\n';
        }

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++) {
            file << 'f' << ' '
                << faceIndices(i, 0) << ' '
                << faceIndices(i, 1) << ' '
                << faceIndices(i, 2) << '\n';
        }
    }
    else {
        throw smpl_error("SMPL", "Cannot export the deformed mesh!");
    }

    return;
}

torch::Tensor SMPL::getOffset() const {
    return m__regressor.getShapeTransformation();
}

torch::Tensor SMPL::getSkinning() const {
    return m__skinner.getSkinningTransformation();
}


//=============================================================================
} // namespace smpl
//=============================================================================
