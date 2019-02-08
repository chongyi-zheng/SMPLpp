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
#include <xtensor/xview.hpp>
//----------
#include "definition/def.h"
#include "smpl/SMPL.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

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
    __modelPath(),
    __vertPath(),
    __model(),
    __blender(),
    __regressor(),
    __transformer(),
    __skinner()
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
SMPL::SMPL(std::string modelPath, std::string vertPath) noexcept(false) :
    __model(),
    __blender(),
    __regressor(),
    __transformer(),
    __skinner()
{
    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        __modelPath = modelPath;
        __vertPath = vertPath;
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
SMPL::SMPL(const SMPL& smpl) noexcept(false)
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
 *      Deconstructor
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
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 * 
 */
SMPL &SMPL::operator=(const SMPL& smpl) noexcept(false)
{
    //
    // hard copy
    //
    std::experimental::filesystem::path path(smpl.__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        __modelPath = smpl.__modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to copy model path!");
    }

    try {
        __vertPath = smpl.__vertPath;

        __model = smpl.__model;
        __blender = smpl.__blender;
        __regressor = smpl.__regressor;
        __transformer = smpl.__transformer;
        __skinner = smpl.__skinner;
    }
    catch(std::exception &e) {
        throw;
    }
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
void SMPL::setModelPath(std::string modelPath) noexcept(false)
{
    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        __modelPath = modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }
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
void SMPL::setVertPath(std::string vertexPath) noexcept(false)
{
    __vertPath = vertexPath;
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
 *      @restShape: - xarray -
 *          Deformed shape in rest pose, (N, 6890, 3)
 * 
 */
xt::xarray<double> SMPL::getRestShape() noexcept(false)
{
    xt::xarray<double> restShape;
    
    try {
        restShape = __regressor.getRestShape();
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
 *      @faceIndices: - xarray -
 *          Vertex indices of each face (triangles), (13776, 3).
 * 
 */
xt::xarray<uint32_t> SMPL::getFaceIndex() noexcept(false)
{
    xt::xarray<uint32_t> faceIndices;

    if (__model.is_null()) {
        throw smpl_error("SMPL", "Failed to get face indices!");
    }
    else {
        xt::from_json(__model["face_indices"], faceIndices);
    }

    if (faceIndices.shape() !=
        xt::xarray<uint32_t>::shape_type(
            {FACE_INDEX_NUM, 3})) {
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
 *      @joints: - xarray -
 *          Joint locations of the deformed mesh in rest pose, (N, 24, 3).
 * 
 */
xt::xarray<double> SMPL::getRestJoint() noexcept(false)
{
    xt::xarray<double> joints;
    
    try {
        joints = __regressor.getJoint();
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
 *      @vertices: - xarray -
 *          Vertex locations of the deformed mesh, (N, 6890, 3).
 * 
 */
xt::xarray<double> SMPL::getVertex() noexcept(false)
{
    xt::xarray<double> vertices;

    try {
        vertices = __skinner.getVertex();
    }
    catch(std::exception &e) {
        throw;
    }

    return vertices;
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
    std::experimental::filesystem::path path(__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        std::ifstream file(path);
        file >> __model;
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
 *      @beta: - xarray -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 *      @theta: - xarray -
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 * 
 *      @translation: - xarray -
 *          Batch of global translation vectors, (N, 3).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void SMPL::launch(
    xt::xarray<double> beta, 
    xt::xarray<double> theta) noexcept(false)
{
    if (__model.is_null()
        && beta.shape() !=
            xt::xarray<double>::shape_type({BATCH_SIZE, SHAPE_BASIS_DIM})
        && theta.shape() != 
            xt::xarray<double>::shape_type({BATCH_SIZE, JOINT_NUM, 3})
        ) {
        throw smpl_error("SMPL", "Cannot launch a SMPL model!");
    }

    try {
        //
        // blend shapes
        //
        xt::xarray<double> shapeBlendBasis;
        xt::xarray<double> poseBlendBasis;
        xt::from_json(__model["shape_blend_shapes"], shapeBlendBasis);
        xt::from_json(__model["pose_blend_shapes"], poseBlendBasis);

        __blender.setBeta(beta);
        __blender.setTheta(theta);
        __blender.setShapeBlendBasis(shapeBlendBasis);
        __blender.setPoseBlendBasis(poseBlendBasis);

        __blender.blend();

        xt::xarray<double> shapeBlendShape = __blender.getShapeBlendShape();
        xt::xarray<double> poseBlendShape = __blender.getPoseBlendShape();
        xt::xarray<double> poseRotation = __blender.getPoseRotation();

        //
        // regress joints
        //
        xt::xarray<double> templateRestShape;
        xt::xarray<double> jointRegressor;
        xt::from_json(__model["vertices_template"], templateRestShape);
        xt::from_json(__model["joint_regressor"], jointRegressor);

        __regressor.setTemplateRestShape(templateRestShape);
        __regressor.setJointRegressor(jointRegressor);
        __regressor.setShapeBlendShape(shapeBlendShape);
        __regressor.setPoseBlendShape(poseBlendShape);

        __regressor.regress();

        xt::xarray<double> restShape = __regressor.getRestShape();
        xt::xarray<double> joints = __regressor.getJoint();

        //
        // transform
        //
        xt::xarray<uint32_t> kinematicTree;
        xt::from_json(__model["kinematic_tree"], kinematicTree);

        __transformer.setKinematicTree(kinematicTree);
        __transformer.setJoint(joints);
        __transformer.setPoseRotation(poseRotation);

        __transformer.transform();

        xt::xarray<double> transformation = __transformer.getTransformation();

        //
        // skinning
        //
        xt::xarray<double> weights;
        xt::from_json(__model["weights"], weights);

        __skinner.setWeight(weights);
        __skinner.setRestShape(restShape);
        __skinner.setTransformation(transformation);

        __skinner.skinning();
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
void SMPL::out(size_t index) noexcept(false)
{
    xt::xarray<double> vertices = __skinner.getVertex();
    xt::xarray<uint32_t> faceIndices;
    xt::from_json(__model["face_indices"], faceIndices);
    if (vertices.shape() ==
            xt::xarray<double>::shape_type(
                {BATCH_SIZE, VERTEX_NUM, 3})
        && faceIndices.shape() ==
            xt::xarray<uint32_t>::shape_type(
                {FACE_INDEX_NUM, 3})
        ) {
        std::ofstream file(__vertPath);

        xt::xarray<double> slice = xt::view(vertices, index);
        for (size_t i = 0; i < VERTEX_NUM; i++) {
            file << 'v' << ' '
                << vertices(i, 0) << ' '
                << vertices(i, 1) << ' ' 
                << vertices(i, 2) << '\n';
        }

        for (size_t i = 0; i < FACE_INDEX_NUM; i++) {
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


//=============================================================================
} // namespace smpl
//=============================================================================
