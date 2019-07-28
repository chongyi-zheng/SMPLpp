# Copyright 2018 Chongyi Zheng. All rights reserved.
#
# This software implements a 3D human skinning model, SMPL, with tensorflow
# and numpy.
# For more detail, see the paper - SMPL: A Skinned Multi-Person Linear Model -
# published by Max Planck Institute for Intelligent Systems on SIGGRAPH ASIA 2015.
#
# Here we provide the software for research purposes only.
# More information about SMPL is available on http://smpl.is.tue.mpg.
#
# ============================= preprocess.py =================================
# File Description:
#
# This file loads the models downloaded from the official SMPL website, grab
# data and write them in to numpy and json format.
#
# =============================================================================
#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pickle as pkl
import json


def main(args):
    """Main entrance.

    Arguments
    ----------
    - args: list of strings
        Command line arguments.

    Returns
    ----------

    """
    gender = args[1]
    raw_model_path = args[2]
    save_dir = args[3]

    if gender == 'female':
        NP_SAVE_FILE = 'smpl_female.npz'
        JSON_SVAE_FILE = 'smpl_female.json'
    elif gender == 'male':
        NP_SAVE_FILE = 'smpl_male.npz'
        JSON_SVAE_FILE = 'smpl_male.json'
    else:
        raise SystemError('Please specify gender of the model!\n'
                          'USAGE: \'*f*.pkl\' - female, '
                          '\'*m*.pkl\' - male')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np_save_path = os.path.join(save_dir, NP_SAVE_FILE)
    json_save_path = os.path.join(save_dir, JSON_SVAE_FILE)

    # * Model Data Description * #
    # vertices_template: global vertex locations of template - (6890, 3)
    # face_indices: vertex indices of each face (triangles) - (13776, 3)
    # joint_regressor: joint regressor - (24, 6890)
    # kinematic_tree_table: table of kinematic tree - (2, 24)
    # weights: weights for linear blend skinning - (6890, 24)
    # shape_blend_shapes: shape blend shapes - (6890, 3, 10)
    # pose_blend_shapes: pose blend shapes - (6890, 3, 207)

    # * Extra Data Description *
    # Besides the data above, the official model provide the following things.
    # The pickle file downloaded from SMPL website seems to be redundant or
    # some of the contents are used for training the model. None of them will
    # be used to generate a new skinning.
    #
    # bs_stype: blend skinning style - (default)linear blend skinning
    # bs_type: blend skinning type - (default) linear rotation minimization
    # J: global joint locations of the template mesh - (24, 3)
    # J_regressor_prior: prior joint regressor - (24, 6890)
    # pose_training_info: pose training information - string list with 6
    #                     elements.
    # vert_sym_idxs: symmetrical corresponding vertex indices - (6890, )
    # weights_prior: prior weights for linear blend skinning
    with open(raw_model_path, 'rb') as f:
        raw_model_data = pkl.load(f)
    vertices_template = np.array(raw_model_data['v_template'])
    face_indices = np.array(raw_model_data['f'] + 1)  # starts from 1
    weights = np.array(raw_model_data['weights'])
    shape_blend_shapes = np.array(raw_model_data['shapedirs'])
    pose_blend_shapes = np.array(raw_model_data['posedirs'])
    joint_regressor = np.array(raw_model_data['J_regressor'].toarray())
    kinematic_tree = np.array(raw_model_data['kintree_table'])

    model_data_np = {
        'vertices_template': vertices_template,
        'face_indices': face_indices,
        'weights': weights,
        'shape_blend_shapes': shape_blend_shapes,
        'pose_blend_shapes': pose_blend_shapes,
        'joint_regressor': joint_regressor,
        'kinematic_tree': kinematic_tree
    }

    # Data must be converted to list before storing as json.
    model_data_json = {
        'vertices_template': vertices_template.tolist(),
        'face_indices': face_indices.tolist(),
        'weights': weights.tolist(),
        'shape_blend_shapes': shape_blend_shapes.tolist(),
        'pose_blend_shapes': pose_blend_shapes.tolist(),
        'joint_regressor': joint_regressor.tolist(),
        'kinematic_tree': kinematic_tree.tolist()
    }

    np.savez(np_save_path, **model_data_np)
    with open(json_save_path, 'wb+') as f:
        json.dump(model_data_json, f, indent=4, sort_keys=True)

    print('Save SMPL Model to: ', os.path.abspath(save_dir))


if __name__ == '__main__':
    if sys.version_info[0] != 2:
        raise EnvironmentError('Run this file with Python2!')
    if len(sys.argv) < 4:
        raise SystemError('Too few arguments!\n'
                          'USAGE: python2 preprocess.py '
                          '<gender> <path-to-the-pkl> '
                          '<dir-to-the-model>')
    elif len(sys.argv) > 4:
        raise SystemError('Too many arguments, only one model at a time!\n'
                          'USAGE: python2 preprocess.py '
                          '<path-to-the-pkl> <dir-to-the-model>')

    main(sys.argv)
