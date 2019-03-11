# SMPL++

A C++ Implementation of SMPL - A Skinned Multi-Person Linear Model.

![SMPL_Modle](docs/media/SMPL_model.png)

## Overview

This project implements a 3D human skinning model - SMPL: A Skinned
Multi-Person Linear Model with C++. The official SMPL model is 
available at smpl.is.tue.mpg.de/.

The author-provided implementation based on `chumpy` and `opendr` contains
spaghetti code,
and it cannot run on GPUs yet. I convert another Tensorflow version of SMPL contributed by [CalciferZh](https://github.com/CalciferZh) to C++ style in
this project.
You can find it [here](https://github.com/CalciferZh/SMPL).
However, Tensorflow C++ APIs are not user-friendly, so I choose the Pytorch C++ APIs - libTorch - instead.

For more detail, see the paper published by Max Planck Institute for
Intelligent Systems on SIGGRAPH ASIA 2015.

## Prerequisite

The code in this project have been tested on my machine. I'm not sure
how the performance will be on other system environments.

- GPU

  NVIDIA GeForce GTX 960M

- OS

  Ubuntu 18.04 LTS

- Packages

## Usage


## Features


## Documentation


## TODO

- [ ] Hyperparameters restore from `npz` files instead of `json` files. (`json` neither saves storage nor performances efficiently when being imported.)

- [ ] Fit the 3D mesh to a 2D image - SMPLify.

- [ ] A trainable SMPL.

Note: The importance of each demand decreases in this list.

## Misc


## Reference

[1] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. "SMPL: a skinned multi-person linear model". ACM Trans. Graph. 34, 6, Article 248 (October 2015), 16 pages.

[2] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, Michael J. Black. "Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image". Lecture Notes in Computer Science (2016): 561–578. Crossref. Web.

[3] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik. "End-to-end Recovery of Human Shape and Pose". Computer Vision and Pattern Regognition (CVPR) 2018.

[4] Github Project Page for End-to-end Recovery of Human Shape and Pose: https://github.com/akanazawa/hmr.

[5] Tensorflow implementation of SMPL: https://github.com/CalciferZh.

[6] Official Website of SMPL: smpl.is.tue.mpg.de/.

[7] Official Website of SMPLify: smplify.is.tue.mpg.de/.

[8] Official Website of HMR: https://akanazawa.github.io/hmr/.
