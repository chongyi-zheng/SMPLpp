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

For more details, see the [paper](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) published by Max Planck Institute for
Intelligent Systems on SIGGRAPH ASIA 2015.

## Prerequisite

The code in this project have been tested on my machine. I'm not sure
how the performance will be on other system environments.

- GPU

  NVIDIA GeForce GTX 960M

- OS

  Ubuntu 18.04 LTS

- Packages

  - [xtensor](https://github.com/QuantStack/xtensor): A C++ library meant for numerical analysis with multi-dimensional array expressions. 
  
    This library are inspired by `NumPy` such that you can even find functions with same names in it.
    There is also a cheat sheet from `Numpy` to `xtensor` in its [documentation](https://xtensor.readthedocs.io/en/latest/).

    Currently, I only use `xtensor` as a IO interface to define random inputs for module test and restore hyperparameters from JSON format. The data in the buffer of a `xtensor` array can be fed into a corresponding "torch tensor" later.
  
  - [nlohmann_json](https://github.com/nlohmann/json): JSON for Modern C++.

    `xtensor` loads and dumps data to json, using the json library written by nlohmann.

  - [libTorch](https://pytorch.org/cppdocs/): Pytorch C++ API.

    PyTorch C++ API simplifies tensor computing and introduces GPU acceleration to this work, using CUDA and cuDNN.

    Note: I installed nightly version of `libTorch` with `CUDA 10.0` support.

  - [CUDA](https://developer.nvidia.com/cuda-downloads): NVIDIA parallel computing platform.
  
    Version 10.0 has been tested on my machine, but I think the corresponding versions within the `libTorch` download list are all available.

  - [CMake](https://cmake.org/download/): Tool to build, test and package a C++ software.
  
    The `CMake` installed through `apt` is `CMake 3.5.1` which causes a failure when `libTorch` tries to find `CUDA`. Here is a [description](https://discuss.pytorch.org/t/install-libtorch-error-pytorch-c-api/26756/2) about the problem.
  
    You should update it to a newer version, say 3.13.4.
    You can delete the old `CMake`, download the newer source code on official website and build it from scratch.  

## Usage


## Features


## Documentation


## TODO

- [ ] Hyperparameters restore from `npz` files instead of `json` files. (`json` neither saves storage nor performances efficiently when being imported.)

- [ ] A OpenGL GUI to render and manipulate the mesh.

- [ ] Fit the 3D mesh to a 2D image - SMPLify.

- [ ] A trainable SMPL.

Note: The importance of each demand decreases in this list.

## Misc

If you find any problem, error or even typo, feel free to contact me directly.

Currently, this project is for research purpose, any commercial usage should be allowed by original authors.

## Reference

[1] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. "SMPL: a skinned multi-person linear model". ACM Trans. Graph. 34, 6, Article 248 (October 2015), 16 pages.

[2] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, Michael J. Black. "Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image". Lecture Notes in Computer Science (2016): 561–578. Crossref. Web.

[3] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik. "End-to-end Recovery of Human Shape and Pose". Computer Vision and Pattern Regognition (CVPR) 2018.

[4] Official Website of SMPL: smpl.is.tue.mpg.de/.

[5] Official Website of SMPLify: smplify.is.tue.mpg.de/.

[6] Official Website of HMR: https://akanazawa.github.io/hmr/.

[7] Github Project Page for End-to-end Recovery of Human Shape and Pose: https://github.com/akanazawa/hmr.
