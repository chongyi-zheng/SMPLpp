# SMPL++

A C++ Implementation of SMPL - A Skinned Multi-Person Linear Model.

![SMPL_Modle](docs/media/front_page.png)

## Overview

This project implements a 3D human skinning model - SMPL: A Skinned
Multi-Person Linear Model with C++. The official SMPL model is 
available at http://smpl.is.tue.mpg.de.

The author-provided implementation based on `chumpy` and `opendr` contains
spaghetti code,
and it cannot run on GPUs yet. I convert and modify another Tensorflow 
version of SMPL contributed by [CalciferZh](https://github.com/CalciferZh) to 
C++ style.
You can find the Tensorflow implementation 
[here](https://github.com/CalciferZh/SMPL).
However, Tensorflow C++ APIs are not user-friendly, so I choose the Pytorch 
C++ APIs - libTorch - instead.

For more details, see the [paper](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) 
published by Max Planck Institute for Intelligent Systems on SIGGRAPH ASIA 
2015.

## Prerequisite

The code in this project have been tested on my machine. I'm not sure
how the performance will be on other system environments.

- GPU

  NVIDIA GeForce GTX 960M

- OS

  Ubuntu 18.04 LTS

- Packages

1. [xtensor](https://github.com/QuantStack/xtensor): A C++ library meant for 
   numerical analysis with multi-dimensional array expressions. 
  
    This library are inspired by `numpy` such that you can even find 
    functions with same names in it.
    There is also a cheat sheet from `numpy` to `xtensor` in its 
    [documentation](https://xtensor.readthedocs.io/en/latest/).

    Currently, I only use `xtensor` as a IO interface to define random inputs 
    for module test and restore hyperparameters from JSON format. The data in 
    the buffer of a `xtensor` array can be fed into a corresponding "torch 
    tensor" later.
  
2. [nlohmann_json](https://github.com/nlohmann/json): JSON for Modern C++.

    `xtensor` loads and dumps data to json, using the json library written by 
    nlohmann.

3. [libTorch](https://pytorch.org/get-started/locally/): Pytorch C++ API.

    PyTorch C++ API simplifies tensor computing and introduces GPU 
    acceleration to this work, using `CUDA` and `cuDNN`.

    Note: I installed nightly version of `libTorch` with `CUDA 10.0` support.

4. [CUDA](https://developer.nvidia.com/cuda-downloads): NVIDIA parallel 
   computing platform.
  
    Version 10.0 has been tested on my machine, but I think the corresponding 
    versions within the `libTorch` download list are all available.

5. [CMake](https://cmake.org/download/): A tool to build, test and package 
   C++ softwares.
  
    The `CMake` installed by `apt-get` is `CMake 3.5.1` which causes a 
    failure when `libTorch` tries to find `CUDA`. Here is a 
    [description](https://discuss.pytorch.org/t/install-libtorch-error-pytorch-c-api/26756/2) 
    about the problem.
  
    You should update it to a newer version, say 3.13.4 (>=3.12.2 should work).
    Delete the old `CMake`, download a newer source code on official website 
    and build it from scratch.

## Usage

- Package Installation

  I only want to talk about tricks to install packages into root directory 
  correctly. If you want to link packages manually, just skip this part and 
  install in the normal ways.

  To install packages successfully, follow the instructions in their official 
  documentations or just search on google.

  - Source Code

    Compile libraries and headers into root directory, e.g. `xtensor`:

  1. change to `xtensor` repo that you have cloned from github.

          cd <xtensor-dir>

  2. create a directory to compile the package and change to it.

          mkdir build
          cd build

  3. configure `CMake` and generate makefiles. Remember to attach the
     installation directory to the root.

          cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..

      You can change "/usr/local" to other root location as long as `CMake` 
      is able to find the package automatically.

  4. compile and install.

          make
          sudo make install

  - Binary Library
    
    Move pre-built packages into root directory, e.g. `libTorch`:

  1. change to `libTorch` directory.

          cd <libTorch-dir>

  2. copy `CMake` configurations to "lib" directory.

          cp -rv share/cmake lib

  3. copy headers and libraries to root directory.

          cp -rv include lib /usr/local

      Afterwards, you don't need to specify path to the `libTorch` library 
      when building `libTorch`-dependent projects with `CMake`. This trick 
      is a little bit different from the guide in official `libTorch` 
      documentation.

- Build and Run

  After installing all packages, you can compile SMPL++ from source:

      mkdir build
      cd build
      cmake ..
      make

  These command will produce a executable program named "smplpp". To run it, 
  just type

      ./smplpp
  
  in the terminal.

  To track the usage of GPU, use following command:

      nvidia-smi -lms

## Instructions

Now, here is only a raw framework for SMPL++. I have written a lot of comments
in the source code, you can just check them directly.

- Forward Propagation

  SMPL++ implement the model described in the paper. The inputs of the system
  are shape coefficients $\beta$, pose axis-angle parameterization $\theta$ 
  and body translation $\vec{t}$. Change them to get different meshes.

  <img src="docs/media/example.png" alt="drawing" width="400"/>

  Note: Backward propagation hasn't been implemented yet.

- Render Meshes

  We don't have a GUI to render the output now! If you would like to see the 
  meshes, try to render them in [MeshLab](http://www.meshlab.net/).

- Pipeline

  Following the paper, we can generate a mesh within four steps.
  See documentations in "docs" for more details.

  1. Generate pose blend shape and shape blend shape.

  2. Regress joints from vertices.

  3. Compute transformation matrix for each joint.

  4. Linear Blend Skinning

- Kinematic Tree

  Finally, we have an kinematic tree for SMPL model:

  <img src="docs/media/kinematic_tree.png" alt="drawing" width="500"/>

## TODO

- [ ] Hyperparameters restore from `npz` files instead of `json` files. 
      (`json` neither saves storage nor performances efficiently when being 
      imported.)

- [ ] A OpenGL GUI to render and manipulate the 3D mesh.

- [ ] Fit the 3D mesh to a 2D image - SMPLify.

- [ ] Export SMPL++ into static or dynamic library.

- [ ] A trainable SMPL.

Note: The importance of each demand decreases in this list.

## Misc

If you find any problem, error or even typo, feel free to contact me directly.

Currently, SMPL++ is for research purpose, any commercial usage should be allowed by original authors.

## Reference

[1] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. "SMPL: a skinned multi-person linear model". ACM Trans. Graph. 34, 6, Article 248 (October 2015), 16 pages.

[2] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, Michael J. Black. "Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image". Lecture Notes in Computer Science (2016): 561–578. Crossref. Web.

[3] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik. "End-to-end Recovery of Human Shape and Pose". Computer Vision and Pattern Recognition (CVPR) 2018.

[4] Official Website of SMPL: http://smpl.is.tue.mpg.de.

[5] Official Website of SMPLify: http://smplify.is.tue.mpg.de.

[6] Official Website of HMR: https://akanazawa.github.io/hmr/.

[7] Github Project Page for End-to-end Recovery of Human Shape and Pose: https://github.com/akanazawa/hmr.
