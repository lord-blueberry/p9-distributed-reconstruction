# p9-distributed-reconstruction

This repository holds the code of the Master-Thesis: Twowards distributed reconstruction.

## Project structure
The solution is split into three projects:

* Core
* DistributedReconstruction
* SingleReconstruction


The core project contains the implementation of the gridder and the deconvolution algorithms. All, except for the distributed coordinate descent algorithm, which is located in the DistributedReconstruction project. 

The DistributedReconstruction project contains all code which uses MPI for distributed processing. 

The SingleReconstruction project does not use MPI. It uses the gridder and deconvolution algorithms from the Core project. The SingleReconstruction project itself contains the code for running the algorithms on a single machine, and all the code which runs the experiments of this P9.

## Getting the data

## Windows installation

Simply open the project with visual studio, it should download all the necessaary dependencies.



## ix installation
On Linux/unix machines we need to build dependencies and add them to the C\# build. First, build the project (preferrably in a self-contained build), and then copy the dependencies into the build folder (for example: ~/Distibuted-Reconstruction/SingleReconstruction/bin/publish/).

 The project has two native code dependencies:

* FFTW
* MPI



FFTW: build the library with the following command:

fftw3 ./configure --prefix=/home/jon/fftw --enable-threads --with-combined-threads --enable-shared

Rename the library to "libfftw3-3-x64.so" and copy it to the build folder.

MPI: Install MPICH on your system via apt/yum, copy "libmpich.so.0.0" to the build folder and rename it to "msmpi.dll". This is only necessaary for running the DisrtibutedReconstruction project.



