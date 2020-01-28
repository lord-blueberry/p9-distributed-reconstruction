# P9-distributed-reconstruction

This repository holds the code of the Master-Thesis: Twowards distributed reconstruction. There are three other repositories which belong to the master thesis:

* [p9-data](https://github.com/i4Ds/Schwammberger-P9-Data), which contains the measurement data used in this project. *Necessary to run the code*
* [p9-results](https://github.com/lord-blueberry/p9-results), which holds the reconstructed images created with the serial and parallel coordinate descent algorithm. 
* [p9-doc](https://github.com/lord-blueberry/p9-doc), which contains the master thesis and the paper-draft

## Getting the data

The data is saved in a github LFS repository. To clone the repository:

 * Install a git client
 * Install [github lfs](https://git-lfs.github.com/) command line extension (may not be necessary).
 * Checkout the [p9-data](https://github.com/i4Ds/Schwammberger-P9-Data) repository. It is a github lfs repository. It takes a while to download all the files.
 
The repository contains the simulated and the LMC observations.

## Windows setup of the development environment
Install/Download the following:
* Install Visual Studio (Community). Install with .Net Core Cross Platform development
* Install FitsViewer, for example [DS9](http://ds9.si.edu/site/Download.html) (Images are written as FITS files)
* Git clone (or download) project to your system.
* Download the p9-data folder to your system. 

Run the project within Visual Studio:
* Open the solution file **project folder**/DistributedReconstruction/DisrtibutedReconstruction.sln with Visual Studio
* Select SingleReconstruction project to run
* Open the file: SingleReconstruction/RunningMethods.cs
* Change the constant variable P9_DATA_FOLDER and put in the full path of the **p9-data** folder
* Press F5 to run in Visual Studio

The file SingleReconstruction/RunningMethods.cs executes the two reconstruction algorithms (serial and parallel coordinate desecnent) on a simulated observation, and on the LMC observation. The output is written in the folder SingleReconstruction/bin/(debug or release)/




## Project structure
The solution is split into three projects:

* Core
* DistributedReconstruction
* SingleReconstruction

The core project contains the implementation of the gridder and the deconvolution algorithms. All, except for the distributed coordinate descent algorithm, which is located in the DistributedReconstruction project. 

The DistributedReconstruction project contains all code which uses MPI for distributed processing. 

The SingleReconstruction project does not use MPI. It uses the gridder and deconvolution algorithms from the Core project. The SingleReconstruction project itself contains the code for running the algorithms on a single machine, and all the code which runs the experiments of this P9.

## Linix installation of a reconstruction pipeline
On Linux/unix machines we need to build dependencies and add them to the C\# build. First, build the project (preferrably in a self-contained build), and then copy the dependencies into the build folder (for example: ~/Distibuted-Reconstruction/SingleReconstruction/bin/publish/).

This project has two native code dependencies:

* FFTW
* MPI


FFTW: build the library with the following command:

```bash
fftw3 ./configure --prefix=/home/jon/fftw --enable-threads --with-combined-threads --enable-shared
```

Rename the library to "libfftw3-3-x64.so" and copy it to the build folder.

MPI: Install MPICH on your system via apt/yum, copy "libmpich.so.0.0" to the build folder and rename it to "msmpi.dll". This is only necessaary for running the DisrtibutedReconstruction project.



