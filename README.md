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
* Install Visual Studio (Community). Install with .Net Core Cross Platform development. If a visual studio is already installed, make sure it is up-to-date.
* Install FitsViewer, for example [DS9](http://ds9.si.edu/site/Download.html) (Images are written as FITS files)
* Git clone, or download the project to your system https://github.com/lord-blueberry/p9-distributed-reconstruction/releases/tag/v1.0.0
* Download the p9-data folder to your system. 

Run the project within Visual Studio:
* Open the solution file **project folder**/DistributedReconstruction/DisrtibutedReconstruction.sln with Visual Studio
* Select SingleReconstruction project to run
* Open the file: SingleReconstruction/RunningMethods.cs
* Change the constant variable P9_DATA_FOLDER and put in the full path of the **p9-data** folder
* Press F5 to run in Visual Studio

The file SingleReconstruction/RunningMethods.cs executes the two reconstruction algorithms (serial and parallel coordinate desecnent) on a simulated observation, and on the LMC observation. 
The output is written in the folder SingleReconstruction/bin/(debug or release)/

### Run on simulated dataset

The simulated observation consists of two point sources in a small image (256 \* 256 pixels) If a NVIDIA GPU is available, then the serial coordinate descent algorithm will use GPU acceleration. The serial coordinate descent algorithm is faster on this dataset than the parallel algorithm. The parallel algorithm is configured to not approximate the PSF (psfCutFactor = 1) and cannot efficiently use multiple processors on this dataset.

### Run on the LMC Dataset

Warning: 16 GB of RAM are needed to reconstruct this dataset. You should run this reconstruction in release mode. Calculating the PSF and the dirty image may take several minutes.

The serial and parallel coordinate descent algorithm use the same configuration as in the comparison in Section 7.4 of the documentation. 


## Project structure
The solution is split into three projects:

* Core
* DistributedReconstruction
* SingleReconstruction

The core project contains the implementation of the gridder and the deconvolution algorithms. All, except for the distributed coordinate descent algorithm, which is located in the DistributedReconstruction project. 

The DistributedReconstruction project contains all code which uses MPI for distributed processing. 

The SingleReconstruction project does not use MPI. It uses the gridder and deconvolution algorithms from the Core project. The SingleReconstruction project itself contains the code for running the algorithms on a single machine, and all the code which runs the experiments of this P9.

## Linux installation of a reconstruction pipeline
For Linux/Unix machines, create a self-contained .Net Core deployment for the Linux platform. Now we need to replace the native code dependencies of fftw and MPI. 


FFTW: build the library with the following command:

```bash
fftw3 ./configure --prefix=/home/jon/fftw --enable-threads --with-combined-threads --enable-shared
```

Rename the library to "libfftw3-3-x64.so" and copy it to the build folder.

MPI: Install MPICH on your system via apt/yum, copy "libmpich.so.0.0" to the build folder and rename it to "msmpi.dll". This is only necessaary for running the DisrtibutedReconstruction project.



