﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Numerics;
using static System.Math;

using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns.Experiments
{
    class ApproxParameters
    {
        static float LAMBDA = 1.0f;
        //static float LAMBDA = 0.8f;
        static float ALPHA = 0.01f;

        private static void ReconstructMinorCycle(Data input, int cutFactor, float[,] fullPsf, string folder, string file, int minorCycles, float searchPercent, float lambdaInput)
        {
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var sidelobeHalf = PSF.CalcMaxSidelobe(fullPsf, 2);
            var random = new Random(123);
            var approx = new ApproxFast(totalSize, psfCut, 8, 1, 0.1f, searchPercent, false, true);


            using(var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1))))
            using (var bMapCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1))))
            using (var residualsConvolver = new PaddedConvolver(totalSize, fullPsf))
            {
                var currentBMapCalculator = bMapCalculator;

                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambda = (float)(lambdaInput * maxLipschitz);
                var lambdaTrue = (float)(lambdaInput * PSF.CalcMaxLipschitz(fullPsf));
                var alpha = ALPHA;
                ApproxFast.LAMBDA_TEST = lambdaTrue;
                ApproxFast.ALPHA_TEST = alpha;

                var switchedToOtherPsf = false;
                var writer = new StreamWriter(folder + "/" + file + "_lambda.txt");
                var data = new ApproxFast.TestingData(new StreamWriter(folder + "/" + file + ".txt"));
                var xImage = new float[input.c.GridSize, input.c.GridSize];
                var residualVis = input.visibilities;
                for (int cycle = 0; cycle < 6; cycle++)
                {
                    Console.WriteLine("cycle " + cycle);
                    var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, folder + "/dirty" + cycle + ".fits");

                    var minLambda = 0.0f;
                    var dirtyCopy = Copy(dirtyImage);
                    var xCopy = Copy(xImage);
                    //var residualsConvolver = new PaddedConvolver(PSF.CalcPaddedFourierConvolution(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1)));
                    for (int minorCycle = 0; minorCycle < minorCycles; minorCycle++)
                    {
                        FitsIO.Write(dirtyImage, folder + "/dirtyMinor_" + minorCycle + ".fits");
                        var maxDirty = Residuals.GetMax(dirtyImage);
                        var bMap = currentBMapCalculator.Convolve(dirtyImage);
                        var maxB = Residuals.GetMax(bMap);
                        var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitz), 1.0f);
                        var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                        var currentLambda = (float)Math.Max(currentSideLobe / alpha, lambda);

                        if (minorCycle == 0)
                            minLambda = (float)(maxB * sidelobeHalf * correctionFactor / alpha);

                        if (currentLambda < minLambda)
                            currentLambda = minLambda;

                        writer.WriteLine(cycle + ";" + minorCycle + ";" + currentLambda + ";" + minLambda);
                        writer.Flush();
                        approx.DeconvolveTest(data, cycle, minorCycle, xImage, dirtyImage, psfCut, fullPsf, currentLambda, alpha, random, 15, 1e-5f);
                        FitsIO.Write(xImage, folder + "/xImageMinor_" + minorCycle + ".fits");

                        if (currentLambda == lambda & !switchedToOtherPsf)
                        {
                            approx.ResetAMap(fullPsf);
                            currentBMapCalculator = bMapCalculator2;
                            lambda = lambdaTrue;
                            switchedToOtherPsf = true;
                            writer.WriteLine("switched");
                            writer.Flush();
                        }

                        if (currentLambda == lambda | currentLambda == minLambda)
                            break;

                        Console.WriteLine("resetting residuals!!");
                        //reset dirtyImage with full PSF
                        var residualsUpdate = new float[xImage.GetLength(0), xImage.GetLength(1)];
                        Parallel.For(0, xCopy.GetLength(0), (i) =>
                        {
                            for (int j = 0; j < xCopy.GetLength(1); j++)
                                residualsUpdate[i, j] = xImage[i, j] - xCopy[i, j];
                        });
                        residualsConvolver.ConvolveInPlace(residualsUpdate);

                        Parallel.For(0, xCopy.GetLength(0), (i) =>
                        {
                            for (int j = 0; j < xCopy.GetLength(1); j++)
                                dirtyImage[i, j] = dirtyCopy[i, j] - residualsUpdate[i, j];
                            
                        });
                    }
                    FitsIO.Write(xImage, folder + "/xImage_" + cycle + ".fits");

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                }

                writer.Close();
            }
        }

        private static void Reconstruct(Data input, int cutFactor, float[,] fullPsf, string folder, string file, int threads, int blockSize, bool accelerated, float randomPercent, float searchPercent, float lambdaInput)
        {
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var random = new Random(123);
            var approx = new ApproxFast(totalSize, psfCut, threads, blockSize, randomPercent, searchPercent, false, true);

            var maxLipschitzCut = PSF.CalcMaxLipschitz(psfCut);
            var lambda = (float)(lambdaInput * PSF.CalcMaxLipschitz(psfCut));
            var lambdaTrue = (float)(lambdaInput * PSF.CalcMaxLipschitz(fullPsf));
            var alpha = ALPHA;
            ApproxFast.LAMBDA_TEST = lambdaTrue;
            ApproxFast.ALPHA_TEST = alpha;

            var switchedToOtherPsf = false;
            var writer = new StreamWriter(folder + "/" + file + "_lambda.txt");
            var data = new ApproxFast.TestingData(new StreamWriter(folder+ "/" + file + ".txt"));
            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            for (int cycle = 0; cycle < 6; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, folder + "/dirty" + cycle + ".fits");

                var maxDirty = Residuals.GetMax(dirtyImage);
                var bMap = bMapCalculator.Convolve(dirtyImage);
                var maxB = Residuals.GetMax(bMap);
                var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitzCut), 1.0f);
                var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                var currentLambda = (float)Math.Max(currentSideLobe / alpha, lambda);

                writer.WriteLine("cycle" + ";" + currentLambda);
                writer.Flush();

                approx.DeconvolveTest(data, cycle, 0, xImage, dirtyImage, psfCut, fullPsf, currentLambda, alpha, random, 15, 1e-5f);
                FitsIO.Write(xImage, folder + "/xImage_" + cycle + ".fits");

                if(currentLambda == lambda & !switchedToOtherPsf)
                {
                    approx.ResetAMap(fullPsf);
                    lambda = lambdaTrue;
                    switchedToOtherPsf = true;
                    writer.WriteLine("switched");
                    writer.Flush();
                }

                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGridW(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
            }
            writer.Close();

        }


        public static void Run()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var data = LMC.Load(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.uvw.GetLength(0); i++)
                for (int j = 0; j < data.uvw.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.uvw[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.frequencies[data.frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < data.flags.GetLength(0); i++)
                for (int j = 0; j < data.flags.GetLength(1); j++)
                    for (int k = 0; k < data.flags.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            visCount2++;

            data.c = new GriddingConstants(data.visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, maxW);
            data.metadata = Partitioner.CreatePartition(data.c, data.uvw, data.frequencies);

            var psfVis = new Complex[data.uvw.GetLength(0), data.uvw.GetLength(1), data.frequencies.Length];
            for (int i = 0; i < data.visibilities.GetLength(0); i++)
                for (int j = 0; j < data.visibilities.GetLength(1); j++)
                    for (int k = 0; k < data.visibilities.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            Console.WriteLine("gridding psf");
            var psfGrid = IDG.GridW(data.c, data.metadata, psfVis, data.uvw, data.frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, data.c.VisibilitiesCount);
            FFT.Shift(psf);
            

            Directory.CreateDirectory("ApproxTest/cpu");
            FitsIO.Write(psf, "psfFull.fits");

            //tryout with simply cutting the PSF
            var outFolder = "ApproxTest/";

            /*
            var cpuTest = new int[] { 8 };
            var blockTest = new int[] { 1, 4, 8, 16};
            foreach(var cpu in cpuTest)
            {
                foreach(var block in blockTest)
                {
                    var file = "Grid_cpu"+ cpu + "block" + block;
                    var currentFolder = outFolder + file;
                    Directory.CreateDirectory(currentFolder);
                    Reconstruct(data, 16, psf, currentFolder, file, cpu, block, true, 0f, 0.25f);
                }
            }*/

            var psfSizes = new int[] {32};
            foreach (var size in psfSizes)
            {
                var file = "Grid_cpu" + 8 + "block" + 1 + "psf" + size;
                var currentFolder = outFolder + file;
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(data, size, psf, currentFolder, file, 3, 0.1f, LAMBDA);
            }

            var searchPercent = new float[] {/*0.01f, 0.05f,*/ 0.1f, /*0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f*/};
            foreach (var search in searchPercent)
            {
                var file = "Grid_cpu" + 8 + "block" + 1+"search"+search;
                var currentFolder = outFolder + file;
                Directory.CreateDirectory(currentFolder);
                //Reconstruct(data, 32, psf, currentFolder, file, 8, 1, true, 0f, search, LAMBDA);
            }

            /*var lambdas = new float[] {0.8f, 0.6f, 0.4f};
            foreach (var lambda in lambdas)
            {
                var file = "Grid_cpu" + 8 + "block" + 1 + "lambda" + lambda;
                var currentFolder = outFolder + file;
                Directory.CreateDirectory(currentFolder);
                Reconstruct(data, 16, psf, currentFolder, file, 8, 1, true, 0f, 0.1f, lambda);
            }*/


        }

        public static void ActiveSetDebug()
        {
            
            var psf = FitsIO.ReadImage("ApproxTest/psfFull.fits");
            var dirty = FitsIO.ReadImage("ApproxTest/dirty7.fits");
            var xImage = FitsIO.ReadImage("ApproxTest/xImage_7.fits");
            var psfCut = PSF.Cut(psf, 8);
            var lambda = 130.84416f;

            var totalSize = new Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));
            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psfCut, new Rectangle(0, 0, dirty.GetLength(0), dirty.GetLength(1)));
            var gExplore = Residuals.CalcGradientMap(dirty, PSFCorr, new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            FitsIO.Write(gExplore, "bMapDebug.fits");
            ApproxFast.GetActiveSet(xImage, gExplore, 8, 8, lambda, ALPHA, PSF.CalcAMap(psf, totalSize));
        }
    }
}
