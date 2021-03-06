﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Numerics;
using static System.Math;

using Core;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using static Core.Common;
using static SingleReconstruction.Experiments.DataLoading;

namespace SingleReconstruction.Experiments
{
    class ApproxParameters
    {
        const float LAMBDA = 1.0f;
        const float ALPHA = 0.01f;

        private static void ReconstructMinorCycle(MeasurementData input, GriddingConstants c, int cutFactor, float[,] fullPsf, string folder, string file, int minorCycles, float searchPercent, bool useAccelerated = true, int blockSize = 1, int maxCycle = 6)
        {
            var metadata = Partitioner.CreatePartition(c, input.UVW, input.Frequencies);

            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var sidelobeHalf = PSF.CalcMaxSidelobe(fullPsf, 2);
            var random = new Random(123);
            var approx = new ApproxFast(totalSize, psfCut, 8, blockSize, 0.1f, searchPercent, false, useAccelerated);

            using(var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1))))
            using (var bMapCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1))))
            using (var residualsConvolver = new PaddedConvolver(totalSize, fullPsf))
            {
                var currentBMapCalculator = bMapCalculator;

                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambda = (float)(LAMBDA * maxLipschitz);
                var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(fullPsf));
                var alpha = ALPHA;
                ApproxFast.LAMBDA_TEST = lambdaTrue;
                ApproxFast.ALPHA_TEST = alpha;

                var switchedToOtherPsf = false;
                var writer = new StreamWriter(folder + "/" + file + "_lambda.txt");
                var data = new ApproxFast.TestingData(new StreamWriter(folder + "/" + file + ".txt"));
                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = input.Visibilities;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    Console.WriteLine("cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, input.UVW, input.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, folder + "/dirty" + cycle + ".fits");

                    var minLambda = 0.0f;
                    var dirtyCopy = Copy(dirtyImage);
                    var xCopy = Copy(xImage);
                    var currentLambda = 0f;
                    //var residualsConvolver = new PaddedConvolver(PSF.CalcPaddedFourierConvolution(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1)));
                    for (int minorCycle = 0; minorCycle < minorCycles; minorCycle++)
                    {
                        FitsIO.Write(dirtyImage, folder + "/dirtyMinor_" + minorCycle + ".fits");
                        var maxDirty = Residuals.GetMax(dirtyImage);
                        var bMap = currentBMapCalculator.Convolve(dirtyImage);
                        var maxB = Residuals.GetMax(bMap);
                        var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitz), 1.0f);
                        var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                        currentLambda = (float)Math.Max(currentSideLobe / alpha, lambda);

                        if (minorCycle == 0)
                            minLambda = (float)(maxB * sidelobeHalf * correctionFactor / alpha);

                        if (currentLambda < minLambda)
                            currentLambda = minLambda;

                        writer.WriteLine(cycle + ";" + minorCycle + ";" + currentLambda + ";" + minLambda);
                        writer.Flush();
                        approx.DeconvolveTest(data, cycle, minorCycle, xImage, dirtyImage, psfCut, fullPsf, currentLambda, alpha, random, 15, 1e-5f);
                        FitsIO.Write(xImage, folder + "/xImageMinor_" + minorCycle + ".fits");

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

                    if (currentLambda == lambda & !switchedToOtherPsf)
                    {
                        approx.ResetAMap(fullPsf);
                        currentBMapCalculator = bMapCalculator2;
                        lambda = lambdaTrue;
                        switchedToOtherPsf = true;
                        writer.WriteLine("switched");
                        writer.Flush();
                    }

                    FitsIO.Write(xImage, folder + "/xImage_" + cycle + ".fits");

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(c, metadata, xGrid, input.UVW, input.Frequencies);
                    residualVis = Visibilities.Substract(input.Visibilities, modelVis, input.Flags);
                }

                writer.Close();
            }
        }

        private static void Reconstruct(Data input, int cutFactor, float[,] fullPsf, string folder, string file, int threads, int blockSize, bool accelerated, float randomPercent, float searchPercent)
        {
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var random = new Random(123);
            var approx = new ApproxFast(totalSize, psfCut, threads, blockSize, randomPercent, searchPercent, false, true);

            var maxLipschitzCut = PSF.CalcMaxLipschitz(psfCut);
            var lambda = (float)(LAMBDA * PSF.CalcMaxLipschitz(psfCut));
            var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(fullPsf));
            var alpha = ALPHA;
            ApproxFast.LAMBDA_TEST = lambdaTrue;
            ApproxFast.ALPHA_TEST = alpha;

            var switchedToOtherPsf = false;
            var writer = new StreamWriter(folder + "/" + file + "_lambda.txt");
            var data = new ApproxFast.TestingData(new StreamWriter(folder+ "/" + file + ".txt"));
            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            for (int cycle = 0; cycle < 7; cycle++)
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
                residualVis = Visibilities.Substract(input.visibilities, modelVis, input.flags);
            }
            writer.Close();

        }

        private static void RunPsfSize(MeasurementData input, GriddingConstants c, float[,] fullPsf, string outFolder)
        {
            var psfTest = new int[] {4 , 8, 16, 32, 64 };
            foreach (var psfSize in psfTest)
            {
                var file = "PsfSize" + psfSize;
                var currentFolder = Path.Combine(outFolder, "PsfSize");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, c, psfSize, fullPsf, currentFolder, file, 3, 0.1f, false);
            }
        }

        private static void RunBlocksize(MeasurementData input, GriddingConstants c, float[,] fullPsf, string outFolder)
        {
            var blockTest = new int[] { 2, 4, 8 };
            foreach (var block in blockTest)
            {
                var file = "block" + block;
                var currentFolder = Path.Combine(outFolder, "BlockSize");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, c, 32, fullPsf, currentFolder, file, 3, 0.1f, true, block);
            }
        }

        private static void RunSearchPercent(MeasurementData input, GriddingConstants c, float[,] fullPsf, string outFolder)
        {
            var searchPercent = new float[] { 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f };
            foreach (var percent in searchPercent)
            {
                var file = "SearchPercent" + percent;
                var currentFolder = Path.Combine(outFolder, "Search");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, c, 32, fullPsf, currentFolder, file, 3, percent, false);
            }
        }

        private static void RunNotAccelerated(MeasurementData input, GriddingConstants c, float[,] fullPsf, string outFolder)
        {

            var file = "TestNotAccelerated";
            var currentFolder = Path.Combine(outFolder, "NotAccelerated");
            Directory.CreateDirectory(currentFolder);
            ReconstructMinorCycle(input, c, 32, fullPsf, currentFolder, file, 3, 0.1f, false);
        }

        public static void Run()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var data = MeasurementData.LoadLMC(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.UVW.GetLength(0); i++)
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.UVW[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.Frequencies[data.Frequencies.Length - 1]);
            double wStep = maxW / (wLayerCount);

            
            var c = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
            var metadata = Partitioner.CreatePartition(c, data.UVW, data.Frequencies);

            var psfVis = new Complex[data.UVW.GetLength(0), data.UVW.GetLength(1), data.Frequencies.Length];
            for (int i = 0; i < data.Visibilities.GetLength(0); i++)
                for (int j = 0; j < data.Visibilities.GetLength(1); j++)
                    for (int k = 0; k < data.Visibilities.GetLength(2); k++)
                        if (!data.Flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            Console.WriteLine("gridding psf");
            var psfGrid = IDG.GridW(c, metadata, psfVis, data.UVW, data.Frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            var outFolder = "ApproxExperiment";
            Directory.CreateDirectory(outFolder);
            FitsIO.Write(psf, Path.Combine(outFolder, "psfFull.fits"));

            //tryout with simply cutting the PSF
            RunPsfSize(data, c, psf, outFolder);
            //RunBlocksize(data, c, psf, outFolder);
            RunSearchPercent(data, c, psf, outFolder);
            //RunNotAccelerated(data, c, psf, outFolder);
        }
    }
}
