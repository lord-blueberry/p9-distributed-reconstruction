using System;
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
        const float LAMBDA = 1.0f;
        const float ALPHA = 0.01f;

        private static void ReconstructMinorCycle(Data input, int cutFactor, float[,] fullPsf, string folder, string file, int minorCycles, float searchPercent, bool useAccelerated = true, int blockSize = 1, int maxCycle = 6)
        {
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
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
                var xImage = new float[input.c.GridSize, input.c.GridSize];
                var residualVis = input.visibilities;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    Console.WriteLine("cycle " + cycle);
                    var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
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
                    var modelVis = IDG.DeGridW(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = Visibilities.Substract(input.visibilities, modelVis, input.flags);
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

        private static void RunProcessor(Data input,float[,] fullPsf, string outFolder)
        {
            var cpus = new int[] { 4, 8, 16, 32 };
            foreach (var cpu in cpus)
            {
                var file = "cpu" + cpu;
                var currentFolder = Path.Combine(outFolder, "CPU");
                Directory.CreateDirectory(currentFolder);
                Reconstruct(input, cpu, fullPsf, currentFolder, file, 8, 1, true, 0f, 0.1f);
            }
        }

        private static void RunPsfSize(Data input, float[,] fullPsf, string outFolder)
        {
            var psfTest = new int[] {4 , 8, 16, 32, 64 };
            foreach (var psfSize in psfTest)
            {
                var file = "PsfSize" + psfSize;
                var currentFolder = Path.Combine(outFolder, "PsfSize");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, psfSize, fullPsf, currentFolder, file, 3, 0.1f);
            }
        }

        private static void RunBlocksize(Data input, float[,] fullPsf, string outFolder)
        {
            var blockTest = new int[] { 2, 4, 8 };
            foreach (var block in blockTest)
            {
                var file = "block" + block;
                var currentFolder = Path.Combine(outFolder, "BlockSize");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, 32, fullPsf, currentFolder, file, 3, 0.1f, true, block);
            }
        }

        private static void RunSearchPercent(Data input, float[,] fullPsf, string outFolder)
        {
            var searchPercent = new float[] { 0.0f, /*0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f,*/ 1.0f };
            foreach (var percent in searchPercent)
            {
                var file = "SearchPercent" + percent;
                var currentFolder = Path.Combine(outFolder, "Search");
                Directory.CreateDirectory(currentFolder);
                ReconstructMinorCycle(input, 32, fullPsf, currentFolder, file, 3, percent);
            }
        }

        private static void RunNotAccelerated(Data input, float[,] fullPsf, string outFolder)
        {

            var file = "TestNotAccelerated";
            var currentFolder = Path.Combine(outFolder, "NotAccelerated");
            Directory.CreateDirectory(currentFolder);
            ReconstructMinorCycle(input, 32, fullPsf, currentFolder, file, 3, 0.1f, false);
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
            double wStep = maxW / (wLayerCount);

            data.c = new GriddingConstants(data.visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
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

            var outFolder = "ApproxExperiment";
            Directory.CreateDirectory(outFolder);
            FitsIO.Write(psf, Path.Combine(outFolder, "psfFull.fits"));

            //tryout with simply cutting the PSF
            RunPsfSize(data, psf, outFolder);
            //RunBlocksize(data, psf, outFolder);
            //RunSearchPercent(data, psf, outFolder);
            RunNotAccelerated(data, psf, outFolder);
        }
    }
}
