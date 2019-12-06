using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Numerics;

using Core;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using static Core.Common;
using Core.Deconvolution.ParallelDeconvolution;

namespace SingleReconstruction.Experiments
{
    class PCDMComparison
    {
        public const int CUT_FACTOR_SERIAL = 16;
        public const int CUT_FACTOR_PCDM = 32;
        const float LAMBDA = 1.0f;
        const float ALPHA = 0.01f;
        const float MAJOR_STOP = 1e-4f;

        private static void ReconstructPCDM(MeasurementData input, GriddingConstants c, float[,] fullPsf, string folder, string file, int minorCycles, float searchPercent, int processorCount)
        {
            var totalWatch = new Stopwatch();
            var currentWatch = new Stopwatch();

            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(fullPsf, CUT_FACTOR_PCDM);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, CUT_FACTOR_PCDM);
            var sidelobeHalf = PSF.CalcMaxSidelobe(fullPsf, 2);
            var random = new Random(123);
            var pcdm = new Deconvolver(totalSize, psfCut, processorCount, 1000, searchPercent);

            var metadata = Partitioner.CreatePartition(c, input.UVW, input.Frequencies);

            using (var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1))))
            using (var bMapCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1))))
            using (var residualsConvolver = new PaddedConvolver(totalSize, fullPsf))
            {
                var currentBMapCalculator = bMapCalculator;

                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambda = (float)(LAMBDA * maxLipschitz);
                var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(fullPsf));
                var alpha = ALPHA;

                var switchedToOtherPsf = false;
                var writer = new StreamWriter(folder + "/" + file + ".txt");
                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = input.Visibilities;
                Deconvolver.DeconvolutionResult lastResult = null;
                for (int cycle = 0; cycle < 6; cycle++)
                {
                    Console.WriteLine("Beginning Major cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, input.UVW, input.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, folder + "/dirty" + cycle + ".fits");

                    currentWatch.Restart();
                    totalWatch.Start();

                    var breakMajor = false;
                    var minLambda = 0.0f;
                    var dirtyCopy = Copy(dirtyImage);
                    var xCopy = Copy(xImage);
                    var currentLambda = 0f;
                    var currentObjective = 0.0;
                    var absMax = 0.0f;
                    for (int minorCycle = 0; minorCycle < minorCycles; minorCycle++)
                    {
                        Console.WriteLine("Beginning Minor Cycle "+ minorCycle);
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
                        currentObjective = Residuals.CalcPenalty(dirtyImage) + ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                        absMax = pcdm.GetAbsMax(xImage, bMap, lambdaTrue, alpha);
                        if (absMax < MAJOR_STOP)
                        {
                            breakMajor = true;
                            break;
                        }
                            
                        lastResult = pcdm.DeconvolvePCDM(xImage, bMap, currentLambda, alpha, 40, 1e-5f);

                        if (currentLambda == lambda | currentLambda == minLambda)
                            break;

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

                    currentWatch.Stop();
                    totalWatch.Stop();
                    writer.WriteLine(cycle + ";" + currentLambda + ";" + currentObjective + ";" + absMax + ";" + lastResult.IterationCount + ";"  + totalWatch.Elapsed.TotalSeconds + ";" + currentWatch.Elapsed.TotalSeconds);
                    writer.Flush();

                    FitsIO.Write(xImage, folder + "/xImage_pcdm_" + cycle + ".fits");

                    if (breakMajor)
                        break;
                    if (currentLambda == lambda & !switchedToOtherPsf)
                    {
                        pcdm.ResetAMap(fullPsf);
                        currentBMapCalculator = bMapCalculator2;
                        lambda = lambdaTrue;
                        switchedToOtherPsf = true;
                        writer.WriteLine("switched");
                        writer.Flush();
                    }

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(c, metadata, xGrid, input.UVW, input.Frequencies);
                    residualVis = Visibilities.Substract(input.Visibilities, modelVis, input.Flags);
                }

                writer.Close();
            }
        }

        private static void ReconstructSerial(MeasurementData input, GriddingConstants c, float[,] fullPsf, string folder, string file, int processorCount)
        {
            var totalWatch = new Stopwatch();
            var currentWatch = new Stopwatch();

            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(fullPsf, CUT_FACTOR_SERIAL);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, CUT_FACTOR_SERIAL);
            var fastCD = new FastGreedyCD(totalSize, psfCut, processorCount);
            var metadata = Partitioner.CreatePartition(c, input.UVW, input.Frequencies);

            var writer = new StreamWriter(folder + "/" + file + ".txt");
            var psfBMap = psfCut;
            using(var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1))))
            using(var bMapCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1))))
            {
                var currentBMapCalculator = bMapCalculator;

                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambda = (float)(LAMBDA * maxLipschitz);
                var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(fullPsf));
                var alpha = ALPHA;

                var switchedToOtherPsf = false;
                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = input.Visibilities;
                DeconvolutionResult lastResult = null;
                for (int cycle = 0; cycle < 6; cycle++)
                {
                    Console.WriteLine("cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, input.UVW, input.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, folder + "/dirty" + cycle + ".fits");

                    currentWatch.Restart();
                    totalWatch.Start();
                    var maxDirty = Residuals.GetMax(dirtyImage);
                    var bMap = bMapCalculator.Convolve(dirtyImage);
                    var maxB = Residuals.GetMax(bMap);
                    var correctionFactor = Math.Max(maxB / (maxDirty * fastCD.MaxLipschitz), 1.0f);
                    var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                    var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                    
                    var objective = Residuals.CalcPenalty(dirtyImage) + ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                    
                    var absMax = fastCD.GetAbsMaxDiff(xImage, bMap, lambdaTrue, alpha);
                    
                    if (absMax >= MAJOR_STOP)
                        lastResult = fastCD.Deconvolve(xImage, bMap, currentLambda, alpha, 30000, 1e-5f);

                    if (lambda == currentLambda & !switchedToOtherPsf)
                    {
                        currentBMapCalculator = bMapCalculator2;
                        lambda = lambdaTrue;
                        switchedToOtherPsf = true;
                    }

                    currentWatch.Stop();
                    totalWatch.Stop();
                    writer.WriteLine(cycle + ";" + currentLambda + ";" + objective + ";" + absMax + ";" + lastResult.IterationCount + ";" + totalWatch.Elapsed.TotalSeconds + ";" + currentWatch.Elapsed.TotalSeconds);
                    writer.Flush();

                    if (absMax < MAJOR_STOP)
                        break;

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(c, metadata, xGrid, input.UVW, input.Frequencies);
                    residualVis = Visibilities.Substract(input.Visibilities, modelVis, input.Flags);
                }

            }
            
        }

        public static void Run()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var data = MeasurementData.LoadLMC(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * Math.PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.UVW.GetLength(0); i++)
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.UVW[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.Frequencies[data.Frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < data.Flags.GetLength(0); i++)
                for (int j = 0; j < data.Flags.GetLength(1); j++)
                    for (int k = 0; k < data.Flags.GetLength(2); k++)
                        if (!data.Flags[i, j, k])
                            visCount2++;
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

            Directory.CreateDirectory("PCDMComparison");
            ReconstructPCDM(data, c, psf, "PCDMComparison", "pcdm", 3, 0.1f, 8);
            ReconstructSerial(data, c, psf, "PCDMComparison", "serial", -1);
        }


        public static void RunProcessorComparison()
        {
            var folder = "/home/jonass/meerkat_tiny/";

            var data = MeasurementData.LoadLMC(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * Math.PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.UVW.GetLength(0); i++)
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.UVW[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.Frequencies[data.Frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < data.Flags.GetLength(0); i++)
                for (int j = 0; j < data.Flags.GetLength(1); j++)
                    for (int k = 0; k < data.Flags.GetLength(2); k++)
                        if (!data.Flags[i, j, k])
                            visCount2++;
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

            Directory.CreateDirectory("PCDMComparison/processors");
            var processorCount = new int[] { 1, 4, 8, 16, 32 };
            foreach(var count in processorCount)
            {
                
                ReconstructPCDM(data, c, psf, "PCDMComparison/processors", "pcdm"+count, 3, 0.1f, count);
                ReconstructSerial(data, c, psf, "PCDMComparison/processors", "serial"+count, count);

            }

        }
    }
}
