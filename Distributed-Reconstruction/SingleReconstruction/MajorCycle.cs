using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Numerics;

using Core;
using static Core.Common;
using Core.Deconvolution;
using Core.Deconvolution.ParallelDeconvolution;
using Core.ImageDomainGridder;

namespace SingleReconstruction
{
    class MajorCycle
    {
        const float MAJOR_EPSILON = 1e-4f;

        /// <summary>
        /// Major cycle implementation for the Serial CD
        /// </summary>
        /// <param name="data"></param>
        /// <param name="c"></param>
        /// <param name="useGPU"></param>
        /// <param name="psfCutFactor"></param>
        /// <param name="maxMajorCycle"></param>
        /// <param name="lambda"></param>
        /// <param name="alpha"></param>
        /// <param name="deconvIterations"></param>
        /// <param name="deconvEpsilon"></param>
        public static void ReconstructSerialCD(MeasurementData data, GriddingConstants c, bool useGPU, int psfCutFactor, int maxMajorCycle, float lambda, float alpha, int deconvIterations, float deconvEpsilon)
        {
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

            var totalWatch = new Stopwatch();
            var currentWatch = new Stopwatch();

            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(psf, psfCutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, psfCutFactor);

            IDeconvolver deconvolver = null;
            if (useGPU & GPUSerialCD.IsGPUSupported())
            {
                deconvolver = new GPUSerialCD(totalSize, psfCut, 1000);
            }
            else if(useGPU & !GPUSerialCD.IsGPUSupported())
            {
                Console.WriteLine("GPU not supported by library. Switching to CPU implementation");
                deconvolver = new FastSerialCD(totalSize, psfCut);
            }
            else
            {
                deconvolver = new FastSerialCD(totalSize, psfCut);
            }

            var psfBMap = psfCut;
            using (var gCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1))))
            using (var gCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1))))
            {
                var currentGCalculator = gCalculator;
                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambdaLipschitz = (float)(lambda * maxLipschitz);
                var lambdaTrue = (float)(lambda * PSF.CalcMaxLipschitz(psf));
                var switchedToOtherPsf = false;

                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = data.Visibilities;
                DeconvolutionResult lastResult = null;
                for (int cycle = 0; cycle < maxMajorCycle; cycle++)
                {
                    Console.WriteLine("Beginning Major cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, data.UVW, data.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, "dirty_serial_majorCycle" + cycle + ".fits");

                    currentWatch.Restart();
                    totalWatch.Start();
                    var maxDirty = Residuals.GetMax(dirtyImage);
                    var gradients = gCalculator.Convolve(dirtyImage);
                    var maxB = Residuals.GetMax(gradients);
                    var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitz), 1.0f);
                    var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                    var currentLambda = (float)Math.Max(currentSideLobe / alpha, lambdaLipschitz);

                    var objective = Residuals.CalcPenalty(dirtyImage) + ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);

                    var absMax = deconvolver.GetAbsMaxDiff(xImage, gradients, lambdaTrue, alpha);

                    if (absMax >= MAJOR_EPSILON)
                        lastResult = deconvolver.Deconvolve(xImage, gradients, currentLambda, alpha, deconvIterations, deconvEpsilon);

                    if (lambda == currentLambda & !switchedToOtherPsf)
                    {
                        currentGCalculator = gCalculator2;
                        lambda = lambdaTrue;
                        maxLipschitz = PSF.CalcMaxLipschitz(psf);
                        switchedToOtherPsf = true;
                    }

                    FitsIO.Write(xImage, "model_serial_majorCycle" + cycle + ".fits");

                    currentWatch.Stop();
                    totalWatch.Stop();

                    if (absMax < MAJOR_EPSILON)
                        break;

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(c, metadata, xGrid, data.UVW, data.Frequencies);
                    residualVis = Visibilities.Substract(data.Visibilities, modelVis, data.Flags);
                }

                Console.WriteLine("Reconstruction finished in (seconds): " + totalWatch.Elapsed.TotalSeconds);
            }
        }

        /// <summary>
        /// Major cycle implemnentation for the parallel coordinate descent algorithm
        /// </summary>
        /// <param name="data"></param>
        /// <param name="c"></param>
        /// <param name="psfCutFactor"></param>
        /// <param name="maxMajorCycle"></param>
        /// <param name="maxMinorCycle"></param>
        /// <param name="lambda"></param>
        /// <param name="alpha"></param>
        /// <param name="deconvIterations"></param>
        /// <param name="deconvEpsilon"></param>
        public static void ReconstructPCDM(MeasurementData data, GriddingConstants c, int psfCutFactor, int maxMajorCycle, int maxMinorCycle, float lambda, float alpha, int deconvIterations, float deconvEpsilon)
        {
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
            var psfGrid = IDG.Grid(c, metadata, psfVis, data.UVW, data.Frequencies);
            var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            var totalWatch = new Stopwatch();
            var currentWatch = new Stopwatch();

            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(psf, psfCutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, psfCutFactor);
            var sidelobeHalf = PSF.CalcMaxSidelobe(psf, 2);

            var pcdm = new ParallelCoordinateDescent(totalSize, psfCut, Environment.ProcessorCount, 1000);

            using (var gCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1))))
            using (var gCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1))))
            using (var residualsConvolver = new PaddedConvolver(totalSize, psf))
            {
                var currentGCalculator = gCalculator;

                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambdaLipschitz = (float)(lambda * maxLipschitz);
                var lambdaTrue = (float)(lambda * PSF.CalcMaxLipschitz(psf));

                var switchedToOtherPsf = false;
                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = data.Visibilities;
                ParallelCoordinateDescent.PCDMStatistics lastResult = null;
                for (int cycle = 0; cycle < maxMajorCycle; cycle++)
                {
                    Console.WriteLine("Beginning Major cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, data.UVW, data.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, "dirty_pcdm_majorCycle" + cycle + ".fits");

                    currentWatch.Restart();
                    totalWatch.Start();

                    var breakMajor = false;
                    var minLambda = 0.0f;
                    var dirtyCopy = Copy(dirtyImage);
                    var xCopy = Copy(xImage);
                    var currentLambda = 0f;
                    var currentObjective = 0.0;
                    var absMax = 0.0f;
                    for (int minorCycle = 0; minorCycle < maxMinorCycle; minorCycle++)
                    {
                        Console.WriteLine("Beginning Minor Cycle " + minorCycle);
                        var maxDirty = Residuals.GetMax(dirtyImage);
                        var bMap = currentGCalculator.Convolve(dirtyImage);
                        var maxB = Residuals.GetMax(bMap);
                        var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitz), 1.0f);
                        var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                        currentLambda = (float)Math.Max(currentSideLobe / alpha, lambdaLipschitz);

                        if (minorCycle == 0)
                            minLambda = (float)(maxB * sidelobeHalf * correctionFactor / alpha);
                        if (currentLambda < minLambda)
                            currentLambda = minLambda;
                        currentObjective = Residuals.CalcPenalty(dirtyImage) + ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                        absMax = pcdm.GetAbsMax(xImage, bMap, lambdaTrue, alpha);
                        if (absMax < MAJOR_EPSILON)
                        {
                            breakMajor = true;
                            break;
                        }

                        lastResult = pcdm.Deconvolve(xImage, bMap, currentLambda, alpha, 40, deconvEpsilon);

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

                    if (breakMajor)
                        break;
                    if (currentLambda == lambda & !switchedToOtherPsf)
                    {
                        pcdm.ResetAMap(psf);
                        currentGCalculator = gCalculator2;
                        lambda = lambdaTrue;
                        switchedToOtherPsf = true;
                    }

                    FitsIO.Write(xImage, "model_pcdm_majorCycle" + cycle + ".fits");

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(c, metadata, xGrid, data.UVW, data.Frequencies);
                    residualVis = Visibilities.Substract(data.Visibilities, modelVis, data.Flags);
                }

                Console.WriteLine("Reconstruction finished in (seconds): " + totalWatch.Elapsed.TotalSeconds);
            }
        }

    }
}
