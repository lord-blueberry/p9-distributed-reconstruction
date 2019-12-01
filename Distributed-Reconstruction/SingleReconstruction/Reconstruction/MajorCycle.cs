using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Numerics;

using Core;
using static Core.Common;
using Core.Deconvolution;
using Core.ImageDomainGridder;

namespace SingleReconstruction.Reconstruction
{
    class MajorCycle
    {
        const float MAJOR_EPSILON = 1e-4f;

        private static void Reconstruct(MeasurementData data, GriddingConstants c, IDeconvolver deconvolver, int psfCutFactor, int maxMajorCycle, float lambda, float alpha, int deconvIterations, float deconvEpsilon)
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
            

            var psfBMap = psfCut;
            using (var gCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1))))
            using (var gCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1))))
            {
                var currentBMapCalculator = gCalculator;
                var maxLipschitz = PSF.CalcMaxLipschitz(psfCut);
                var lambdaLipschitz = (float)(lambda * maxLipschitz);
                var lambdaTrue = (float)(lambda * PSF.CalcMaxLipschitz(psf));
                var switchedToOtherPsf = false;

                var xImage = new float[c.GridSize, c.GridSize];
                var residualVis = data.Visibilities;
                DeconvolutionResult lastResult = null;
                for (int cycle = 0; cycle < maxMajorCycle; cycle++)
                {
                    Console.WriteLine("Major Cycle " + cycle);
                    var dirtyGrid = IDG.GridW(c, metadata, residualVis, data.UVW, data.Frequencies);
                    var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                    FFT.Shift(dirtyImage);
                    FitsIO.Write(dirtyImage, "/dirty" + cycle + ".fits");

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
                        currentBMapCalculator = gCalculator2;
                        lambda = lambdaTrue;
                        maxLipschitz = PSF.CalcMaxLipschitz(psf);
                        switchedToOtherPsf = true;
                    }

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
            }

        }
    }
}
