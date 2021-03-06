﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;
using System.Diagnostics;
using static Core.Common;

namespace Core.Deconvolution
{
    /// <summary>
    /// Fast implementation of the serial coordinate descent algorithm
    /// </summary>
    public class FastSerialCD : IDeconvolver
    {
        /*
         * ImageSize, patchSize and subPatchSize. Only relevant for ISubpatchDeconvolver
         * 
         * imagesize: size of the whole image consisting of several patches.
         * patchSize: a patch inside the image, consisting of several subpatches. bMap.Size == xImage.Size == patchSize
         * subPatchsize: size of a patch inside a patch = subpatch.
         */
        
        readonly Rectangle patch;
        float[,] psf2;
        float[,] aMap;
        ParallelOptions parallelOptions;

        public float MaxLipschitz { get; private set; }

        public FastSerialCD(Rectangle totalSize, float[,] psf, int processorLimit = -1) :
            this(totalSize, totalSize, psf, PSF.CalcPSFSquared(psf), processorLimit)
        {
            
        }

        public FastSerialCD(Rectangle totalSize, Rectangle patchSize, float[,] psf, float[,] psfSquared, int processorLimit = -1)
        {
            this.patch = patchSize;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, totalSize, patchSize);
            MaxLipschitz = Residuals.GetMax(psfSquared);

            parallelOptions = new ParallelOptions();
            parallelOptions.MaxDegreeOfParallelism = processorLimit;
        }

        #region ISubpatchDeconvolver implementation
        public DeconvolutionResult DeconvolvePath(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001F)
        {
            bool converged = false;
            var totalIter = 0;
            var totalTime = new Stopwatch();
            totalTime.Start();
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var max = GetAbsMax(subpatch, reconstruction, bMap, parallelOptions, 0.0f, 1.0f);

                var lambdaMax = 1 / alpha * max.PixelMaxDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                Console.WriteLine("-----------------------------FastGreedyCD with lambda " + lambdaCurrent + "------------------------");
                var pathResult = Deconvolve(subpatch, reconstruction, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                converged = lambdaMin == lambdaCurrent & pathResult.Converged;
                totalIter += pathResult.IterationCount;

                if (converged)
                    break;
            }
            totalTime.Stop();

            return new DeconvolutionResult(converged, totalIter, totalTime.Elapsed);
        }

        public DeconvolutionResult Deconvolve(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 0.0001F)
        {
            var watch = new Stopwatch();
            watch.Start();

            bool converged = false;
            int iter = 0;
            while (!converged & iter < iterations)
            {
                var maxPixel = GetAbsMax(subpatch, reconstruction, bMap, parallelOptions, lambda, alpha);
                converged = maxPixel.PixelMaxDiff < epsilon;
                if (!converged)
                {
                    var yLocal = maxPixel.Y - patch.Y;
                    var xLocal = maxPixel.X - patch.X;
                    var pixelOld = reconstruction[yLocal, xLocal];
                    reconstruction[yLocal, xLocal] = maxPixel.PixelNew;
                    UpdateGradients(bMap, psf2, parallelOptions, maxPixel.Y, maxPixel.X, pixelOld - maxPixel.PixelNew);
                    if (iter % 50 == 0)
                        Console.WriteLine("iter\t" + iter + "\tcurrentUpdate\t" + Math.Abs(maxPixel.PixelNew - pixelOld));
                    iter++;
                }
            }
            watch.Stop();

            return new DeconvolutionResult(converged, iter, watch.Elapsed);
        }
        #endregion

        #region IDeconvolver implementation
        public DeconvolutionResult DeconvolvePath(float[,] xImage, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            return DeconvolvePath(patch, xImage, bMap, lambdaMin, lambdaFactor, alpha, maxPathIteration, maxIteration, epsilon);
        }

        public DeconvolutionResult Deconvolve(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon = 1e-4f)
        {
            return Deconvolve(patch, xImage, bMap, lambda, alpha, maxIteration, epsilon);
        }

        public void ResetLipschitzMap(float[,] psf)
        {
            var psf2Local = PSF.CalcPSFSquared(psf);
            var maxFull = Residuals.GetMax(psf2Local);
            MaxLipschitz = maxFull;
            aMap = PSF.CalcAMap(psf, patch);

            var maxCut = Residuals.GetMax(psf2);
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                    psf2[i, j] *= (maxFull / maxCut);
        }

        public float GetAbsMaxDiff(float[,] xImage, float[,] gradients, float lambda, float alpha)
        {
            return GetAbsMax(this.patch, xImage, gradients, this.parallelOptions, lambda, alpha).PixelMaxDiff;
        }
        #endregion

        private Pixel GetAbsMax(Rectangle subpatch, float[,] xImage, float[,] gradients, ParallelOptions options, float lambda, float alpha)
        {
            var maxPixels = new Pixel[subpatch.YExtent()];
            Parallel.For(subpatch.Y, subpatch.YEnd, options, (y) =>
            {
                var yLocal = y;

                var currentMax = new Pixel(-1, -1, 0, 0);
                for (int x = subpatch.X; x < subpatch.XEnd; x++)
                {
                    var xLocal = x;
                    var currentA = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    //var xTmp = old + bMap[y, x] / currentA;
                    //xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
                    var xTmp = ElasticNet.ProximalOperator(old * currentA + gradients[y, x], currentA, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax.PixelMaxDiff < Math.Abs(xDiff))
                    {
                        currentMax.Y = y;
                        currentMax.X = x;
                        currentMax.PixelMaxDiff = Math.Abs(xDiff);
                        currentMax.PixelNew = xTmp;
                    }
                }
                maxPixels[yLocal] = currentMax;
            });

            var maxPixel = new Pixel(-1, -1, 0, 0);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel.PixelMaxDiff < maxPixels[i].PixelMaxDiff)
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

        private Pixel GetAbsMaxSingle(Rectangle subpatch, float[,] xImage, float[,] gradients, float lambda, float alpha)
        {
            var maxPixels = new Pixel[subpatch.YExtent()];
            for(int y = subpatch.Y; y < subpatch.YEnd; y++)
            {
                var yLocal = y;

                var currentMax = new Pixel(-1, -1, 0, 0);
                for (int x = subpatch.X; x < subpatch.XEnd; x++)
                {
                    var xLocal = x;
                    var currentA = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    //var xTmp = old + bMap[y, x] / currentA;
                    //xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
                    var xTmp = ElasticNet.ProximalOperator(old * currentA + gradients[y, x], currentA, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax.PixelMaxDiff < Math.Abs(xDiff))
                    {
                        currentMax.Y = y;
                        currentMax.X = x;
                        currentMax.PixelMaxDiff = Math.Abs(xDiff);
                        currentMax.PixelNew = xTmp;
                    }
                }
                maxPixels[yLocal] = currentMax;
            }

            var maxPixel = new Pixel(-1, -1, 0, 0);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel.PixelMaxDiff < maxPixels[i].PixelMaxDiff)
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

        private static void UpdateGradientsSingle(float[,] gradients, float[,] psf2, int yPixel, int xPixel, float xDiff)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;

            var yMin = Math.Max(yPixel - yPsf2Half, 0);
            var xMin = Math.Max(xPixel - xPsf2Half, 0);
            var yMax = Math.Min(yPixel - yPsf2Half + psf2.GetLength(0), gradients.GetLength(0));
            var xMax = Math.Min(xPixel - xPsf2Half + psf2.GetLength(1), gradients.GetLength(1));
            for (int i = yMin; i < yMax; i++)
                for (int j = xMin; j < xMax; j++)
                {
                    var yBUpdate = i + yPsf2Half - yPixel;
                    var xBUpdate = j + xPsf2Half - xPixel;
                    gradients[i, j] += psf2[yBUpdate, xBUpdate] * xDiff;
                }
        }

        private static void UpdateGradients(float[,] gradients, float[,] psf2, ParallelOptions options,  int yPixel, int xPixel, float xDiff)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;

            var yMin = Math.Max(yPixel - yPsf2Half, 0);
            var xMin = Math.Max(xPixel - xPsf2Half, 0);
            var yMax = Math.Min(yPixel - yPsf2Half + psf2.GetLength(0), gradients.GetLength(0));
            var xMax = Math.Min(xPixel - xPsf2Half + psf2.GetLength(1), gradients.GetLength(1));
            Parallel.For(yMin, yMax, options, (i) =>
            {
                for (int j = xMin; j < xMax; j++)
                {
                    var yBUpdate = i + yPsf2Half - yPixel;
                    var xBUpdate = j + xPsf2Half - xPixel;
                    gradients[i, j] += psf2[yBUpdate, xBUpdate] * xDiff;
                }
            });
        }

        private struct Pixel
        {
            public int Y;
            public int X;
            public float PixelMaxDiff;
            public float PixelNew;

            public Pixel(int y, int x, float xMax, int xNew)
            {
                Y = y;
                X = x;
                PixelMaxDiff = xMax;
                PixelNew = xNew;
            }
        }
    }
}
