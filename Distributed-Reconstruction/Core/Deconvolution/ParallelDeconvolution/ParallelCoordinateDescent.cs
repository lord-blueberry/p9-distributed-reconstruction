﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

using static Core.Common;

namespace Core.Deconvolution.ParallelDeconvolution
{
    /// <summary>
    /// Main class of the parallel coordinate descent algorithm. It uses an AsyncDeconvolver object for each processor in use, and the SharedData object contains all the data shared between the processors.
    /// </summary>
    public class ParallelCoordinateDescent
    {
        int processorCount;
        int concurrentIterations;
        float searchFraction;

        float[,] aMap;
        float[,] psf;
        float[,] psf2;
        Rectangle totalSize;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="totalSize"></param>
        /// <param name="psf"></param>
        /// <param name="processorCount">Number of async. processors to use</param>
        /// <param name="concurrentIterations">Number of async iterations each processor performs in one active set iteration</param>
        /// <param name="searchFraction"></param>
        public ParallelCoordinateDescent(Rectangle totalSize, float[,] psf, int processorCount = 8, int concurrentIterations = 1000,  float searchFraction = 0.1f)
        {
            this.totalSize = totalSize;
            this.psf = psf;
            this.psf2 = PSF.CalcPSFSquared(psf);
            this.aMap = PSF.CalcAMap(psf, totalSize);

            this.processorCount = processorCount;
            this.concurrentIterations = concurrentIterations;
            this.searchFraction = searchFraction;
        }

        public void ResetAMap(float[,] psf)
        {
            var psf2Local = PSF.CalcPSFSquared(psf);
            var maxFull = Residuals.GetMax(psf2Local);
            aMap = PSF.CalcAMap(psf, totalSize);
            this.psf = psf;

            var maxCut = Residuals.GetMax(psf2);
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                    psf2[i, j] *= (maxFull / maxCut);
        }

        private static List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, float lambda, float alpha, float[,] lipschitzMap)
        {
            var output = new List<Tuple<int, int>>();
            for (int y = 0; y < xExplore.GetLength(0); y++)
                for (int x = 0; x < xExplore.GetLength(1); x++)
                {
                    var lipschitz = lipschitzMap[y, x];
                    var tmp = gExplore[y, x] + xExplore[y, x] * lipschitz;
                    tmp = ElasticNet.ProximalOperator(tmp, lipschitz, lambda, alpha);
                    if (0.0f < Math.Abs(tmp - xExplore[y, x]))
                    {
                        output.Add(new Tuple<int, int>(y, x));
                    }
                }
            return output;
        }

        public float GetAbsMax(float[,] xImage, float[,] gradients, float lambda, float alpha)
        {
            var maxPixels = new float[xImage.GetLength(0)];
            Parallel.For(0, xImage.GetLength(0), (y) =>
            {
                var yLocal = y;

                var currentMax = 0.0f;
                for (int x = 0; x < xImage.GetLength(1); x++)
                {
                    var xLocal = x;
                    var L = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    var xTmp = ElasticNet.ProximalOperator(old * L + gradients[y, x], L, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax < Math.Abs(xDiff))
                        currentMax = Math.Abs(xDiff);
                }
                maxPixels[yLocal] = currentMax;
            });

            var maxPixel = 0.0f;
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel < maxPixels[i])
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

        #region deconvolveApprox
        /// <summary>
        /// Deconvolve with gradient acceleration
        /// </summary>
        /// <param name="xImage">intermediate solution</param>
        /// <param name="gradients">map of gradients</param>
        /// <param name="lambda">elastic net regularization parameter</param>
        /// <param name="alpha"></param>
        /// <param name="maxActiveSetIterations">maximum number of active set iterations</param>
        /// <param name="epsilon">abort algorithm when a change to the xImage becomes smaller than epsilon</param>
        /// <returns>PCDM Runtime Statistics</returns>
        public PCDMStatistics DeconvolveAccelerated(float[,] xImage, float[,] gradients, float lambda, float alpha, int maxActiveSetIterations = 100, float epsilon = 1e-4f)
        {
            Stopwatch watch = new Stopwatch();
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var gExplore = gradients;
            var gCorrection = new float[gradients.GetLength(0), gradients.GetLength(1)];

            var shared = new SharedData(lambda, alpha, processorCount,
                CountNonZero(psf), psf2, aMap,
                xExplore, xCorrection, gExplore, gCorrection);
            shared.ActiveSet = GetActiveSet(xExplore, gExplore, lambda, alpha, shared.AMap);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.MaxConcurrentIterations = concurrentIterations;

            var output = DeconvolveAcceleratedConcurrent(shared, maxActiveSetIterations, epsilon);

            var theta0 = shared.ProcessorCount / (shared.XExpl.Length);
            var theta = output.Theta;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xImage[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];

            return output;
        }

        private PCDMStatistics DeconvolveAcceleratedConcurrent(SharedData shared, int activeSetIterations, float epsilon)
        {
            var watch = new Stopwatch();

            var blockCount = shared.XExpl.Length;
            //var blockCount = shared.ActiveSet.Count;
            var theta0 = shared.ProcessorCount / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.TestRestart = 0.0f;
            shared.Theta0 = theta0;

            var xDiffs = new float[shared.ProcessorCount];
            var deconvolvers = new List<AsyncDeconvolver.Approx>(shared.ProcessorCount);
            for (int i = 0; i < shared.ProcessorCount; i++)
                deconvolvers.Add(new AsyncDeconvolver.Approx(shared, i + 1, shared.ProcessorCount, searchFraction, theta0));

            int iter = 0;
            var converged = false;
            var lastAbsMax = GetAbsMax(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha);
            var concurrentFactor = -1f;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {
                shared.AsyncFinished = 0;
                watch.Start();
                //async iterations
                Parallel.For(0, deconvolvers.Count, (i) =>
                {
                    deconvolvers[i].Deconvolve();
                });
                watch.Stop();

                if (concurrentFactor == -1f)
                    concurrentFactor = lastAbsMax / deconvolvers.Max(d => d.DiffMax);

                var currentAbsMax = GetAbsMax(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha);
                var activeSetValid = currentAbsMax > lastAbsMax && lastAbsMax / deconvolvers.Max(d => d.DiffMax) > concurrentFactor;
                activeSetValid |= lastAbsMax / deconvolvers.Max(d => d.DiffMax) > concurrentFactor * 2;
                Console.WriteLine("active set iteration: " + iter + " current max pixel: " + currentAbsMax);
                
                if (shared.TestRestart > 0.0f | activeSetValid)
                {
                    Console.WriteLine("restarting");
                    concurrentFactor = -1f;
                    var currentTheta = deconvolvers[0].Theta;
                    var tmpTheta = currentTheta < 1.0f ? ((currentTheta * currentTheta) / (1.0f - currentTheta)) : theta0;
                    Parallel.For(0, shared.XExpl.GetLength(0), (y) =>
                    {
                        for (int x = 0; x < shared.XExpl.GetLength(1); x++)
                        {
                            shared.XExpl[y, x] += tmpTheta * shared.XCorr[y, x];
                            shared.XCorr[y, x] = 0;
                            shared.GExpl[y, x] += tmpTheta * shared.GCorr[y, x];
                            shared.GCorr[y, x] = 0;
                        }
                    });

                    //new active set
                    shared.ActiveSet = GetActiveSet(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha, shared.AMap);
                    shared.BlockLock = new int[shared.ActiveSet.Count];
                    blockCount = shared.XExpl.Length;
                    theta0 = shared.ProcessorCount / (float)blockCount;
                    for (int i = 0; i < deconvolvers.Count; i++)
                        deconvolvers[i].Theta = theta0;
                    shared.TestRestart = 0.0f;
                    shared.Theta0 = theta0;
                }

                lastAbsMax = currentAbsMax;
                if (currentAbsMax < epsilon)
                    converged = true;

                iter++;
            }

            return new PCDMStatistics(converged, deconvolvers.Sum(x => x.AsyncIterations), deconvolvers[0].Theta);
        }
        #endregion

        #region deconvolve PCDM
        /// <summary>
        /// Deconvolve
        /// </summary>
        /// <param name="xImage">intermediate solution</param>
        /// <param name="gradients">map of gradients</param>
        /// <param name="lambda">elastic net regularization parameter</param>
        /// <param name="alpha"></param>
        /// <param name="maxActiveSetIterations">maximum number of active set iterations</param>
        /// <param name="epsilon">abort algorithm when a change to the xImage becomes smaller than epsilon</param>
        /// <returns>PCDM Runtime Statistics</returns>
        public PCDMStatistics Deconvolve(float[,] xImage, float[,] gradients, float lambda, float alpha, int maxActiveSetIterations = 100, float epsilon = 1e-4f)
        {
            var shared = new SharedData(lambda, alpha, processorCount,
                CountNonZero(psf), psf2, aMap,
                xImage, null, gradients, null);
            shared.ActiveSet = GetActiveSet(xImage, gradients, lambda, alpha, shared.AMap);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.MaxConcurrentIterations = concurrentIterations;

            var output = DeconvolvePCDMConcurrent(shared, maxActiveSetIterations, epsilon);

            return output;
        }

        private PCDMStatistics DeconvolvePCDMConcurrent(SharedData shared, int activeSetIterations, float epsilon)
        {
            var watch = new Stopwatch();

            var blockCount = shared.XExpl.Length;
            //var blockCount = shared.ActiveSet.Count;
            var theta0 = shared.ProcessorCount / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.TestRestart = 0.0f;
            shared.Theta0 = theta0;

            var xDiffs = new float[shared.ProcessorCount];
            var deconvolvers = new List<AsyncDeconvolver.PCDM>(shared.ProcessorCount);
            for (int i = 0; i < shared.ProcessorCount; i++)
                deconvolvers.Add(new AsyncDeconvolver.PCDM(shared, i + 1, shared.ProcessorCount, searchFraction));

            int iter = 0;
            var converged = false;
            var lastAbsMax = GetAbsMax(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha);
            var concurrentFactor = -1f;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {
                shared.AsyncFinished = 0;
                watch.Start();
                //async iterations
                Parallel.For(0, deconvolvers.Count, (i) =>
                {
                    deconvolvers[i].Deconvolve();
                });
                watch.Stop();

                if (concurrentFactor == -1f)
                    concurrentFactor = lastAbsMax / deconvolvers.Max(d => d.DiffMax);

                var currentAbsMax = GetAbsMax(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha);
                var activeSetInvalid = currentAbsMax > lastAbsMax && lastAbsMax / deconvolvers.Max(d => d.DiffMax) > concurrentFactor;
                activeSetInvalid |= lastAbsMax / deconvolvers.Max(d => d.DiffMax) > concurrentFactor * 2;

                Console.WriteLine("active set iteration: " + iter + " current max pixel: " + currentAbsMax);

                if (shared.TestRestart > 0.0f | activeSetInvalid)
                {
                    Console.WriteLine("restarting");
                    concurrentFactor = -1f;

                    //new active set
                    shared.ActiveSet = GetActiveSet(shared.XExpl, shared.GExpl, shared.Lambda, shared.Alpha, shared.AMap);
                    shared.BlockLock = new int[shared.ActiveSet.Count];
                    blockCount = shared.XExpl.Length;
                }

                lastAbsMax = currentAbsMax;
                if (currentAbsMax < epsilon)
                    converged = true;
                

                iter++;
            }

            return new PCDMStatistics(converged, deconvolvers.Sum(x => x.AsyncIterations), 0f);
        }
        #endregion

        public class PCDMStatistics
        {
            public bool Converged { get; private set; }
            public int IterationCount { get; private set; }
            public float Theta { get; private set; }


            public PCDMStatistics(bool converged, int iteration, float theta)
            {
                Converged = converged;
                IterationCount = iteration;
                Theta = theta;
            }
        } 
    }
}
