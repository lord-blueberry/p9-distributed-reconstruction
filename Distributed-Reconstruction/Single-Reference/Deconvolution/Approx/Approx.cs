using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.Approx
{
    public class Approx
    {
        private static double CalcESO(int processorCount, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (processorCount - 1.0) / (Math.Max(1.0, (blockCount - 1)));

        const float ACTIVE_SET_CUTOFF = 1e-8f;
        int threadCount;
        int blockSize;
        float randomFraction;
        float searchFraction;

        float MaxLipschitz;
        float[,] aMap;
        float[,] psf;
        float[,] psf2;
        Rectangle totalSize;

        public Approx(Rectangle totalSize, float[,] psf, int threadCount, int blockSize, float randomFraction, float searchFraction)
        {
            this.totalSize = totalSize;
            this.psf = psf;
            this.psf2 = PSF.CalcPSFSquared(psf);
            MaxLipschitz = (float)PSF.CalcMaxLipschitz(psf);
            this.aMap = PSF.CalcAMap(psf, totalSize);

            this.threadCount = threadCount;
            this.blockSize = blockSize;
            this.randomFraction = randomFraction;
            this.searchFraction = searchFraction;
        }

        public void ResetAMap(float[,] psf)
        {
            var psf2Local = PSF.CalcPSFSquared(psf);
            var maxFull = Residuals.GetMax(psf2Local);
            MaxLipschitz = maxFull;
            aMap = PSF.CalcAMap(psf, totalSize);
            this.psf = psf;

            var maxCut = Residuals.GetMax(psf2);
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                    psf2[i, j] *= (maxFull / maxCut);
        }

        public static List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, float lambda, float alpha, float[,] lipschitzMap)
        {
            var debug = new float[xExplore.GetLength(0), xExplore.GetLength(1)];
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
                        debug[y, x] = 1.0f;
                    }
                }
            FitsIO.Write(debug, "activeSet.fits");
            return output;
        }

        private static float GetAbsMax(float[,] xImage, float[,] gradients, float[,] lipschitz, float lambda, float alpha)
        {
            var maxPixels = new float[xImage.GetLength(0)];
            Parallel.For(0, xImage.GetLength(0), (y) =>
            {
                var yLocal = y;

                var currentMax = 0.0f;
                for (int x = 0; x < xImage.GetLength(1); x++)
                {
                    var xLocal = x;
                    var L = lipschitz[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    //var xTmp = old + bMap[y, x] / currentA;
                    //xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
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



        public void Deconvolve(float[,] xImage, float[,] gradients, float[,] psf, float[,] psfFull, float lambda, float alpha, Random random, int maxIteration = 100, float epsilon = 1e-4f)
        {
            Stopwatch watch = new Stopwatch();
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var gExplore = gradients;
            var gCorrection = new float[gradients.GetLength(0), gradients.GetLength(1)];

            var shared = new SharedData(lambda, alpha, blockSize, blockSize, threadCount,
                CountNonZero(psf), psf2, aMap,
                xExplore, xCorrection, gExplore, gCorrection, random);
            shared.ActiveSet = GetActiveSet(xExplore, gExplore, lambda, alpha, shared.AMap);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.maxLipschitz = MaxLipschitz;
            shared.MaxConcurrentIterations = 600;

            var theta = DeconvolveConcurrentTest(shared, maxIteration, epsilon, xImage, gradients, psf, psfFull);

            var theta0 = shared.ProcessorCount / (shared.XExpl.Length / (shared.YBlockSize * shared.XBlockSize));
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xImage[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];
        }

        private float DeconvolveConcurrentTest(SharedData shared, int activeSetIterations, float epsilon, float[,] xImage, float[,] residuals, float[,] psf, float[,] psfFull)
        {
            var watch = new Stopwatch();

            var blockCount = shared.XExpl.Length / (shared.YBlockSize * shared.XBlockSize);
            //var blockCount = shared.ActiveSet.Count;
            var theta0 = shared.ProcessorCount / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.testRestart = 0.0f;
            shared.theta0 = theta0;

            var xDiffs = new float[shared.ProcessorCount];
            var deconvolvers = new List<AsyncDeconvolver>(shared.ProcessorCount);
            for (int i = 0; i < shared.ProcessorCount; i++)
                deconvolvers.Add(new AsyncDeconvolver(shared, i + 1, shared.ProcessorCount, searchFraction, theta0));

            int iter = 0;
            var converged = false;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {
                shared.asyncFinished = 0;
                watch.Start();
                //async iterations
                Parallel.For(0, deconvolvers.Count, (i) =>
                {
                    deconvolvers[i].Deconvolve();
                });
                watch.Stop();


                if (shared.testRestart > 0.0f)
                {
                    Console.WriteLine("restarting");
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
                    blockCount = shared.XExpl.Length / (shared.YBlockSize * shared.XBlockSize);
                    //blockCount = shared.ActiveSet.Count; 
                    theta0 = shared.ProcessorCount / (float)blockCount;
                    for (int i = 0; i < deconvolvers.Count; i++)
                        deconvolvers[i].Theta = theta0;
                    shared.testRestart = 0.0f;
                    shared.theta0 = theta0;
                }

                if (deconvolvers.Max(d => d.xDiffMax) < epsilon)
                {
                    if (GetAbsMax(shared.XExpl, shared.GExpl, shared.AMap, shared.Lambda, shared.Alpha) < epsilon)
                    {
                        converged = true;
                    }
                }

                Console.WriteLine("Done Active Set iteration " + iter);
                iter++;
            }

            return deconvolvers[0].Theta;
        }

    }
}
