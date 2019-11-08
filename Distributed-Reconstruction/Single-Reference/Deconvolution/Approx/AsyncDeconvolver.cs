using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.Approx
{
    class AsyncDeconvolver
    {
        readonly SharedData shared;
        readonly float[] updateCache;

        //configuration for pseudo random block selection
        readonly int id;
        readonly int totalNodes;
        float searchPercentage;
        public float xDiffMax { get; set; }

        readonly Random random;

        public float Theta { get; set; }

        public AsyncDeconvolver(SharedData shared, int id, int total, float searchPercentage, float theta0)
        {
            this.id = id;
            this.totalNodes = total;
            this.searchPercentage = searchPercentage;
            this.shared = shared;

            this.random = new Random();
            this.updateCache = new float[shared.Psf2.GetLength(1)];

            Theta = theta0;
        }

        public void Deconvolve()
        {
            var update = 0.0f;
            var blockCount = shared.XExpl.Length;
            //var blockCount = shared.ActiveSet.Count;
            float eta = 1.0f / blockCount;

            var beta = CalcESO(shared.ProcessorCount, shared.DegreeOfSeperability, blockCount);
            var continueAsync = Thread.VolatileRead(ref shared.asyncFinished) == 0;
            for (int inner = 0; inner < shared.MaxConcurrentIterations & continueAsync; inner++)
            {
                continueAsync = Thread.VolatileRead(ref shared.asyncFinished) == 0;
                var stepFactor = (float)beta * Theta / shared.theta0;
                var theta2 = Theta * Theta;
                var blockIdx = GetPseudoRandomBlock(stepFactor, theta2);
                var blockSample = shared.ActiveSet[blockIdx];
                var yPixel = blockSample.Item1;
                var xPixel = blockSample.Item2;
                var step = shared.AMap[yPixel, xPixel] * stepFactor;

                var correctionFactor = -(1.0f - Theta / shared.theta0) / theta2;

                var xExpl = Thread.VolatileRead(ref shared.XExpl[yPixel, xPixel]);
                update = theta2 * Thread.VolatileRead(ref shared.GCorr[yPixel, xPixel]) + Thread.VolatileRead(ref shared.GExpl[yPixel, xPixel]) + xExpl * step;
                update = ElasticNet.ProximalOperator(update, step, shared.Lambda, shared.Alpha) - xExpl;
                xDiffMax = Math.Max(xDiffMax, Math.Abs(update));

                //update gradients
                if (0.0f != Math.Abs(update))
                {
                    UpdateGradients(shared.GExpl, shared.GCorr, shared.Psf2, updateCache, yPixel, xPixel, correctionFactor, update);
                    var oldExplore = shared.XExpl[yPixel, xPixel];    //does not need to be volatile, this index is blocked until this process is finished, and we already made sure with a volatile read that the latest value is in the cache
                    var oldXCorr = Thread.VolatileRead(ref shared.XCorr[yPixel, xPixel]);
                    var newXExplore = oldExplore + update;

                    Thread.VolatileWrite(ref shared.XExpl[yPixel, xPixel], shared.XExpl[yPixel, xPixel] + update);
                    Thread.VolatileWrite(ref shared.XCorr[yPixel, xPixel], shared.XCorr[yPixel, xPixel] + update * correctionFactor);

                    //not 100% sure this is the correct generalization from single pixel thread rule to block rule
                    var testRestartUpdate = (update) * (newXExplore - (theta2 * oldXCorr + oldExplore));
                    ConcurrentUpdateTestRestart(ref shared.testRestart, eta, testRestartUpdate);
                }

                //unlockBlock
                Thread.VolatileWrite(ref shared.BlockLock[blockIdx], 0);
                Theta = (float)(Math.Sqrt((theta2 * theta2) + 4 * (theta2)) - theta2) / 2.0f;
            }

            Thread.VolatileWrite(ref shared.asyncFinished, 1);
        }

        private int GetPseudoRandomBlock(float stepFactor, float theta2)
        {
            var startIdx = GetRandomBlockIdx(random, id, shared.BlockLock);

            var currentIdx = startIdx;
            var currentPixelValue = GetMaxAbsPixelValue(shared, shared.ActiveSet[currentIdx], stepFactor, theta2);
            var searchLength = (int)(shared.ActiveSet.Count / totalNodes * searchPercentage);
            for (int i = 1; i < searchLength; i++)
            {
                var checkIdx = (startIdx + i) % shared.ActiveSet.Count;
                if (Thread.VolatileRead(ref shared.BlockLock[checkIdx]) == 0)
                {
                    //check if next block is better than current block
                    var checkPixelValue = GetMaxAbsPixelValue(shared, shared.ActiveSet[checkIdx], stepFactor, theta2);

                    if (currentPixelValue < checkPixelValue)
                        if (Interlocked.CompareExchange(ref shared.BlockLock[checkIdx], this.id, 0) == 0)
                        {
                            //locking of better block successful, continue
                            Thread.VolatileWrite(ref shared.BlockLock[currentIdx], 0);
                            currentIdx = checkIdx;
                            currentPixelValue = checkPixelValue;
                        }
                        else
                        {
                            //some other thread conflicted with ours, jump to another random place
                            startIdx = random.Next(0, shared.ActiveSet.Count);
                        }
                }

            }

            return currentIdx;
        }

        private static float GetMaxAbsPixelValue(SharedData shared, Tuple<int, int> block, float stepFactor, float theta2)
        {
            var yPixel = block.Item1;
            var xPixel = block.Item2;

            var step = shared.AMap[yPixel, xPixel] * stepFactor;
            var xExpl = Thread.VolatileRead(ref shared.XExpl[yPixel, xPixel]);
            var update = theta2 * Thread.VolatileRead(ref shared.GCorr[yPixel, xPixel]) + Thread.VolatileRead(ref shared.GExpl[yPixel, xPixel]) + xExpl * step;
            update = ElasticNet.ProximalOperator(update, step, shared.Lambda, shared.Alpha) - xExpl;

            return Math.Abs(update);
        }

        private static void UpdateGradients(float[,] gExplore, float[,] gCorrection, float[,] psf2, float[] lineCache, int yPixel, int xPixel, float correctionFactor, float update)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;

            var yMin = Math.Max(yPixel - yPsf2Half, 0);
            var xMin = Math.Max(xPixel - xPsf2Half, 0);
            var yMax = Math.Min(yPixel - yPsf2Half + psf2.GetLength(0), gExplore.GetLength(0));
            var xMax = Math.Min(xPixel - xPsf2Half + psf2.GetLength(1), gExplore.GetLength(1));
            for (int y = yMin; y < yMax; y++)
            {
                var yPsfIdx = y + yPsf2Half - yPixel;
                for (int x = xMin; x < xMax; x++)
                {
                    var xPsfIdx = x + xPsf2Half - xPixel;
                    lineCache[xPsfIdx] = psf2[yPsfIdx, xPsfIdx] * update;
                }

                for (int x = xMin; x < xMax; x++)
                {
                    var xPsfIdx = x + xPsf2Half - xPixel;

                    var exploreUpdate = lineCache[xPsfIdx];
                    var correctionUpdate = lineCache[xPsfIdx] * correctionFactor;
                    ConcurrentSubtract(gExplore, y, x, exploreUpdate);
                    ConcurrentSubtract(gCorrection, y, x, correctionUpdate);
                }
            }
        }

        static void UpdateBMaps(float[,] updateBlock, int yB, int xB, float[,] psf2, float[,] gExplore, float[,] gCorrection, float correctionFactor)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;
            var yPixelIdx = yB * updateBlock.GetLength(0);
            var xPixelIdx = xB * updateBlock.GetLength(1);

            var yMin = Math.Max(yPixelIdx - yPsf2Half, 0);
            var xMin = Math.Max(xPixelIdx - xPsf2Half, 0);
            var yMax = Math.Min(yPixelIdx + updateBlock.GetLength(0) - yPsf2Half + psf2.GetLength(0), gExplore.GetLength(0));
            var xMax = Math.Min(xPixelIdx + updateBlock.GetLength(1) - xPsf2Half + psf2.GetLength(1), gExplore.GetLength(1));
            for (int globalY = yMin; globalY < yMax; globalY++)
                for (int globalX = xMin; globalX < xMax; globalX++)
                {
                    var exploreUpdate = 0.0f;
                    var correctionUpdate = 0.0f;

                    var yPsfMin = Math.Max(globalY + yPsf2Half - yPixelIdx - updateBlock.GetLength(0) + 1, 0);
                    var xPsfMin = Math.Max(globalX + xPsf2Half - xPixelIdx - updateBlock.GetLength(1) + 1, 0);
                    var yPsfMax = Math.Min(globalY + yPsf2Half - yPixelIdx + 1, psf2.GetLength(0));
                    var xPsfMax = Math.Min(globalX + xPsf2Half - xPixelIdx + 1, psf2.GetLength(1));
                    for (int psfY = yPsfMin; psfY < yPsfMax; psfY++)
                        for (int psfX = xPsfMin; psfX < xPsfMax; psfX++)
                        {
                            var blockY = -1 * (psfY - yPsf2Half - globalY + yPixelIdx);
                            var blockX = -1 * (psfX - xPsf2Half - globalX + xPixelIdx);

                            var update = updateBlock[blockY, blockX] * psf2[psfY, psfX];
                            exploreUpdate += update;
                            correctionUpdate += update * correctionFactor;
                        }

                    ConcurrentSubtract(gExplore, globalY, globalX, exploreUpdate);
                    ConcurrentSubtract(gCorrection, globalY, globalX, correctionUpdate);
                }
        }

        static int GetRandomBlockIdx(Random rand, int id, int[] blockLock)
        {
            var succesfulLock = false;
            var idx = -1;
            do
            {
                idx = rand.Next(0, blockLock.Length);
                var old = Interlocked.CompareExchange(ref blockLock[idx], id, 0);
                succesfulLock = old == 0;
            } while (!succesfulLock);

            return idx;
        }

        static void ConcurrentSubtract(float[,] map, int yIdx, int xIdx, float value)
        {
            var successfulWrite = value == 0.0f;    //skip write if value to subtract is zero
            while (!successfulWrite)
            {
                var read = Thread.VolatileRead(ref map[yIdx, xIdx]);
                var old = Interlocked.CompareExchange(ref map[yIdx, xIdx], read - value, read);
                successfulWrite = old == read | !float.IsNormal(old);
            }
        }

        static void ConcurrentUpdateTestRestart(ref float test, float eta, float subtract)
        {
            var successfulWrite = subtract == 0.0f;    //skip write if value to subtract is zero
            while (!successfulWrite)
            {
                var read = Thread.VolatileRead(ref test);
                var update = (1.0f - eta) * read - subtract;
                var old = Interlocked.CompareExchange(ref test, update, read);
                successfulWrite = old == read | !float.IsNormal(old);
            }
        }
    }
}
