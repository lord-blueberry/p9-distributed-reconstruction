using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using static Single_Reference.Common;

using MPI;

namespace DistributedReconstruction
{
    class MPIGreedyCD
    {
        readonly Intracommunicator comm;
        readonly Rectangle imageSection;
        readonly float[,] psf2;
        readonly float[,] aMap;

        public MPIGreedyCD(Intracommunicator comm, Rectangle totalSize, Rectangle imageSection, float[,] psf)
        {
            this.comm = comm;
            this.imageSection = imageSection;
            psf2 = PSF.CalcPSFSquared(psf);
            aMap = PSF.CalcAMap(psf, totalSize, imageSection);
        }

        public Statistics DeconvolvePath(float[,] xImage, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            bool converged = false;
            var watch = new Stopwatch();
            watch.Start();
            long totalIterations = 0;
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var max = GetAbsMax(xImage, bMap, 0.0f, 1.0f);

                var lambdaMax = 1 / alpha * max.MaxDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                Console.WriteLine("-----------------------------MPIGreedy with lambda " + lambdaCurrent + "------------------------");
                var pathConverged = Deconvolve(xImage, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                converged = lambdaMin == lambdaCurrent & pathConverged.Converged;

                totalIterations += pathConverged.IterationsRun;
                if (converged)
                    break;
            }
            watch.Stop();
            return new Statistics(converged, totalIterations, watch.Elapsed);
        }

        public Statistics Deconvolve(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon = 1e-4f)
        {
            var watch = new Stopwatch();
            watch.Start();

            bool converged = false;
            int iter = 0;
            while (!converged & iter < maxIteration)
            {
                var maxPixel = GetAbsMax(xImage, bMap, lambda, alpha);
                converged = maxPixel.MaxDiff < epsilon;
                if (!converged)
                {
                    if(imageSection.PointInRectangle(maxPixel.Y, maxPixel.X))
                    {
                        var yLocal = maxPixel.Y - imageSection.Y;
                        var xLocal = maxPixel.X - imageSection.X;
                        xImage[yLocal, xLocal] += maxPixel.MaxDiff * maxPixel.Sign;
                    }
                    
                    UpdateB(bMap, psf2, maxPixel.Y, maxPixel.X, maxPixel);
                    if (iter % 50 == 0 & comm.Rank == 0)
                        Console.WriteLine("iter\t" + iter + "\tcurrentUpdate\t" + maxPixel.MaxDiff);
                    iter++;
                }
            }
            watch.Stop();
            double iterPerSecond = iter;
            iterPerSecond = iterPerSecond / watch.ElapsedMilliseconds * 1000.0;
            if (comm.Rank == 0)
                Console.WriteLine(iter + " iterations in:" + watch.Elapsed + "\t" + iterPerSecond + " iterations per second");

            return new Statistics(converged, iter, watch.Elapsed);
        }

        private Pixel GetAbsMax(float[,] xImage, float[,] bMap, float lambda, float alpha)
        {
            var maxPixels = new Pixel[imageSection.YExtent()];
            Parallel.For(0, imageSection.YExtent(), (y) =>
            {
                var currentMax = new Pixel(0, -1, -1);
                for (int x = 0; x < imageSection.XExtent(); x++)
                {
                    var currentA = aMap[y, x];
                    var old = xImage[y, x];
                    var xTmp = ElasticNet.ProximalOperator(old * currentA + bMap[y, x], currentA, lambda, alpha);
                    var xDiff = xTmp - old;

                    if (currentMax.MaxDiff < Math.Abs(xDiff))
                    {
                        var yGlobal = y + imageSection.Y;
                        var xGlobal = x + imageSection.X;
                        currentMax = new Pixel(xDiff, yGlobal, xGlobal);
                    }
                }
                maxPixels[y] = currentMax;
            });

            var maxPixelLocal = new Pixel(0, -1, -1);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixelLocal.MaxDiff < maxPixels[i].MaxDiff)
                    maxPixelLocal = maxPixels[i];

            var maxPixelGlobal = comm.Allreduce(maxPixelLocal, (aC, bC) => aC.MaxDiff > bC.MaxDiff ? aC : bC);
            return maxPixelGlobal;
        }

        private void UpdateB(float[,] b, float[,] bUpdate, int yPixel, int xPixel, Pixel max)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imageSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imageSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imageSection.YEnd);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imageSection.XEnd);
            Parallel.For(yBMin, yBMax, (i) =>
            {
                for (int j = xBMin; j < xBMax; j++)
                {
                    var yLocal = i - imageSection.Y;
                    var xLocal = j - imageSection.X;
                    var yBUpdate = i + yBHalf - yPixel;
                    var xBUpdate = j + xBHalf - xPixel;
                    b[yLocal, xLocal] -= bUpdate[yBUpdate, xBUpdate] * (max.MaxDiff * max.Sign);
                }
            });

        }

        #region maxPixel
        [Serializable]
        private class Pixel
        {
            public float MaxDiff { get;  set; }
            public int Sign { get; set; }

            public int Y { get;  set; }
            public int X { get; set; }


            public Pixel(float pixelValue, int y, int x)
            {
                MaxDiff = Math.Abs(pixelValue);
                Sign = Math.Sign(pixelValue);
                Y = y;
                X = x;
            }
        }
        #endregion

        public class Statistics
        {
            public bool Converged { get; private set; }
            public long IterationsRun { get; private set; }
            public TimeSpan ElapsedMilliseconds { get; private set; }

            public Statistics(bool converged, long iterations, TimeSpan elapsed)
            {
                Converged = converged;
                IterationsRun = iterations;
                ElapsedMilliseconds = elapsed;
            }


        }
    }
}
