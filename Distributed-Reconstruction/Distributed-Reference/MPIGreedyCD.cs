using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using static Single_Reference.Common;

using MPI;

namespace Distributed_Reference
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

        public bool DeconvolvePath(float[,] xImage, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            bool converged = false;
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var max = GetAbsMax(xImage, bMap, 0.0f, 1.0f);

                var lambdaMax = 1 / alpha * max.PixelMaxDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                Console.WriteLine("-----------------------------MPIGreedy with lambda " + lambdaCurrent + "------------------------");
                var pathConverged = DeconvolveImpl(xImage, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                converged = lambdaMin == lambdaCurrent & pathConverged;

                if (converged)
                    break;
            }

            return converged;
        }

        public bool Deconvolve(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon = 1e-4f)
        {
            return DeconvolveImpl(xImage, bMap, lambda, alpha, maxIteration, epsilon);
        }

        private bool DeconvolveImpl(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon)
        {
            var watch = new Stopwatch();
            watch.Start();

            bool converged = false;
            int iter = 0;
            while (!converged & iter < maxIteration)
            {
                var maxPixel = GetAbsMax(xImage, bMap, lambda, alpha);
                converged = maxPixel.PixelMaxDiff < epsilon;
                if (!converged)
                {
                    var yLocal = maxPixel.Y - imageSection.Y;
                    var xLocal = maxPixel.X - imageSection.X;
                    var pixelOld = xImage[yLocal, xLocal];
                    xImage[yLocal, xLocal] = maxPixel.PixelNew;
                    UpdateB(bMap, psf2, maxPixel.Y, maxPixel.X, pixelOld - maxPixel.PixelNew);
                    if (iter % 50 == 0)
                        Console.WriteLine("iter\t" + iter + "\tcurrentUpdate\t" + Math.Abs(maxPixel.PixelNew - pixelOld));
                    iter++;
                }
            }
            watch.Stop();
            double iterPerSecond = iter;
            iterPerSecond = iterPerSecond / watch.ElapsedMilliseconds * 1000.0;
            Console.WriteLine(iter + " iterations in:" + watch.Elapsed + "\t" + iterPerSecond + " iterations per second");

            return converged;
        }

        private Pixel GetAbsMax(float[,] xImage, float[,] bMap, float lambda, float alpha)
        {
            var maxPixels = new Pixel[imageSection.YExtent()];
            Parallel.For(imageSection.Y, imageSection.YEnd, (y) =>
            {
                var yLocal = y - imageSection.Y;

                var currentMax = new Pixel(-1, -1, 0, 0);
                for (int x = imageSection.X; x < imageSection.XEnd; x++)
                {
                    var xLocal = x - imageSection.X;
                    var currentA = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    var xTmp = old + bMap[y, x] / currentA;
                    xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
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

            var maxPixelLocal = new Pixel(-1, -1, 0, 0);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixelLocal.PixelMaxDiff < maxPixels[i].PixelMaxDiff)
                    maxPixelLocal = maxPixels[i];

            var maxPixelGlobal = comm.Allreduce(maxPixelLocal, (aC, bC) => aC.PixelMaxDiff > bC.PixelMaxDiff ? aC : bC);
            return maxPixelGlobal;
        }

        private void UpdateBSingle(double[,] b, double[,] bUpdate, int yPixel, int xPixel, double xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imageSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imageSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imageSection.YEnd);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imageSection.XEnd);
            for (int i = yBMin; i < yBMax; i++)
                for (int j = xBMin; j < xBMax; j++)
                {
                    var yLocal = i - imageSection.Y;
                    var xLocal = j - imageSection.X;
                    var yBUpdate = i + yBHalf - yPixel;
                    var xBUpdate = j + xBHalf - xPixel;
                    b[yLocal, xLocal] += bUpdate[yBUpdate, xBUpdate] * xDiff;
                }
        }

        private void UpdateB(float[,] b, float[,] bUpdate, int yPixel, int xPixel, float xDiff)
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
                    b[yLocal, xLocal] += bUpdate[yBUpdate, xBUpdate] * xDiff;
                }
            });

        }

        #region maxPixel
        [Serializable]
        private class Pixel
        {
            public float PixelMaxDiff { get;  set; }
            public float PixelNew { get; set; }

            public int Y { get;  set; }
            public int X { get; set; }


            public Pixel(float o, float xDiff, int y, int x)
            {
                PixelMaxDiff = o;
                PixelNew = xDiff;
                Y = y;
                X = x;
            }
        }
        #endregion
    }
}
