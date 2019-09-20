using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;
using System.Diagnostics;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class FastGreedyCD : IDeconvolver, ISubpatchDeconvolver
    {
        /*
         * ImageSize, patchSize and subPatchSize. Only relevant for ISubpatchDeconvolver
         * 
         * imagesize: size of the whole image consisting of several patches.
         * patchSize: a patch inside the image, consisting of several subpatches. bMap.Size == xImage.Size == patchSize
         * subPatchsize: size of a patch inside a patch = subpatch.
         */
        
        readonly Rectangle patch;
        readonly float[,] psf2;
        readonly float[,] aMap;

        public FastGreedyCD(Rectangle imageSize, float[,] psf) :
            this(imageSize, imageSize, psf, PSF.CalcPSFSquared(psf))
        {

        }

        public FastGreedyCD(Rectangle imageSize, Rectangle patchSize, float[,] psf, float[,] psfSquared)
        {
            this.patch = patchSize;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, imageSize, patchSize);
        }

        #region ISubpatchDeconvolver implementation
        public bool DeconvolvePath(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001F)
        {
            bool converged = false;
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var max = GetAbsMax(subpatch, reconstruction, bMap, 0.0f, 1.0f);

                var lambdaMax = 1 / alpha * max.PixelMaxDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                Console.WriteLine("-----------------------------FastGreedyCD with lambda " + lambdaCurrent + "------------------------");
                var pathConverged = Deconvolve(subpatch, reconstruction, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                converged = lambdaMin == lambdaCurrent & pathConverged;

                if (converged)
                    break;
            }

            return converged;
        }

        public bool Deconvolve(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 0.0001F)
        {
            var watch = new Stopwatch();
            watch.Start();

            bool converged = false;
            int iter = 0;
            while (!converged & iter < iterations)
            {
                var maxPixel = GetAbsMax(subpatch, reconstruction, bMap, lambda, alpha);
                converged = maxPixel.PixelMaxDiff < epsilon;
                if (!converged)
                {
                    var yLocal = maxPixel.Y - patch.Y;
                    var xLocal = maxPixel.X - patch.X;
                    var pixelOld = reconstruction[yLocal, xLocal];
                    reconstruction[yLocal, xLocal] = maxPixel.PixelNew;
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
        #endregion

        #region IDeconvolver implementation
        public bool DeconvolvePath(float[,] xImage, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            return DeconvolvePath(patch, xImage, bMap, lambdaMin, lambdaFactor, alpha, maxPathIteration, maxIteration, epsilon);
        }

        public bool Deconvolve(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon = 1e-4f)
        {
            return Deconvolve(patch, xImage, bMap, lambda, alpha, maxIteration, epsilon);
        }
        #endregion

        private Pixel GetAbsMax(Rectangle subpatch, float[,] xImage, float[,] bMap, float lambda, float alpha)
        {
            var maxPixels = new Pixel[subpatch.YExtent()];
            Parallel.For(subpatch.Y, subpatch.YEnd, (y) =>
            {
                var yLocal = y;

                var currentMax = new Pixel(-1, -1, 0, 0);
                for (int x = subpatch.X; x < subpatch.XEnd; x++)
                {
                    var xLocal = x;
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

            var maxPixel = new Pixel(-1, -1, 0, 0);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel.PixelMaxDiff < maxPixels[i].PixelMaxDiff)
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

        private static void UpdateBSingle(float[,] bMap, float[,] psf2, int yPixel, int xPixel, float xDiff)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;

            var yMin = Math.Max(yPixel - yPsf2Half, 0);
            var xMin = Math.Max(xPixel - xPsf2Half, 0);
            var yMax = Math.Min(yPixel - yPsf2Half + psf2.GetLength(0), bMap.GetLength(0));
            var xMax = Math.Min(xPixel - xPsf2Half + psf2.GetLength(1), bMap.GetLength(1));
            for (int i = yMin; i < yMax; i++)
                for (int j = xMin; j < xMax; j++)
                {
                    var yBUpdate = i + yPsf2Half - yPixel;
                    var xBUpdate = j + xPsf2Half - xPixel;
                    bMap[i, j] += psf2[yBUpdate, xBUpdate] * xDiff;
                }
        }

        private static void UpdateB(float[,] bMap, float[,] psf2,  int yPixel, int xPixel, float xDiff)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;

            var yMin = Math.Max(yPixel - yPsf2Half, 0);
            var xMin = Math.Max(xPixel - xPsf2Half, 0);
            var yMax = Math.Min(yPixel - yPsf2Half + psf2.GetLength(0), bMap.GetLength(0));
            var xMax = Math.Min(xPixel - xPsf2Half + psf2.GetLength(1), bMap.GetLength(1));
            Parallel.For(yMin, yMax, (i) =>
            {
                for (int j = xMin; j < xMax; j++)
                {
                    var yBUpdate = i + yPsf2Half - yPixel;
                    var xBUpdate = j + xPsf2Half - xPixel;
                    bMap[i, j] += psf2[yBUpdate, xBUpdate] * xDiff;
                }
            });
        }

        public static double CalcDataPenalty(float[,] residuals)
        {
            double output = 0;
            for (int i = 0; i < residuals.GetLength(0); i++)
                for (int j = 0; j < residuals.GetLength(1); j++)
                    output += residuals[i, j] * residuals[i, j];
            return output;
        }

        public double CalcRegularizationPenalty(float[,] xImage, float lambda, float alpha)
        {
            double output = 0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    output += ElasticNetPenalty(xImage[i, j], alpha) * lambda * 2 * aMap[i, j];

            return output;
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
