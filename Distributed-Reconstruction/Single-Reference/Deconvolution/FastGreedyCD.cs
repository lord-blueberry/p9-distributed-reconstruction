using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;
using System.Diagnostics;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class FastGreedyCD : IDeconvolver
    {
        readonly Rectangle residualSection;
        readonly Rectangle imageSection;    //image section in residuals. Meaning what window we actually optimize in residuals
        readonly Rectangle psfSize;
        readonly Complex[,] psfCorrelation;
        readonly float[,] psf2;
        readonly float[,] aMap;

        public FastGreedyCD(Rectangle totalSize, float[,] psf) :
            this(totalSize, totalSize, totalSize, psf, PSF.CalcPaddedFourierCorrelation(psf, totalSize), PSF.CalcPSFSquared(psf))
        {

        }

        public FastGreedyCD(Rectangle totalSize, Rectangle residualSection, Rectangle imageSection, float[,] psf) :
            this(totalSize, residualSection, imageSection, psf, PSF.CalcPaddedFourierCorrelation(psf, totalSize), PSF.CalcPSFSquared(psf))
        {

        }

        public FastGreedyCD(Rectangle totalSize, Rectangle residualSection, Rectangle imageSection, float[,] psf, Complex[,] psfCorrelation, float[,] psfSquared)
        {
            this.residualSection = residualSection;
            this.imageSection = imageSection;
            psfSize = new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1));
            this.psfCorrelation = psfCorrelation;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, totalSize, residualSection);
        }

        public bool DeconvolvePath(float[,] xImage, float[,] residuals, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            bool converged = false;
            var bMap = Residuals.CalcBMap(residuals, psfCorrelation, psfSize);
            for (int y = 0; y < bMap.GetLength(0); y++)
                for (int x = 0; x < bMap.GetLength(1); x++)
                    bMap[y, x] /= aMap[y, x];

            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var max = GetAbsMax(xImage, bMap, 0.0f, 1.0f);

                var lambdaMax = 1 / alpha * max.PixelMaxDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                Console.WriteLine("-----------------------------FastGreedyCD with lambda " + lambdaCurrent + "------------------------");
                if (lambdaCurrent > lambdaMin)
                {
                    DeconvolveImpl(xImage, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                }
                else
                {
                    converged = DeconvolveImpl(xImage, bMap, lambdaCurrent, alpha, maxIteration, epsilon);
                    if (converged)
                        break;
                }
            }

            return converged;
        }

        public bool Deconvolve(float[,] xImage, float[,] residuals, float lambda, float alpha, int maxIteration, float epsilon)
        {
            var bMap = Residuals.CalcBMap(residuals, psfCorrelation, psfSize);
            for (int y = 0; y < bMap.GetLength(0); y++)
                for (int x = 0; x < bMap.GetLength(1); x++)
                    bMap[y, x] /= aMap[y, x];
            return DeconvolveImpl(xImage, bMap, lambda, alpha, maxIteration, epsilon);
        }

        private bool DeconvolveImpl(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon)
        {
            var watch = new Stopwatch();
            watch.Start();

            int iter = 0;
            bool converged = false;
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
                    UpdateB(bMap, maxPixel.Y, maxPixel.X, pixelOld - maxPixel.PixelNew);
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

        private MaxPixel GetAbsMax(float[,] xImage, float[,] bMap, float lambda, float alpha)
        {
            var maxPixels = new MaxPixel[imageSection.YExtent()];
            Parallel.For(imageSection.Y, imageSection.YEnd, (y) =>
            {
                var yLocal = y - imageSection.Y;

                var currentMax = new MaxPixel(-1, -1, 0, 0);
                for (int x = imageSection.X; x < imageSection.XEnd; x++)
                {
                    var xLocal = x - imageSection.X;
                    var old = xImage[yLocal, xLocal];
                    var xTmp = old + bMap[y, x];
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

            var maxPixel = new MaxPixel(-1, -1, 0, 0);
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel.PixelMaxDiff < maxPixels[i].PixelMaxDiff)
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

        private void UpdateBSingle(float[,] bMap, int yPixel, int xPixel, float xDiff)
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
                    var yLocal = i;
                    var xLocal = j;
                    var yBUpdate = i + yPsf2Half - yPixel;
                    var xBUpdate = j + xPsf2Half - xPixel;
                    bMap[yLocal, xLocal] += psf2[yBUpdate, xBUpdate] * xDiff;
                }
        }

        private void UpdateB(float[,] bMap, int yPixel, int xPixel, float xDiff)
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
                    bMap[i, j] += (psf2[yBUpdate, xBUpdate] * xDiff) / aMap[i, j];
                }
            });
        }

        private struct MaxPixel
        {
            public int Y;
            public int X;
            public float PixelMaxDiff;
            public float PixelNew;

            public MaxPixel(int y, int x, float xMax, int xNew)
            {
                Y = y;
                X = x;
                PixelMaxDiff = xMax;
                PixelNew = xNew;
            }
        }
    }
}
