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
        readonly Rectangle imageSection;
        readonly Rectangle psfSize;
        readonly Complex[,] psfCorrelation;
        readonly float[,] psf2;
        readonly float[,] aMap;

        public FastGreedyCD(int nCores)
        {

        }

        public FastGreedyCD(Rectangle totalSize, Rectangle imageSection, float[,] psf, Complex[,] psfCorrelation) :
            this(totalSize, imageSection, psf, psfCorrelation, PSF.CalcPSFSquared(psfCorrelation))
        {

        }

        public FastGreedyCD(Rectangle totalSize, Rectangle imageSection, float[,] psf, Complex[,] psfCorrelation, float[,] psfSquared)
        {
            this.imageSection = imageSection;
            psfSize = new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1));
            this.psfCorrelation = psfCorrelation;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, totalSize, imageSection);
        }

        public FastGreedyCD(Rectangle totalSize, Rectangle imageSection, float[,] psf, Complex[,] psfCorrelation, float[,] psfSquared, int nCores):
            this(totalSize, imageSection, psf, psfCorrelation, psfSquared)
        {

        }

        public bool DeconvolvePath(float[,] reconstruction, float[,] residuals, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            var bMap = Residuals.CalcBMap(residuals, psfCorrelation, psfSize);



            return false;
        }

        public bool Deconvolve(float[,] xImage, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon)
        {
            var watch = new Stopwatch();
            watch.Start();

            int iter = 0;
            bool converged = false;
            while (!converged & iter < maxIteration)
            {
                var maxPixels = new MaxPixel[imageSection.YExtent()];
                Parallel.For(imageSection.Y, imageSection.YEnd, (y) =>
                {
                    var yLocal = y - imageSection.Y;

                    var currentMax = new MaxPixel(-1, -1, 0, 0);
                    for (int x = imageSection.X; x < imageSection.XEnd; x++)
                    {                 
                        var xLocal = x - imageSection.X;
                        var currentA = aMap[yLocal, xLocal];
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + bMap[y, x] / currentA;
                        xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
                        var xDiff = old - xTmp;

                        if (currentMax.PixelMax < Math.Abs(xDiff))
                        {
                            currentMax.Y = y;
                            currentMax.X = x;
                            currentMax.PixelMax = Math.Abs(xDiff);
                            currentMax.PixelNew = xTmp;
                        }
                    }
                    maxPixels[yLocal] = currentMax;
                });

                var maxPixel = new MaxPixel(-1, -1, 0, 0);
                for (int i = 0; i < maxPixels.Length; i++)
                    if (maxPixel.PixelMax < maxPixels[i].PixelMax)
                        maxPixel = maxPixels[i];

                converged = maxPixel.PixelMax < epsilon;
                if (!converged)
                {
                    var yLocal = maxPixel.Y - imageSection.Y;
                    var xLocal = maxPixel.X - imageSection.X;
                    var xOld = xImage[yLocal, xLocal];
                    xImage[yLocal, xLocal] = maxPixel.PixelNew;
                    UpdateBSingle(bMap, psf2, imageSection, maxPixel.Y, maxPixel.X, xOld - maxPixel.PixelNew);
                }
            }

            return converged;
        }

        private static void UpdateBSingle(float[,] b, float[,] bUpdate, Rectangle imageSection, int yPixel, int xPixel, float xDiff)
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

        private static void UpdateB(float[,] b, float[,] bUpdate, Rectangle imageSection, int yPixel, int xPixel, float xDiff)
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

        private struct MaxPixel
        {
            public int Y;
            public int X;
            public float PixelMax;
            public float PixelNew;

            public MaxPixel(int y, int x, float xMax, int xNew)
            {
                Y = y;
                X = x;
                PixelMax = xMax;
                PixelNew = xNew;
            }
        }
    }
}
