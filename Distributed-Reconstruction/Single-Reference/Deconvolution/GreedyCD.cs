using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {

        public static void Deconvolve(double[,] xImage, double[,] b, double[,] psf, double lambda, int maxIteration = 100)
        {
            int iter = 0;
            bool converged = false;

            // the "a" of the parabola equation a* x*x + b*x + c
            var currentLambda = Double.PositiveInfinity;
            var a = CalcPSFSquared(psf);
            var residualTmp = new double[psf.GetLength(0), psf.GetLength(1)];
            while (!converged & iter < maxIteration)
            {
                var maxB = Double.NegativeInfinity;
                var yPixel = -1;
                var xPixel = -1;
                for (int i = 0; i < b.GetLength(0); i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                        if (b[i, j] > maxB)
                        {
                            maxB = b[i, j];
                            yPixel = i;
                            xPixel = j;
                        }

                var xOld = xImage[yPixel, xPixel];
                var xDiff = (maxB / a);             //calculate minimum of parabola, e.g -b/2a. Simplifications led rise to this line
                if(xDiff < currentLambda)
                {
                    currentLambda = xDiff;
                    var xNew = ShrinkAbsolute(xOld + xDiff, lambda);
                    xImage[yPixel, xPixel] = xNew;
                    UpdateB(b, residualTmp, psf, yPixel, xPixel, xOld - xNew);
                }

                converged = currentLambda < lambda;
                iter++;
            }
        }

        public static void UpdateB(double[,] b, double[,] residualTmp, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            for (int i = 0; i < residualTmp.GetLength(0); i++)
                for (int j = 0; j < residualTmp.GetLength(1); j++)
                    residualTmp[i , j] = xDiff * psf[i, j];

            var bYMin = Math.Max(yPixel - psf.GetLength(0), 0);
            var bYMax = Math.Min(yPixel + psf.GetLength(0), b.GetLength(0));
            var bXMin = Math.Max(xPixel - psf.GetLength(1), 0);
            var bXMax = Math.Min(xPixel + psf.GetLength(1), b.GetLength(1));
            for (int y = bYMin; y < bYMax; y++)
            {
                for(int x = bXMin; x < bXMax; x++)
                {
                    var resYOffset = Math.Max(0, y - yPixel + 1);
                    var resYCount = Math.Min(psf.GetLength(0), psf.GetLength(0) - (yPixel - y) + 1);
                    var resXOffset = Math.Max(0, x - xPixel + 1);
                    var resXCount = Math.Min(psf.GetLength(1), psf.GetLength(1) - (xPixel- x) + 1);
                    var bDiff = 0.0;
                    for (int resY = resYOffset; resY < resYCount; resY++)
                        for(int resX = resXOffset; resX < resXCount; resX++)
                            bDiff += psf[resY, resX] * residualTmp[resY, resX];
                    b[y, x] += bDiff;
                }
            }
        }

        private static double CalcPSFSquared(double[,] psf)
        {
            double squared = 0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(0); x++)
                    squared += psf[y, x] * psf[y, x];

            return squared;
        }

        private static double ShrinkAbsolute(double value, double lambda)
        {
            value = Math.Max(value, 0.0) - lambda;
            return Math.Max(value, 0.0);
        }
    }
}
