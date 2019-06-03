using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD2
    {
        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf);
            var imageSection = new DebugCyclic.Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));

            //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
            var psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfTmp);
            var PsfCorr = FFT.FFTDebug(psfTmp, 1.0);

            psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            SetPsf(psfTmp, xImage, psf, xImage.GetLength(0) / 2, xImage.GetLength(1) / 2);
            var tmp = FFT.FFTDebug(psfTmp, 1.0);
            var tmp2 = IDG.Multiply(tmp, PsfCorr);

            //cached bUpdate. When the PSF is not masked
            var bUpdateCache = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));

            //masked psf
            var maskedPsf = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            double[,] bUpdateMasked;

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var xMax = 0.0;
                var xNew = 0.0;
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                    {
                        var yLocal = y;
                        var xLocal = x;
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, b.GetLength(0), b.GetLength(1));
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y, x] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;

                        if (Math.Abs(xDiff) > xMax)
                        {
                            yPixel = y;
                            xPixel = x;
                            xMax = Math.Abs(xDiff);
                            xNew = xTmp;
                        }
                    }

                //exchange max
                converged = Math.Abs(xMax) < epsilon;
                if (!converged)
                {
                    var yLocal2 = yPixel;
                    var xLocal2 = xPixel;
                    var xOld = xImage[yLocal2, xLocal2];
                    xImage[yLocal2, xLocal2] = xNew;

                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yLocal2 + "\t" + xLocal2);
                    
                    if (yPixel - yPsfHalf >= 0 & yPixel + yPsfHalf < xImage.GetLength(0) & xPixel - xPsfHalf >= 0 & xPixel + xPsfHalf < xImage.GetLength(0))
                    {
                        UpdateB(b, bUpdateCache, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    else
                    {
                        /*
                        SetPsf(maskedPsf, xImage, psf, yPixel, xPixel);
                        tmp = FFT.FFTDebug(maskedPsf, 1.0);
                        tmp2 = IDG.Multiply(tmp, PsfCorr);
                        bUpdateMasked = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));
                        UpdateB(b, bUpdateMasked, imageSection, yPixel, xPixel, xOld - xNew);*/
                        UpdateB(b, bUpdateCache, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    iter++;
                }
            }

            return converged;
        }

        public static double[,] PadResiduals(double[,] res, double[,] psf)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var resPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    resPadded[y + yPsfHalf, x + xPsfHalf] = res[y, x];

            return resPadded;
        }

        public static Complex[,] PadAndInvertPsf(double[,] psf, int yLength, int xLength)
        {
            var psfPadded = new double[yLength + psf.GetLength(0), xLength + psf.GetLength(1)];
            var yPsfHalf = yLength / 2;
            var xPsfHalf = xLength / 2;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPadded[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfPadded);
            var PSFPadded = FFT.FFTDebug(psfPadded, 1.0);

            return PSFPadded;
        }

        public static double[,] RemovePadding(double[,] img, double[,] psf)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var imgNoPadding = new double[img.GetLength(0) - psf.GetLength(0), img.GetLength(1) - psf.GetLength(1)];
            for (int y = 0; y < imgNoPadding.GetLength(0); y++)
                for (int x = 0; x < imgNoPadding.GetLength(1); x++)
                    imgNoPadding[y, x] = img[y + yPsfHalf, x + xPsfHalf];

            return imgNoPadding;
        }

        public static void SetPsf(double[,] psfPadded, double[,] window, double[,] psf, int yPixel, int xPixel)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < window.GetLength(0) & x >= 0 & x < window.GetLength(1))
                    {
                        psfPadded[i+ yPsfHalf, j + xPsfHalf] = psf[i, j];
                    }
                    else
                    {
                        psfPadded[i + yPsfHalf, j + yPsfHalf] = 0.0;
                    }
                }
        }

        public static void UpdateB(double[,] b, double[,] bUpdate, int yPixel, int xPixel, double xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;
            for (int i = 0; i < bUpdate.GetLength(0); i++)
                for (int j = 0; j < bUpdate.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yBHalf;
                    var x = (xPixel + j) - xBHalf;
                    if (y >= 0 & y < b.GetLength(0) & x >= 0 & x < b.GetLength(1))
                        b[y, x] += bUpdate[i, j] * xDiff;
                }
        }

        public static void UpdateB(double[,] b, double[,] bUpdate, DebugCyclic.Rectangle imageSection, int yPixel, int xPixel, double xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imageSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imageSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imageSection.YLength);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imageSection.XLength);
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

    }
}
