﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD2
    {
        public static bool DeconvolvePath(double[,] xImage, double[,] b, double[,] psf, double lambdaMin, double lambdaFactor, double alpha, int maxPathIteration = 10,  int maxIteration = 100, double epsilon = 1e-4)
        {
            var integral = GreedyCD.CalcPSf2Integral(psf);
            var converged = false;
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var xMaxAbsDiff = 0.0;
                var xMaxDiff = 0.0;
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                    {
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, b.GetLength(0), b.GetLength(1));
                        var xDiff = b[y, x] / currentA;

                        if (Math.Abs(xDiff) > xMaxAbsDiff)
                        {
                            xMaxAbsDiff = Math.Abs(xDiff);
                            xMaxDiff = xDiff;
                        }
                    }

                var lambdaMax = 1 / alpha * xMaxDiff;
                if (lambdaMax / lambdaMin > lambdaFactor)
                {
                    Console.WriteLine("-----------------------------GreedyCD2 with lambda " + lambdaMax / lambdaFactor + "------------------------");
                    converged = Deconvolve(xImage, b, psf, lambdaMax / lambdaFactor, alpha, maxIteration, epsilon);
                }
                else
                {
                    Console.WriteLine("-----------------------------GreedyCD2 with lambda " + lambdaMin + "------------------------");
                    converged = Deconvolve(xImage, b, psf, lambdaMin, alpha, maxIteration, epsilon);
                    if(converged)
                        break;
                }
            }
            return converged;
        }

        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var aMap = CommonMethods.PSF.CalcAMap(xImage, psf);
            var imageSection = new DebugCyclic.Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));

            //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
            var psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfTmp);
            var PsfCorr = FFT.FFTDebug(psfTmp, 1.0);

            psfTmp = new double[psf.GetLength(0) + psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            CommonMethods.PSF.SetPSFInWindow(psfTmp, xImage, psf, xImage.GetLength(0) / 2, xImage.GetLength(1) / 2);
            var tmp = FFT.FFTDebug(psfTmp, 1.0);
            var tmp2 = IDG.Multiply(tmp, PsfCorr);

            //cached bUpdate. When the PSF is not masked
            var bUpdateCache = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));

            //masked psf
            var maskedPsf = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            double[,] bUpdateMasked;

            int iter = 0;
            bool converged = false;
            var watch = new System.Diagnostics.Stopwatch();
            watch.Start();
            while (!converged & iter < maxIteration)
            {
                /*
                var shrinked = new double[b.GetLength(0), b.GetLength(1)];
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                    {
                        var old = xImage[y, x];
                        var xTmp = old + b[y, x] / aMap[y, x];
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        shrinked[y, x] = Math.Abs(xTmp - old);
                    }*/

                var yPixel = -1;
                var xPixel = -1;
                var xMax = 0.0;
                var xNew = 0.0;
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                    {
                        var yLocal = y;
                        var xLocal = x;
                        var currentA = aMap[y, x];
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

                    Console.WriteLine(iter + "\t" + (xNew - xOld) + "\t" + yLocal2 + "\t" + xLocal2);
                    
                    if (yPixel - yPsfHalf >= 0 & yPixel + yPsfHalf < xImage.GetLength(0) & xPixel - xPsfHalf >= 0 & xPixel + xPsfHalf < xImage.GetLength(0))
                    {
                        UpdateB(b, bUpdateCache, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    else
                    {

                        /*CommonMethods.PSF.SetPSFInWindow(maskedPsf, xImage, psf, yPixel, xPixel);
                        tmp = FFT.FFTDebug(maskedPsf, 1.0);
                        tmp2 = IDG.Multiply(tmp, PsfCorr);
                        bUpdateMasked = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));
                        UpdateB(b, bUpdateMasked, imageSection, yPixel, xPixel, xOld - xNew);*/
                        UpdateB(b, bUpdateCache, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    iter++;
                            
                    if(iter == 1000)
                    {
                        //FitsIO.Write(shrinked, "shrinkedReal.fits");
                        //FitsIO.Write(b, "candidatesGreedy2.fits");
                        //FitsIO.Write(xImage, "xImageGreedy2.fits");
                    }

                }
            }
            watch.Stop();
            Console.WriteLine(watch.Elapsed);
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

        private static void UpdateB(double[,] b, double[,] bUpdate, DebugCyclic.Rectangle imageSection, int yPixel, int xPixel, double xDiff)
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
