using System;
using System.Collections.Generic;
using System.Text;
using MPI;
using Single_Reference.Deconvolution;
using Single_Reference.IDGSequential;

namespace Distributed_Reference
{
    class DGreedyCD2
    {
        public static bool Deconvolve(Intracommunicator comm, Communication.Rectangle imageSection, Communication.Rectangle totalSize, double[,] xImage, double[,] b, double[,] psf, double lambda, double alpha, int maxIteration = 1000)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf);

            //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
            var psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfTmp);
            var PsfCorr = FFT.FFTDebug(psfTmp, 1.0);

            psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            SetPsf(psfTmp, totalSize, psf, totalSize.YLength / 2, totalSize.XLength / 2);
            var tmp = FFT.FFTDebug(psfTmp, 1.0);
            var tmp2 = IDG.Multiply(tmp, PsfCorr);

            //cached bUpdate. When the PSF is not masked
            var bUpdateCache = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));

            //masked psf update. When the psf is masked by the image borders
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
                for (int y = imageSection.Y; y < imageSection.YLength; y++)
                    for (int x = imageSection.X; x < imageSection.XLength; x++)
                    {
                        var yLocal = y - imageSection.Y;
                        var xLocal = x - imageSection.X;
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, totalSize.YLength, totalSize.XLength);
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
                var yLocal2 = yPixel - imageSection.Y;
                var xLocal2 = xPixel - imageSection.X;
                Communication.Candidate candidate = null;
                var xOld = 0.0;
                if (xMax > 0.0)
                {
                    xOld = xImage[yLocal2, xLocal2];
                    candidate = new Communication.Candidate(xMax, xOld - xNew, yPixel, xPixel);
                }
                else
                {
                    candidate = new Communication.Candidate(0.0, 0, -1, -1);
                }
                var maxCandidate = comm.Allreduce(candidate, (aC, bC) => aC.XDiffMax > bC.XDiffMax ? aC : bC);
                converged = Math.Abs(candidate.XDiffMax) < epsilon;
                if (!converged)
                {
                    if (maxCandidate.YPixel == yPixel && maxCandidate.XPixel == xPixel)
                        xImage[yLocal2, xLocal2] = xNew;

                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yLocal2 + "\t" + xLocal2);

                    if (yPixel - yPsfHalf >= 0 & yPixel + yPsfHalf < totalSize.YLength & xPixel - xPsfHalf >= 0 & xPixel + xPsfHalf < totalSize.XLength)
                    {
                        UpdateB(b, bUpdateCache, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    else
                    {
                        SetPsf(maskedPsf, totalSize, psf, yPixel, xPixel);
                        tmp = FFT.FFTDebug(maskedPsf, 1.0);
                        tmp2 = IDG.Multiply(tmp, PsfCorr);
                        bUpdateMasked = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));
                        UpdateB(b, bUpdateMasked, imageSection, yPixel, xPixel, xOld - xNew);
                    }
                    iter++;
                }
            }

            return converged;
        }

        public static void SetPsf(double[,] psfPadded, Communication.Rectangle window, double[,] psf, int yPixel, int xPixel)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < window.YLength & x >= 0 & x < window.XLength)
                    {
                        psfPadded[i + yPsfHalf, j + xPsfHalf] = psf[i, j];
                    }
                    else
                    {
                        psfPadded[i + yPsfHalf, j + yPsfHalf] = 0.0;
                    }
                }
        }

        public static void UpdateB(double[,] b, double[,] bUpdate, Communication.Rectangle imageSection, int yPixel, int xPixel, double xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imageSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imageSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imageSection.YLength);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imageSection.XLength);
            for (int i = yBMin; i < yBMax; i++)
                for(int j = xBMin; j < xBMax; j++)
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
