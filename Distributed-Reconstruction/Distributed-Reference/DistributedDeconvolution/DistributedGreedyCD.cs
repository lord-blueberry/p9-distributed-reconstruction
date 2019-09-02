using System;
using System.Collections.Generic;
using System.Text;
using MPI;
using Single_Reference.IDGSequential;
using static Distributed_Reference.Communication;
using Single_Reference;

namespace Distributed_Reference.DistributedDeconvolution
{
    class DistributedGreedyCD
    { 
        public static bool DeconvolvePath(Intracommunicator comm, Common.Rectangle imgSection, Common.Rectangle totalSize, float[,] xImage, float[,] b, float[,] psf, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 1e-4)
        {
            //TODO: Debug this
            var aMap = Common.PSF.CalcAMap(psf, totalSize, imgSection);
            
            var converged = false;
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                var xMaxAbsDiff = 0.0f;
                var xMaxDiff = 0.0f;
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                    {
                        var xDiff = b[y, x] / aMap[y, x];

                        if (Math.Abs(xDiff) > xMaxAbsDiff)
                        {
                            xMaxAbsDiff = Math.Abs(xDiff);
                            xMaxDiff = xDiff;
                        }
                    }
                    
                var lambdaMax = 1 / alpha * xMaxDiff;
                lambdaMax = comm.Allreduce(lambdaMax, (x, y) => x+y);
                if (lambdaMax / lambdaFactor > lambdaMin)
                {
                    Console.WriteLine("-----------------------------GreedyCD2 with lambda " + lambdaMax / lambdaFactor + "------------------------");
                    converged = Deconvolve(comm, imgSection, totalSize, xImage, aMap, b, psf, lambdaMax / lambdaFactor, alpha, maxIteration, epsilon);
                }
                else
                {
                    Console.WriteLine("-----------------------------GreedyCD2 with lambda " + lambdaMin + "------------------------");
                    converged = Deconvolve(comm, imgSection, totalSize, xImage, aMap, b, psf, lambdaMin, alpha, maxIteration, epsilon);
                    if (converged)
                        break;
                }
            }
            return converged;
        }

        public static bool Deconvolve(Intracommunicator comm, Common.Rectangle imgSection, Common.Rectangle totalSize, float[,] xImage, float[,] aMap, float[,] b, float[,] psf, float lambda, float alpha, int maxIteration = 1000, float epsilon = 1e-4)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;

            //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
            var psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfTmp);
            var PsfCorr = FFT.Forward(psfTmp, 1.0);

            psfTmp = new double[psf.GetLength(0) + psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            SetPsf(psfTmp, totalSize, psf, totalSize.YEnd / 2, totalSize.XEnd / 2);
            var tmp = FFT.Forward(psfTmp, 1.0);
            var tmp2 = Common.Fourier2D.Multiply(tmp, PsfCorr);

            //cached bUpdate. When the PSF is not masked
            var bUpdateCache = FFT.Backward(tmp2, (double)(tmp2.GetLength(0) * tmp2.GetLength(1)));

            //masked psf update. When the psf is masked by the image borders
            var maskedPsf = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];

            int iter = 0;
            bool converged = false;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var xMax = 0.0;
                var xNew = 0.0;
                for (int y = imgSection.Y; y < imgSection.YEnd; y++)
                    for (int x = imgSection.X; x < imgSection.XEnd; x++)
                    {
                        var yLocal = y - imgSection.Y;
                        var xLocal = x - imgSection.X;
                        var currentA = aMap[yLocal, xLocal];
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[yLocal, xLocal] / currentA;
                        xTmp = Common.ShrinkElasticNet(xTmp, lambda, alpha);

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
                var yLocal2 = yPixel - imgSection.Y;
                var xLocal2 = xPixel - imgSection.X;
                Candidate candidate = null;
                var xOld = 0.0;
                if (xMax > 0.0)
                {
                    xOld = xImage[yLocal2, xLocal2];
                    candidate = new Candidate(xMax, xOld - xNew, yPixel, xPixel);
                }
                else
                {
                    candidate = new Candidate(0.0, 0, -1, -1);
                }

                var maxCandidate = comm.Allreduce(candidate, (aC, bC) => aC.XDiffMax > bC.XDiffMax ? aC : bC);
                converged = Math.Abs(maxCandidate.XDiffMax) < epsilon;
                if (!converged)
                {
                    if (maxCandidate.YPixel == yPixel && maxCandidate.XPixel == xPixel)
                        xImage[yLocal2, xLocal2] = xNew;

                    if(comm.Rank == 0)
                        Console.WriteLine(iter + "\t" + maxCandidate.XDiff + "\t" + maxCandidate.YPixel + "\t" + maxCandidate.XPixel);

                    if (maxCandidate.YPixel - yPsfHalf >= 0 & maxCandidate.YPixel + yPsfHalf < totalSize.YEnd & maxCandidate.XPixel - xPsfHalf >= 0 & maxCandidate.XPixel + xPsfHalf < totalSize.XEnd)
                    {
                        UpdateB(b, bUpdateCache, imgSection, maxCandidate.YPixel, maxCandidate.XPixel, maxCandidate.XDiff);
                    }
                    else
                    {
                        /*
                        SetPsf(maskedPsf, totalSize, psf, maxCandidate.YPixel, maxCandidate.XPixel);
                        tmp = FFT.FFTDebug(maskedPsf, 1.0);
                        tmp2 = Common.Fourier2D.Multiply(tmp, PsfCorr);
                        bUpdateMasked = FFT.IFFTDebug(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));
                        UpdateB(b, bUpdateMasked, imgSection, maxCandidate.YPixel, maxCandidate.XPixel, maxCandidate.XDiff);*/
                        UpdateB(b, bUpdateCache, imgSection, maxCandidate.YPixel, maxCandidate.XPixel, maxCandidate.XDiff);
                    }
                    
                    iter++;
                }
            }

            return converged;
        }

        private static void SetPsf(double[,] psfPadded, Common.Rectangle window, double[,] psf, int yPixel, int xPixel)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < window.YEnd & x >= 0 & x < window.XEnd)
                    {
                        psfPadded[i + yPsfHalf, j + xPsfHalf] = psf[i, j];
                    }
                    else
                    {
                        psfPadded[i + yPsfHalf, j + yPsfHalf] = 0.0;
                    }
                }
        }

        private static void UpdateB(float[,] b, float[,] bUpdate, Common.Rectangle imgSection, int yPixel, int xPixel, float xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imgSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imgSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imgSection.YEnd);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imgSection.XEnd);
            for (int i = yBMin; i < yBMax; i++)
                for(int j = xBMin; j < xBMax; j++)
                {
                    var yLocal = i - imgSection.Y;
                    var xLocal = j - imgSection.X;
                    var yBUpdate = i + yBHalf - yPixel;
                    var xBUpdate = j + xBHalf - xPixel;
                    b[yLocal, xLocal] += bUpdate[yBUpdate, xBUpdate] * xDiff;
                }
        }
    }
}
