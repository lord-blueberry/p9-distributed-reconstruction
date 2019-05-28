using System;
using System.Collections.Generic;
using System.Text;

using MPI;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using System.Numerics;

namespace Distributed_Reference
{
    class Greedy
    {
        [Serializable]
        public struct PixelExchange
        {
            public int Rank;
            public int X;
            public int Y;
            public double Value;
        }

        public static bool Deconvolve2(Intracommunicator comm, double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, DGreedyCD.Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf);
            var resPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    resPadded[y + yPsfHalf, x + xPsfHalf] = res[y, x];

            //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
            var psfPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            var psfYOffset = res.GetLength(0) / 2;
            var psfXOffset = res.GetLength(1) / 2;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPadded[y + psfYOffset + 1, x + psfXOffset + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfPadded);
            var PSFPadded = FFT.FFTDebug(psfPadded, 1.0);

            DeconvolveGreedy2(comm, xImage, resPadded, res, psf, PSFPadded, integral, lambda, alpha, rec, 200);

            var xCummulatedDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];
            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                double objective = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
                objective = comm.Allreduce(objective, (aC, bC) => aC + bC);
                objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                if (comm.Rank == 0)
                {
                    Console.WriteLine("Objective \t" + objective);
                }

                var RES = FFT.FFTDebug(resPadded, 1.0);
                var B = IDG.Multiply(RES, PSFPadded);
                var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

                Console.WriteLine("--------------------adding to active set------------------");
                var activeSet = new List<Tuple<int, int>>();
                for (int y = 0; y < xImage.GetLength(0); y++)
                {
                    for (int x = 0; x < xImage.GetLength(1); x++)
                    {
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = old - xTmp;

                        if (Math.Abs(xDiff) > 1e-8)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                            /*xImage[y, x] = xTmp;
                            xCummulatedDiff[y, x] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));*/
                        }
                    }
                }

                //objective = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
                //objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                //Console.WriteLine("Objective test \t" + objective);
                Console.WriteLine("--------------------count:" + activeSet.Count + "------------------");

                //active set iterations
                converged = activeSet.Count == 0;
                bool activeSetConverged = activeSet.Count == 0;
                var innerMax = 20;
                var innerIter = 0;
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    foreach (var pixel in activeSet.ToArray())
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var xOld = xImage[yLocal, xLocal];
                        var currentB = CalculateB(resPadded, xImage, psf, y, x);

                        //calculate minimum of parabola, eg -2b/a
                        var xTmp = xOld + currentB / GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = xOld - xTmp;

                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSetConverged = false;
                            //Console.WriteLine(Math.Abs(xOld - xTmp) + "\t" + y + "\t" + x);
                            xImage[yLocal, xLocal] = xTmp;
                            xCummulatedDiff[yLocal, xLocal] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, xImage, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                        }
                        else if (xTmp == 0.0)
                        {
                            // zero, remove from active set
                            activeSetConverged = false;
                            xImage[yLocal, xLocal] = 0.0;
                            xCummulatedDiff[yLocal, xLocal] += xOld;
                            GreedyCD.UpdateResiduals2(resPadded, xImage, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            delete.Add(pixel);
                            //Console.WriteLine("drop pixel \t" + xTmp + "\t" + y + "\t" + x);
                        }
                    }
                    innerIter++;

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }

                //exchange with other nodes
                if (iter % 2 == 0)
                {
                    var allXDiff = new List<PixelExchange>();
                    for (int y = 0; y < xCummulatedDiff.GetLength(0); y++)
                        for (int x = 0; x < xCummulatedDiff.GetLength(1); x++)
                        {
                            if (xCummulatedDiff[y, x] > 0.0)
                            {
                                var p = new PixelExchange();
                                p.Rank = comm.Rank;
                                p.Y = rec.Y + y;
                                p.X = rec.X + x;
                                p.Value = xCummulatedDiff[y, x];
                                allXDiff.Add(p);
                                xCummulatedDiff[y, x] = 0.0;
                            }
                        }

                    var allNonZeros = comm.Allreduce(allXDiff, (aC, bC) =>
                    {
                        aC.AddRange(bC);
                        return aC;
                    });

                    foreach (var p in allXDiff)
                        if (p.Rank != comm.Rank)
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, p.Y, p.X, p.Value, yPsfHalf, xPsfHalf);
                }

                RES = FFT.FFTDebug(resPadded, 1.0);
                B = IDG.Multiply(RES, PSFPadded);
                b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

                iter++;
            }

            //copy back the residuals
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    res[y, x] = resPadded[y + yPsfHalf, x + xPsfHalf];

            return converged;
        }

        public static bool DeconvolveGreedy2(Intracommunicator comm, double[,] xImage, double[,] resPadded, double[,] res, double[,] psf, Complex[,] PSFPadded, double[,] integral, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

            double objective = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
            objective = comm.Allreduce(objective, (aC, bC) => aC + bC);
            objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
            if (comm.Rank == 0)
            {
                Console.WriteLine("Objective \t" + objective);
            }

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var xMax = 0.0;
                var xNew = 0.0;
                for (int y = 0; y < res.GetLength(0); y++)
                    for (int x = 0; x < res.GetLength(1); x++)
                    {
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[y, x];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
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
                var yLocal2 = yPixel - rec.Y;
                var xLocal2 = xPixel - rec.X;
                DGreedyCD.Candidate candidate = null;
                var xOld = 0.0;
                if (xMax > 0.0)
                {
                    xOld = xImage[yLocal2, xLocal2];
                    candidate = new DGreedyCD.Candidate(xMax, xOld - xNew, yPixel, xPixel);
                }
                else
                {
                    candidate = new DGreedyCD.Candidate(0.0, 0, -1, -1);
                }

                var maxCandidate = comm.Allreduce(candidate, (aC, bC) => aC.OImprov > bC.OImprov ? aC : bC);
                converged = Math.Abs(maxCandidate.OImprov) < epsilon;
                if (!converged)
                {
                    if (maxCandidate.YPixel == yPixel && maxCandidate.XPixel == xPixel)
                        xImage[yLocal2, xLocal2] = xNew;

                    if (comm.Rank == 0)
                        Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, maxCandidate.YPixel, maxCandidate.XPixel, maxCandidate.XDiff, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    iter++;
                }
            }

            return converged;
        }

        public static double CalculateB(double[,] resPadded, double[,] xImage, double[,] psf, int yPixel, int xPixel)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            int yOffset = yPixel - yPsfHalf;
            int xOffset = xPixel - xPsfHalf;

            var b = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var ySrc = yOffset + i;
                    var xSrc = xOffset + j;
                    if (ySrc >= 0 & ySrc < xImage.GetLength(0) & xSrc >= 0 & xSrc < xImage.GetLength(1))
                        b += resPadded[ySrc + yPsfHalf, xSrc + xPsfHalf] * psf[i, j];
                }


            return b;
        }


        public static bool DeconvolveCyclic(Intracommunicator comm, double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, DGreedyCD.Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            DeconvolveGreedy(comm, xImage, res, psf, lambda, alpha, rec, 100);

            var xCummulatedDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];

            //TODO REMOVE HACK
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf);

            var resPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    resPadded[y + yPsfHalf, x + xPsfHalf] = res[y, x];

            var psfPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            var psfYOffset = res.GetLength(0) / 2;
            var psfXOffset = res.GetLength(1) / 2;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPadded[y + psfYOffset, x + psfXOffset] = psf[y, x];

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var PSFPadded = FFT.FFTDebug(psfPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
            FFT.Shift(b);

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var activeSet = new List<Tuple<int, int>>();
                //add to active set
                for (int y = rec.Y; y < rec.YLength; y++)
                {
                    for (int x = rec.X; x < rec.XLength; x++)
                    {
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;
                        if(Math.Abs(xDiff) > epsilon)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                            xImage[yLocal, xLocal] = xTmp;
                            xCummulatedDiff[yLocal, xLocal] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);
                        }
                    }
                }

                //exchange with other nodes
                if (iter % 2 == 1)
                {
                    var allXDiff = new List<PixelExchange>();
                    for(int y = 0; y < xCummulatedDiff.GetLength(0); y++)
                        for(int x = 0; x < xCummulatedDiff.GetLength(1); x++)
                        {
                            if(xCummulatedDiff[y, x] > 0.0)
                            {
                                var p = new PixelExchange();
                                p.Rank = comm.Rank;
                                p.Y = rec.Y + y;
                                p.X = rec.X + x;
                                p.Value = xCummulatedDiff[y, x];
                                allXDiff.Add(p);
                                xCummulatedDiff[y, x] = 0.0;
                            }
                        }

                    var allNonZeros = comm.Allreduce(allXDiff, (aC, bC) =>
                    { 
                        aC.AddRange(bC);
                        return aC;
                    });

                    foreach(var p in allXDiff)
                        if (p.Rank != comm.Rank)
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, p.Y, p.X, p.Value, yPsfHalf, xPsfHalf);

                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    FFT.Shift(b);
                }

                //active set iterations
                var totalActiveSetCount = comm.Allreduce(activeSet.Count, (aC, bC) => aC + bC);
                converged = totalActiveSetCount == 0;
                bool activeSetConverged = activeSet.Count == 0;
                var innerMax = 2000;
                var innerIter = 0;
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    foreach (var pixel in activeSet.ToArray())
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var xOld = xImage[yLocal, xLocal];
                        var currentB = b[y+yPsfHalf, x+xPsfHalf];

                        //calculate minimum of parabola, eg -2b/a
                        var xTmp = xOld + currentB / GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1)); 
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = xOld - xTmp;

                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSetConverged = false;
                            xImage[yLocal, xLocal] = xTmp;
                            xCummulatedDiff[yLocal, xLocal] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);
                            innerIter++;
                        }
                        else if (xTmp < epsilon)
                        {
                            //approximately zero, remove from active set
                            activeSetConverged = false;
                            xImage[yLocal, xLocal] = 0.0;
                            xCummulatedDiff[yLocal, xLocal] += xOld;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);
                            delete.Add(pixel);
                            innerIter++;
                        }
                    }

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }

                iter++;
            }

            return converged;
        }


        public static bool DeconvolveGreedy(Intracommunicator comm, double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, DGreedyCD.Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf);

            var resPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    resPadded[y + yPsfHalf, x + xPsfHalf] = res[y, x];

            var psfPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            var psfYOffset = res.GetLength(0) / 2;
            var psfXOffset = res.GetLength(1) / 2;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPadded[y + psfYOffset, x + psfXOffset] = psf[y, x];

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var PSFPadded = FFT.FFTDebug(psfPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
            FFT.Shift(b);

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var xMax = 0.0;
                var xNew = 0.0;
                for (int y = rec.Y; y < rec.YLength; y++)
                {
                    for (int x = rec.X; x < rec.XLength; x++)
                    {
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
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
                }

                //exchange max
                var yLocal2 = yPixel - rec.Y;
                var xLocal2 = xPixel - rec.X;
                DGreedyCD.Candidate candidate = null;
                var xOld = 0.0;
                if (xMax > 0.0)
                {
                    xOld = xImage[yLocal2, xLocal2];
                    candidate = new DGreedyCD.Candidate(xMax, xOld - xNew, yPixel, xPixel);
                }
                else
                {
                    candidate = new DGreedyCD.Candidate(0.0, 0, -1, -1);
                }

                var maxCandidate = comm.Allreduce(candidate, (aC, bC) => aC.OImprov > bC.OImprov ? aC : bC);
                converged = Math.Abs(maxCandidate.OImprov) < epsilon;
                if (!converged)
                {
                    if (maxCandidate.YPixel == yPixel && maxCandidate.XPixel == xPixel)
                        xImage[yLocal2, xLocal2] = xNew;

                    if (comm.Rank == 0)
                        Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, maxCandidate.YPixel, maxCandidate.XPixel, maxCandidate.XDiff, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    FFT.Shift(b);
                    iter++;
                }
            }

            return converged;
        }

    }
}
