using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class DebugCyclic
    {
        public struct PixelExchange
        {
            public int Rank;
            public int X;
            public int Y;
            public double Value;
        }

        public class Rectangle
        {
            public int Y { get; private set; }
            public int X { get; private set; }

            public int YLength { get; private set; }
            public int XLength { get; private set; }
            
            public Rectangle(int y, int x, int yLen, int xLen)
            {
                Y = y;
                X = x;
                YLength = yLen;
                XLength = xLen;
            }
        }

        public static bool Deconvolve2(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
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

            DeconvolveGreedy2(xImage, resPadded, res, psf, PSFPadded, integral, lambda, alpha, rec, 100);

            var xCummulatedDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];
            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            double objective = double.MaxValue;
            while (!converged & iter < maxIteration)
            {
                var oOld = objective;
                objective = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, rec.Y, rec.X);
                objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                Console.WriteLine("Objective \t" + objective);

                if (oOld < objective)
                    Console.Write("error");
                var RES = FFT.FFTDebug(resPadded, 1.0);
                var B = IDG.Multiply(RES, PSFPadded);
                var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

                var activeSet = new List<Tuple<int, int>>();
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

                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                        }
                    }
                }

                //active set iterations
                Console.WriteLine("--------------------count:" + activeSet.Count + "------------------");
                converged = activeSet.Count == 0;
                bool activeSetConverged = activeSet.Count == 0;
                var innerMax = 40;
                var innerIter = 0;
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    var oTest = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, rec.Y, rec.X);
                    oTest += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);

                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    foreach (var pixel in activeSet)
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var xOld = xImage[yLocal, xLocal];
                        var currentB = CyclicCD2.CalculateB(resPadded, res, psf, y, x);

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
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                        }
                        else if (xTmp == 0.0)
                        {
                            // zero, remove from active set
                            activeSetConverged = false;
                            xImage[yLocal, xLocal] = 0.0;
                            xCummulatedDiff[yLocal, xLocal] += xOld;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            delete.Add(pixel);
                            //Console.WriteLine("drop pixel \t" + xTmp + "\t" + y + "\t" + x);
                        }
                    }
                    innerIter++;
                }

                /*
                foreach (var pixel in delete)
                    activeSet.Remove(pixel);

                //exchange with other nodes
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
                        GreedyCD.UpdateResiduals2(resPadded, res, psf, p.Y, p.X, p.Value, yPsfHalf, xPsfHalf);*/

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

        public static bool DeconvolveGreedy2(double[,] xImage, double[,] resPadded, double[,] res, double[,] psf, Complex[,] PSFPadded, double[,] integral, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

            double objective = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, rec.Y, rec.X);
            objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);

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

                //exchange max
                
                converged = Math.Abs(xMax) < epsilon;
                if (!converged)
                {
                    var yLocal2 = yPixel - rec.Y;
                    var xLocal2 = xPixel - rec.X;
                    var xOld = xImage[yLocal2, xLocal2];
                    xImage[yLocal2, xLocal2] = xNew;

                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yLocal2 + "\t" + xLocal2);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    iter++;
                }
            }

            return converged;
        }



        #region should not be here
        public static bool DeconvolveGreedy3(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
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

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));

            double objective = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, rec.Y, rec.X);
            objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);

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

                //exchange max

                converged = Math.Abs(xMax) < epsilon;
                if (!converged)
                {
                    var yLocal2 = yPixel - rec.Y;
                    var xLocal2 = xPixel - rec.X;
                    var xOld = xImage[yLocal2, xLocal2];
                    xImage[yLocal2, xLocal2] = xNew;

                    FitsIO.Write(GreedyCD2.RemovePadding(b, psf), "b_greedyTrue.fits");
                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yLocal2 + "\t" + xLocal2);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    iter++;
                }
            }

            return converged;
        }
        #endregion
    }
}
