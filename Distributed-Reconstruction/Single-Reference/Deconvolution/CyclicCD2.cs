using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;

namespace Single_Reference.Deconvolution
{
    public class CyclicCD2
    {
        public static bool Deconvolve(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = Common.PSF.CalcScan(psf);
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
            var PSFPaddedCorr = FFT.Forward(psfPadded, 1.0);

            var psfPaddedConv = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPaddedConv[y + psfYOffset + 1, x + psfXOffset + 1] = psf[y, x];
            FFT.Shift(psfPaddedConv);
            var PSFPaddedConv = FFT.Forward(psfPaddedConv, 1.0);

            DeconvolveGreedy(xImage, resPadded, res, psf, PSFPaddedCorr, integral, lambda, alpha, 100);

            var xCummulatedDiff = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            int iter = 0;
            bool converged = false;
            double epsilon = 1e-6;
            while (!converged & iter < maxIteration)
            {
                var RES = FFT.Forward(resPadded, 1.0);
                var B = Common.Fourier2D.Multiply(RES, PSFPaddedCorr);
                var b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));

                var activeSet = new List<Tuple<int, int>>();
                for (int y = 0; y < xImage.GetLength(0); y++)
                {
                    for (int x = 0; x < xImage.GetLength(1); x++)
                    {
                        var currentA = Common.PSF.QueryScan(integral, y, x, xImage.GetLength(0), xImage.GetLength(1));
                        var old = xImage[y, x];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = old - xTmp;

                        if (Math.Abs(xDiff) > epsilon/50.0)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                        }
                    }
                }

                var objective = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, 0,0);
                objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                Console.WriteLine("Objective test \t" + objective);
                Console.WriteLine("--------------------count:" + activeSet.Count + "------------------");

                //active set iterations
                converged = activeSet.Count == 0;
                bool activeSetConverged = activeSet.Count == 0;
                var innerMax = 2000;
                var innerIter = 0;
                Randomize(activeSet);
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    var i = 0;
                    foreach (var pixel in activeSet)
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var xOld = xImage[y, x];
                        var currentB = b[y + yPsfHalf, x + xPsfHalf];

                        var xTmp = xOld + currentB / Common.PSF.QueryScan(integral, y, x, xImage.GetLength(0), xImage.GetLength(1)); ;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = (xOld - xTmp) /50.0;

                        if (Math.Abs(xDiff) > epsilon / 50.0)
                        {
                            activeSetConverged = false;
                            //Console.WriteLine(Math.Abs(xOld - xTmp) + "\t" + y + "\t" + x);
                            xImage[y, x] = xTmp;
                            xCummulatedDiff[y+yPsfHalf, x +xPsfHalf] += xDiff;
                        }
                        else if (xTmp == 0.0)
                        {
                            //approximately zero, remove from active set
                            activeSetConverged = false;
                            xImage[y, x] = 0.0;
                            xCummulatedDiff[y + yPsfHalf, x+xPsfHalf] += xOld;
                            delete.Add(pixel);
                            //Console.WriteLine("drop pixel \t" + xTmp + "\t" + y + "\t" + x);
                        }

                        if(i % 50 == 0)
                        {
                            var Xdiff = FFT.Forward(xCummulatedDiff, 1.0);
                            var RESdiff = Common.Fourier2D.Multiply(Xdiff, PSFPaddedConv);
                            var resDiff = FFT.Backward(RESdiff, (double)(RESdiff.GetLength(0) * RESdiff.GetLength(1)));
                            
                            for (int y2 = 0; y2 < res.GetLength(0); y2++)
                                for (int x2 = 0; x2 < res.GetLength(1); x2++)
                                    resPadded[y2 + yPsfHalf, x2 + xPsfHalf] += resDiff[y2 + yPsfHalf, x2 + xPsfHalf];

                            xCummulatedDiff = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
                            RES = FFT.Forward(resPadded, 1.0);
                            B = Common.Fourier2D.Multiply(RES, PSFPaddedCorr);
                            b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));

                            var objective2 = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, 0,0);
                            objective2 += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                            Console.WriteLine("Objective test \t" + objective2);
                        }
                        i++;
                    }
                    var Xdiff2 = FFT.Forward(xCummulatedDiff, 1.0);
                    var RESdiff2 = Common.Fourier2D.Multiply(Xdiff2, PSFPaddedConv);
                    var resDiff2 = FFT.Backward(RESdiff2, (double)(RESdiff2.GetLength(0) * RESdiff2.GetLength(1)));

                    for (int y2 = 0; y2 < res.GetLength(0); y2++)
                        for (int x2 = 0; x2 < res.GetLength(1); x2++)
                            resPadded[y2 + yPsfHalf, x2 + xPsfHalf] += resDiff2[y2 + yPsfHalf, x2 + xPsfHalf];

                    xCummulatedDiff = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
                    RES = FFT.Forward(resPadded, 1.0);
                    B = Common.Fourier2D.Multiply(RES, PSFPaddedCorr);
                    b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));

                    var objective3 = GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, 0,0);
                    objective3 += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                    Console.WriteLine("Objective done iteration \t" + objective3);

                    //FitsIO.Write(resPadded, "debugResiduals.fits");
                    /*
                    foreach (var pixel in activeSet)
                    {
                        //serial descent
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var xOld = xImage[y, x];
                        var currentB = CalculateB(resPadded, xImage, psf, y, x);

                        //calculate minimum of parabola, eg -2b/a
                        var xTmp = xOld + currentB / Common.PSF.QueryScan(integral, y, x, xImage.GetLength(0), xImage.GetLength(1)); ;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = xOld - xTmp;


                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSetConverged = false;
                            //Console.WriteLine(Math.Abs(xOld - xTmp) + "\t" + y + "\t" + x);
                            xImage[y, x] = xTmp;
                            xCummulatedDiff[y, x] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, xImage, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                        }
                        else if (xTmp == 0.0)
                        {
                            //approximately zero, remove from active set
                            activeSetConverged = false;
                            xImage[y, x] = 0.0;
                            xCummulatedDiff[y, x] += xOld;
                            GreedyCD.UpdateResiduals2(resPadded, xImage, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            delete.Add(pixel);
                            //Console.WriteLine("drop pixel \t" + xTmp + "\t" + y + "\t" + x);
                        }
                    }*/
                    innerIter++;

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }

                RES = FFT.Forward(resPadded, 1.0);
                B = Common.Fourier2D.Multiply(RES, PSFPaddedCorr);
                b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));

                iter++;
            }

            //copy back the residuals
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    res[y, x] = resPadded[y + yPsfHalf, x + xPsfHalf];

            return converged;
        }

        public static bool DeconvolveGreedy(double[,] xImage, double[,] resPadded, double[,] res, double[,] psf, Complex[,] PSFPadded, double[,] integral, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;

            var RES = FFT.Forward(resPadded, 1.0);
            var B = Common.Fourier2D.Multiply(RES, PSFPadded);
            var b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
            
            double objective = 0;
            objective += GreedyCD.CalcElasticNetObjective(xImage, res, integral, lambda, alpha, 0, 0);
            objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
            Console.WriteLine("Objective \t" + objective);

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
                        var currentA = Common.PSF.QueryScan(integral, y, x, res.GetLength(0), res.GetLength(1));
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
                var xOld = xImage[yPixel, xPixel];
                converged = Math.Abs(xOld - xNew) < epsilon;
                if (!converged)
                {
                    xImage[yPixel, xPixel] = xNew;
                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.Forward(resPadded, 1.0);
                    B = Common.Fourier2D.Multiply(RES, PSFPadded);
                    b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
                    iter++;
                }
            }

            return converged;
        }

        private static void Randomize<T>(List<T> arr)
        {
            Random rand = new Random();
            
            for(int i = 0; i < arr.Count; i++)
            {
                int selelct = rand.Next(i, arr.Count);
                T tmp = arr[i];
                arr[i] = arr[selelct];
                arr[selelct] = tmp;
            }
        }
    }
}
