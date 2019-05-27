using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class CyclicCD2
    {
        public static bool DeconvolveCyclic(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            //DeconvolveGreedy(xImage, res, psf, lambda, alpha, 100);

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

            var XTMPBefore = new double[res.GetLength(0), res.GetLength(1)];
            var XTMP = new double[res.GetLength(0), res.GetLength(1)];
            var OIMPROV = new double[res.GetLength(0), res.GetLength(1)];

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                double objective = 0;
                objective += GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
                objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                Console.WriteLine("Objective \t" + objective);

                Console.WriteLine("--------------------adding to active set------------------");
                var activeSet = new List<Tuple<int, int>>();
                //add to active set
                for (int y = 0; y < res.GetLength(0); y++)
                {
                    for (int x = 0; x < res.GetLength(1); x++)
                    {
                        FitsIO.Write(b, "current_b.fits");
                        FitsIO.Write(resPadded, "currentRes.fits");

                        var currentA = GreedyCD.QueryIntegral(integral, y, x);
                        //var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[y, x];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        var xTmpBefore = xTmp;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = old - xTmp;

                        
                        var oImprov = GreedyCD.EstimateObjectiveImprovement2(resPadded, res, psf, y, x, old - xTmp);
                        var lambdaA = lambda * 2 * currentA;
                        var oImprovEl = lambdaA * GreedyCD.ElasticNetRegularization(old, alpha);
                        oImprovEl -= lambdaA * GreedyCD.ElasticNetRegularization(xTmp, alpha);

                        XTMPBefore[y, x] = xTmpBefore;
                        XTMP[y, x] = xTmp;
                        OIMPROV[y, x] = oImprov;

                        if (Math.Abs(xDiff) > epsilon)
                        //if(x == 0 & y == 0)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                            xImage[y, x] = xTmp;

                            xCummulatedDiff[y, x] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);

                            var oTmp = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha) + GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                            if (oTmp > objective)
                                Console.Write("E.");
                            else
                                Console.Write("");
                            objective = oTmp;

                        }
                    }
                }

                /*for (int y = 0; y < res.GetLength(0); y++)
                {
                    for (int x = 0; x < res.GetLength(1); x++)
                    {
                        var currentA = GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[y, x];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        var xTmpBefore = xTmp;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = old - xTmp;

                        var oImprov = GreedyCD.EstimateObjectiveImprovement(res, psf, y, x, old - xTmp);
                        var lambdaA = lambda * 2 * currentA;
                        var oImprovL1 = lambdaA * GreedyCD.ElasticNetRegularization(old, alpha);
                        oImprovL1 -= lambdaA * GreedyCD.ElasticNetRegularization(xTmp, alpha);

                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSet.Add(new Tuple<int, int>(y, x));
                            xImage[y, x] = xTmp;


                            xCummulatedDiff[y, x] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);

                            var oTmp = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha) + GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                            if (oTmp > objective)
                                Console.Write("E.");
                            else
                                Console.Write("");
                            objective = oTmp;
                        }
                    }
                        
                }*/

                FitsIO.Write(XTMPBefore, "xTmpBefore.fits");
                FitsIO.Write(XTMP, "xTmp.fits");
                FitsIO.Write(OIMPROV, "oImprov.fits");

                objective = GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
                objective += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                Console.WriteLine("Objective test \t" + objective);

                Console.WriteLine("--------------------count:" + activeSet.Count + "------------------");



                //active set iterations
                converged = activeSet.Count == 0;
                bool activeSetConverged = activeSet.Count == 0;
                var innerMax = 2000;
                var innerIter = 0;
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    var objectiveNew = 0.0;
                    objectiveNew = GreedyCD.CalcL1Objective(xImage, integral, lambda);
                    objectiveNew += GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                    Console.WriteLine("Objective \t" + objectiveNew + "\t valid "+ (objectiveNew <= objective));
                    objective = objectiveNew;

                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    foreach (var pixel in activeSet.ToArray())
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var xOld = xImage[y, x];
                        var currentB = b[y + yPsfHalf, x + xPsfHalf];

                        //calculate minimum of parabola, eg -2b/a
                        var xTmp = xOld + currentB / GreedyCD.QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1)); ;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));
                        var xDiff = xOld - xTmp;

                        if (Math.Abs(xDiff) > epsilon)
                        {
                            activeSetConverged = false;
                            //Console.WriteLine(Math.Abs(xOld - xTmp) + "\t" + y + "\t" + x);
                            xImage[y, x] = xTmp;
                            xCummulatedDiff[y, x] += xDiff;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);
                            innerIter++;

                            var oTmp = GreedyCD.CalcL1Objective(xImage, integral, lambda) + GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                            if (oTmp > objective)
                                Console.Write("");
                            else
                                Console.Write("");
                            objective = oTmp;
                        }
                        else if (xTmp == 0.0)
                        {
                            //approximately zero, remove from active set
                            activeSetConverged = false;
                            xImage[y, x] = 0.0;
                            xCummulatedDiff[y, x] += xOld;
                            GreedyCD.UpdateResiduals2(resPadded, res, psf, y, x, xDiff, yPsfHalf, xPsfHalf);
                            RES = FFT.FFTDebug(resPadded, 1.0);
                            B = IDG.Multiply(RES, PSFPadded);
                            b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                            FFT.Shift(b);
                            delete.Add(pixel);
                            //Console.WriteLine("drop pixel \t" + xTmp + "\t" + y + "\t" + x);
                            innerIter++;

                            var oTmp = GreedyCD.CalcL1Objective(xImage, integral, lambda) + GreedyCD.CalcDataObjective(resPadded, res, yPsfHalf, yPsfHalf);
                            if (oTmp > objective)
                                Console.Write("");
                            else
                                Console.Write("");
                            objective = oTmp;
                        }
                    }

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }

                iter++;
            }

            //copy back the residuals
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    res[y, x] = resPadded[y + yPsfHalf, x + xPsfHalf];

            return converged;
        }


        public static bool DeconvolveGreedy(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
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

            double objective = 0;
            objective += GreedyCD.CalcElasticNetObjective(xImage, integral, lambda, alpha);
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
                {
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
                }

                //exchange max
                var xOld = xImage[yPixel, xPixel];
                converged = Math.Abs(xOld - xNew) < epsilon;
                if (!converged)
                {
                    xImage[yPixel, xPixel] = xNew;
                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    GreedyCD.UpdateResiduals2(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    FFT.Shift(b);
                    iter++;
                }
            }

            //copy back the residuals
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    res[y, x] = resPadded[y + yPsfHalf, x + xPsfHalf];

            return converged;
        }
    }
}
