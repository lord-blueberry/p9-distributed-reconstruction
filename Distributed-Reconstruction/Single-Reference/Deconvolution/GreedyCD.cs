﻿using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {
        public static bool Deconvolve2(double[,] xImage, double[,] res, double[,]b, double[,]psf, double lambda, double[,] dirtyCopy, int maxIteration=100)
        {
            var integral = CalcPSf2Integral(psf);
            var resUpdate = new double[res.GetLength(0), res.GetLength(1)];

            double objective = 0;
            objective += CalcL1Objective(xImage, integral, lambda);
            objective += CalcDataObjective(res);

            var objective2 = 0.0;
            var O2 = new double[b.GetLength(0), b.GetLength(1)];

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var maxImprov = 0.0;
                var xNew = 0.0;
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int j = 0; j < b.GetLength(1); j++)
                    {
                        var currentA = QueryIntegral(integral, y, j);
                        var old = xImage[y, j];
                        var xTmp = old + b[y, j] / currentA;
                        xTmp = ShrinkAbsolute(xTmp, lambda);
                        var xDiff = old - xTmp;
                        var oImprov = EstimateObjectiveImprovement(res, psf, y, j, xDiff);
                        var lambdaA = lambda * 2 * currentA;
                        oImprov += lambdaA * Math.Abs(old) - lambdaA * Math.Abs(xTmp);
                        O2[y, j] = objective - oImprov;

                        //sanity check
                        if (Math.Abs(xDiff) > 1e-6)
                            if (oImprov <= objective + 1e-2)
                                Console.Write("");
                            else
                            {
                                Console.Write("ERROR");
                                throw new Exception("Error in CD");
                            }
                                

                        if (oImprov > maxImprov)
                        {
                            yPixel = y;
                            xPixel = j;
                            maxImprov = oImprov;
                            xNew = xTmp;
                        }
                    }

                FitsIO.Write(O2, "greedy_FOO.fits");
                converged = maxImprov == 0.0;
                if(!converged)
                {
                    var xOld = xImage[yPixel, xPixel];
                    xImage[yPixel, xPixel] = xNew;


                    var conv = IdiotCD.ConvolveFFTPadded(xImage, psf);
                    var resReal = IdiotCD.Subtract(dirtyCopy, conv);
                    var bReal = IdiotCD.ConvolveFFTPadded(resReal, psf);
                    FitsIO.Write(resReal, "greedy_resreal" + iter + ".fits");
                    FitsIO.Write(bReal, "greedy_breal" + iter + ".fits");
                    objective2 = CalcL1Objective(xImage, integral, lambda);
                    objective2 += CalcDataObjective(resReal);


                    UpdateResidualNonCD(res, resUpdate, psf, yPixel, xPixel, xOld - xNew);
                    UpdateB(b, resUpdate, psf, yPixel, xPixel);
                    objective -= maxImprov;
                    Console.WriteLine(Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel +"\t"+objective);


                    FitsIO.Write(res, "greedy_residuals" + iter + ".fits");
                    FitsIO.Write(b, "greedy_b" + iter + ".fits");


                    iter++;
                }
            }

            var conv2 = IdiotCD.ConvolveFFTPadded(xImage, psf);
            FitsIO.Write(conv2, "reconstruction.fits");
            FitsIO.Write(res, "residuals.fits");
            return converged;
        }


        private static double EstimateObjectiveImprovement(double[,] residual, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var totalDiff = 0.0;
            var resOld = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < residual.GetLength(0) &
                        x >= 0 & x < residual.GetLength(1))
                    {
                        var newRes = residual[y, x] + psf[i, j] * xDiff;
                        resOld += (residual[y, x] * residual[y, x]);
                        totalDiff += (newRes * newRes);
                    }
                }
            }

            return resOld - totalDiff;
        }

        private static void UpdateResidualNonCD(double[,] residual, double[,] resUpdate, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < residual.GetLength(0) & x >= 0 & x < residual.GetLength(1))
                    {
                        residual[y, x] += psf[i, j] * xDiff;
                        resUpdate[y, x] = psf[i, j] * xDiff;
                    }
                }
        }

        private static void UpdateB(double[,] b, double[,] resUpdate, double[,] psf, int yPixel, int xPixel)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int y = 0; y < resUpdate.GetLength(0); y++)
                for (int x = 0; x < resUpdate.GetLength(1); x++)
                    if (resUpdate[y, x] != 0.0)
                    {
                        var diff = resUpdate[y, x];
                        resUpdate[y, x] = 0.0;
                        for (int yPsf = 0; yPsf < psf.GetLength(0); yPsf++)
                        {
                            for (int xPsf = 0; xPsf < psf.GetLength(1); xPsf++)
                            {
                                var yConv = (y + yPsf) - yPsfHalf;
                                var xConv = (x + xPsf) - xPsfHalf;
                                if (yConv >= 0 & yConv < b.GetLength(0) & xConv >= 0 & xConv < b.GetLength(1))
                                {
                                    b[yConv, xConv] += psf[yPsf, xPsf] * diff;
                                }
                            }
                        }
                    }
        }



        #region deleteme when sure
        private static double CalcResImprovement(double[,] residual, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var totalDiff = 0.0;
            var resOld = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) % residual.GetLength(0);
                    var x = (xPixel + j) % residual.GetLength(1);
                    var yPsf = (i + yPsfHalf) % psf.GetLength(0);
                    var xPsf = (j + xPsfHalf) % psf.GetLength(1);

                    var newRes = residual[y, x] + psf[yPsf, xPsf] * xDiff;

                    resOld += (residual[y, x] * residual[y, x]);
                    totalDiff += (newRes * newRes);
                }
            }

            return resOld - totalDiff;
        }

        private static void UpdateResidual(double[,] residual, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) % residual.GetLength(0);
                    var x = (xPixel + j) % residual.GetLength(1);
                    var yPsf = (i + yPsfHalf) % psf.GetLength(0);
                    var xPsf = (j + xPsfHalf) % psf.GetLength(1);
                    var diff = psf[yPsf, xPsf] * xDiff;
                    residual[y, x] += psf[yPsf, xPsf] * xDiff;
                }
            }

        }

        #endregion

        public static double[,] CalcPSf2Integral(double[,] psf)
        {
            var integral = new double[psf.GetLength(0), psf.GetLength(1)];
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var iBefore = i > 0 ? integral[i - 1, j] : 0.0;
                    var jBefore = j > 0 ? integral[i, j - 1] : 0.0;
                    var ijBefore = i > 0 & j > 0 ? integral[i - 1, j - 1] : 0.0;
                    var current = psf[i, j] * psf[i, j];
                    integral[i, j] = current + iBefore + jBefore - ijBefore;
                }

            return integral;
        }

        public static double[,] CalcPSfIntegral(double[,] psf)
        {
            var integral = new double[psf.GetLength(0), psf.GetLength(1)];
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var iBefore = i > 0 ? integral[i - 1, j] : 0.0;
                    var jBefore = j > 0 ? integral[i, j - 1] : 0.0;
                    var ijBefore = i > 0 & j > 0 ? integral[i - 1, j - 1] : 0.0;
                    var current = psf[i, j];
                    integral[i, j] = current + iBefore + jBefore - ijBefore;
                }

            return integral;
        }

        public static double QueryIntegral(double[,] integral, int yPixel, int xPixel)
        {
            var yPsfHalf = 32;
            var xPsfHalf = 32;
            var yOverShoot = integral.GetLength(0) * 2 - (yPixel + yPsfHalf) - 1;
            var xOverShoot = integral.GetLength(1) * 2 - (xPixel + xPsfHalf) - 1;

            var yCorrection = yOverShoot % integral.GetLength(0);
            var xCorrection = xOverShoot % integral.GetLength(1);

            if (yCorrection == yOverShoot & xCorrection == xOverShoot)
            {
                return integral[yCorrection, xCorrection];
            }
            else if (yCorrection == yOverShoot | xCorrection == xOverShoot)
            {
                var y = Math.Min(yOverShoot, integral.GetLength(0) - 1);
                var x = Math.Min(xOverShoot, integral.GetLength(1) - 1);
                return integral[y, x] - integral[yCorrection, xCorrection];
            }
            else
            {
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1]
                       - integral[integral.GetLength(0) - 1, xCorrection]
                       - integral[yCorrection, integral.GetLength(1) - 1]
                       + integral[yCorrection, xCorrection];
            }
        }

        private static double CalcDataObjective(double[,] res)
        {
            double objective = 0;
            for (int i = 0; i < res.GetLength(0); i++)
                for (int j = 0; j < res.GetLength(1); j++)
                    objective += res[i, j] * res[i, j];
            return objective;
        }

        private static double CalcL1Objective(double[,] xImage, double[,] aMap, double lambda)
        {
            double objective = 0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    objective += Math.Abs(xImage[i, j]) * lambda * 2 * QueryIntegral(aMap, i, j);
            return objective;
        }


        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf2, double lambda, double alpha, int maxIteration = 100)
        {
            int iter = 0;
            bool converged = false;
            double precision = 1e-12;

            var aa = psf2[psf2.GetLength(0) / 2, psf2.GetLength(1) / 2];
            //var a2 = CDClean.CalcPSFSquared(psf);
            while (!converged & iter < maxIteration)
            {
                var maxB = Double.NegativeInfinity;
                var maxAbsB = Double.NegativeInfinity;
                var yPixel = -1;
                var xPixel = -1;
                for (int i = 0; i < b.GetLength(0); i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                    {
                        var currentB = b[i, j];
                        if (xImage[i, j] > 0.0)
                            currentB = Math.Abs(currentB);

                        if (currentB > maxAbsB)
                        {
                            maxB = b[i, j];
                            maxAbsB = currentB;
                            yPixel = i;
                            xPixel = j;
                        }
                    }

                var xOld = xImage[yPixel, xPixel];
                var xDiff = (maxB / aa);             //calculate minimum of parabola, e.g -b/2a. Simplifications led rise to this line

                Console.WriteLine(xDiff + "\t" + yPixel + "\t" + xPixel);
                var xNew = xOld + xDiff;
                //xNew = Math.Max(Math.Max(0, (xNew * xNew) - lambda) * xNew / (xNew * xNew) , 0);
                //xNew = xNew / (1 + lambda * (1 - alpha));
                xNew = ShrinkAbsolute(xNew, lambda * alpha) / (1 + lambda * (1 - alpha));
                xImage[yPixel, xPixel] = xNew;
                UpdateB2(b, psf2, yPixel, xPixel, xOld - xNew);
                iter++;
                converged = Math.Abs(xOld-xNew) < precision;
            }

            return converged;
        }

        private static void UpdateB2(double[,] b, double[,] psf2, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf2.GetLength(0) / 2;
            var xPsfHalf = psf2.GetLength(1) / 2;
            for (int i = 0; i < psf2.GetLength(0); i++)
            {
                for (int j = 0; j < psf2.GetLength(1); j++)
                {
                    var y = (yPixel + i) % b.GetLength(0);
                    var x = (xPixel + j) % b.GetLength(1);
                    var yPsf = (i + yPsfHalf) % psf2.GetLength(0);
                    var xPsf = (j + xPsfHalf) % psf2.GetLength(1);
                    b[y, x] += psf2[yPsf, xPsf] * xDiff;
                }
            }
        }

        private static double ShrinkAbsolute(double value, double lambda)
        {
            value = Math.Max(value, 0.0) - lambda;
            return Math.Max(value, 0.0);
        }
    }
}
