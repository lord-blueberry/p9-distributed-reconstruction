using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {
        public static bool Deconvolve2(double[,] xImage, double[,] res, double[,]b, double[,]psf, double[,] psf2, double lambda, int maxIteration=100)
        {
            double objective = 0;
            objective += CalcL1Objective(xImage, lambda);
            objective += CalcDataObjective(res);

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var maxObjective = 0.0;
                var xNew = 0.0;
                var totO = 0.0;
                for (int i = 0; i < b.GetLength(0); i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                    {
                        if (i == 24 & j == 31)
                            Console.Write("");
                        var old = xImage[i, j];
                        var xTmp = old + b[i, j];
                        xTmp = ShrinkAbsolute(xTmp, lambda);
                        var xDiff = old - xTmp;
                        if (Math.Abs(xDiff) > epsilon)
                        {
                            var oImprov = CalcResImprovement(res, psf, i, j, xDiff);
                            var oImprovN0 = CalcResImprovement(res, psf, i, j, xDiff-0.5);
                            var oImprovN1 = CalcResImprovement(res, psf, i, j, xDiff + 0.5);
                            if (oImprov < 0.0)
                                Console.Write("");
                            oImprov = oImprov + lambda * Math.Abs(old) - lambda * Math.Abs(xTmp);
                            if (oImprov > maxObjective)
                            {
                                yPixel = i;
                                xPixel = j;
                                maxObjective = oImprov;
                                xNew = xTmp;
                            }
                        }
                    }

                converged = yPixel == -1;
                if(!converged)
                {
                    var xOld = xImage[yPixel, xPixel];
                    xImage[yPixel, xPixel] = xNew;
                    UpdateResidual(res, psf, yPixel, xPixel, xOld - xNew);
                    UpdateB2(b, psf2, yPixel, xPixel, xOld - xNew);
                    var objective2 = CalcL1Objective(xImage, lambda);
                    objective2 += CalcDataObjective(res);
                    objective -= maxObjective;

                    Console.WriteLine(Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    iter++;
                }
            }

            return converged;
        }

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

                    var diff = residual[y, x] + psf[yPsf, xPsf] * xDiff;

                    resOld += (residual[y, x] * residual[y, x]);
                    totalDiff += (diff * diff);
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

        private static double CalcDataObjective(double[,] res)
        {
            double objective = 0;
            for (int i = 0; i < res.GetLength(0); i++)
                for (int j = 0; j < res.GetLength(1); j++)
                    objective += res[i, j] * res[i, j];
            return objective;
        }

        private static double CalcL1Objective(double[,] xImage, double lambda)
        {
            double objective = 0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    objective += Math.Abs(xImage[i, j]) * lambda;
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
            var yPsfHalf = (int)Math.Ceiling(psf2.GetLength(0) / 2.0);
            var xPsfHalf = (int)Math.Ceiling(psf2.GetLength(1) / 2.0);
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

        public static double CalcPSFSquared(double[,] psf)
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
