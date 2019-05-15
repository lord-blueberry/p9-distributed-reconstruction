using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {
        public static bool Deconvolve2(double[,] xImage, double[,] res, double[,]b, double[,]psf, double[,] psf2, double lambda, double[,] dirtyCopy, int maxIteration=100)
        {
            double objective = 0;
            objective += CalcL1Objective(xImage, lambda);
            objective += CalcDataObjective(res);

            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                FitsIO.Write(b, "greedyB.fits");
                var yPixel = -1;
                var xPixel = -1;
                var minObjective = objective;
                var xNew = 0.0;
                var totO = 0.0;
                var FO = new double[xImage.GetLength(0), xImage.GetLength(1)];
                var XO = new double[xImage.GetLength(0), xImage.GetLength(1)];
                for (int i = 0; i < b.GetLength(0); i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                    {
                        var old = xImage[i, j];
                        var xTmp = old + b[i, j];
                        xTmp = ShrinkAbsolute(xTmp, lambda);
                        var xDiff = old - xTmp;
                        var oImprov = CalcResImprovement(res, psf, i, j, xDiff);
                        oImprov = oImprov + lambda * Math.Abs(old) - lambda * Math.Abs(xTmp);
                        FO[i, j] = oImprov;
                        XO[i, j] = xTmp;

                        //sanity check
                        if (Math.Abs(xDiff) > 1e-6)
                            if (oImprov <= objective + 1e-6)
                                Console.Write("");
                            else
                                Console.Write("ERROR");

                        if (oImprov < minObjective)
                        {
                            yPixel = i;
                            xPixel = j;
                            minObjective = oImprov;
                            xNew = xTmp;
                        }
                        
                    }

                FitsIO.Write(FO, "greedyFO.fits");
                FitsIO.Write(XO, "greedyXO.fits");
                converged = yPixel == -1;
                if(!converged)
                {
                    var xOld = xImage[yPixel, xPixel];
                    xImage[yPixel, xPixel] = xNew;

                    FitsIO.Write(b, "greedyBBeforeUpdate.fits");
                    FitsIO.Write(psf2, "psf2.fits");
                    UpdateResidual(res, psf, yPixel, xPixel, xOld - xNew);
                    FitsIO.Write(res, "greedyResUpdated.fits");
                    UpdateB2(b, psf2, yPixel, xPixel, xOld - xNew);
                    FitsIO.Write(b, "greedyBUpdated.fits");

                    var conv = IdiotCD.ConvolveFFT(xImage, psf);
                    var resReal = IdiotCD.Subtract(dirtyCopy, conv);
                    FitsIO.Write(resReal, "resReal.fits");
                    FitsIO.Write(IdiotCD.Subtract(res, resReal), "resdiff.fits");
                    var bReal = IdiotCD.ConvolveFFT(resReal, psf);
                    FitsIO.Write(IdiotCD.Subtract(b, bReal), "bDiff.fits");
                    FitsIO.Write(bReal, "bReal.fits");

                    var objective2 = CalcL1Objective(xImage, lambda);
                    objective2 += CalcDataObjective(res);
                    objective = minObjective;

                    var objective3 = CalcL1Objective(xImage, lambda);
                    objective3 += CalcDataObjective(resReal);


                    Console.WriteLine(Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel);
                    iter++;
                    FitsIO.Write(xImage, "greedyx.fits");
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

                    var newRes = residual[y, x] + psf[yPsf, xPsf] * xDiff;

                    resOld += (residual[y, x] * residual[y, x]);
                    totalDiff += (newRes * newRes);
                }
            }

            //return resOld - totalDiff;
            return totalDiff;
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
