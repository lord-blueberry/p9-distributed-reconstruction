using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {
        public static bool Deconvolve(double[,] xImage, double[,] res, double[,]psf, double lambda, double alpha, int maxIteration=100, double[,] dirtyCopy=null)
        {
            var integral = CalcPSf2Integral(psf);
            var resUpdate = new double[res.GetLength(0), res.GetLength(1)];
            var b = ConvolveFFTPadded(res, psf);

            double objective = 0;
            objective += CalcL1Objective(xImage, integral, lambda);
            objective += CalcDataObjective(res);

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
                        //xTmp = ShrinkAbsolute(xTmp, lambda);
                        xTmp = ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;
                        var oImprov = EstimateObjectiveImprovement(res, psf, y, j, xDiff);
                        var lambdaA = lambda * 2 * currentA;
                        oImprov += lambdaA * ElasticNetRegularization(old, alpha);
                        oImprov -= lambdaA * ElasticNetRegularization(xTmp, alpha);

                        //sanity check
                        if (Math.Abs(xDiff) > 1e-6 & oImprov > objective + 1e-2)
                            throw new Exception("Error in CD");
                        
                        if (oImprov > maxImprov)
                        {
                            yPixel = y;
                            xPixel = j;
                            maxImprov = oImprov;
                            xNew = xTmp;
                        }
                    }

                //FitsIO.Write(O2, "greedy_FOO.fits");
                converged = maxImprov < epsilon;
                if(!converged)
                {
                    var xOld = xImage[yPixel, xPixel];
                    xImage[yPixel, xPixel] = xNew;
                    UpdateResiduals(res, resUpdate, psf, yPixel, xPixel, xOld - xNew);
                    UpdateB(b, resUpdate, psf, yPixel, xPixel);
                    objective -= maxImprov;
                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel +"\t"+objective);

                    /*
                    if (iter % 50 == 0)
                    {
                        var conv = IdiotCD.ConvolveFFTPadded(xImage, psf);
                        var resReal = IdiotCD.Subtract(dirtyCopy, conv);
                        var bReal = IdiotCD.ConvolveFFTPadded(resReal, psf);
                        FitsIO.Write(resReal, "greedy_resreal" + iter + ".fits");
                        FitsIO.Write(bReal, "greedy_breal" + iter + ".fits");
                        var objective2 = CalcL1Objective(xImage, integral, lambda);
                        objective2 += CalcDataObjective(resReal);
                        FitsIO.Write(res, "greedy_residuals" + iter + ".fits");
                        FitsIO.Write(b, "greedy_b" + iter + ".fits");
                        FitsIO.Write(IdiotCD.Subtract(b, bReal), "greedy_breal_diff" + iter + ".fits");
                        FitsIO.Write(IdiotCD.Subtract(res, resReal), "greedy_res_diff" + iter + ".fits");
                    }*/

                    iter++;
                }
            }

            /*var conv2 = ConvolveFFTPadded(xImage, psf);
            FitsIO.Write(conv2, "greedy_reconstruction.fits");
            FitsIO.Write(res, "greedy_residuals.fits");
            FitsIO.Write(xImage, "greedy_x.fits");*/
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

        private static void UpdateResiduals(double[,] residual, double[,] resUpdate, double[,] psf, int yPixel, int xPixel, double xDiff)
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

        public static double QueryIntegral(double[,] integral, int yPixel, int xPixel)
        {
            var yPsfHalf = 32;
            var xPsfHalf = 32;

            //possible off by one error for odd psf dimensions
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

        private static double ShrinkPositive(double value, double lambda)
        {
            value = Math.Max(value, 0.0) - lambda;
            return Math.Max(value, 0.0);
        }

        private static double ElasticNetRegularization(double value, double alpha) => (1 - alpha) * 1 / 2 * Math.Abs(value * value) + alpha * Math.Abs(value);

        public static double[,] ConvolveFFTPadded(double[,] img, double[,] psf)
        {
            var yHalf = img.GetLength(0) / 2;
            var xHalf = img.GetLength(1) / 2;
            var img2 = new double[img.GetLength(0) * 2, img.GetLength(1) * 2];
            var psf2 = new double[img.GetLength(0) * 2, img.GetLength(1) * 2];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    img2[i + yHalf, j + xHalf] = img[i, j];
                    psf2[i + yHalf, j + xHalf] = psf[i, j];
                }
            var IMG = FFT.ForwardFFTDebug(img2, 1.0);
            var PSF = FFT.ForwardFFTDebug(psf2, 1.0);
            var CONV = IDG.Multiply(IMG, PSF);
            var conv = FFT.ForwardIFFTDebug(CONV, img2.GetLength(0) * img2.GetLength(1));
            FFT.Shift(conv);

            var convOut = new double[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    convOut[i, j] = conv[i + yHalf, j + xHalf];
                }

            return convOut;
        }
    }
}
