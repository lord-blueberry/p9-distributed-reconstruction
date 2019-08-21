using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;

namespace Single_Reference.Deconvolution
{
    public class GreedyCD
    {
        public static double CalcDataObjective(double[,] res)
        {
            double objective = 0;
            for (int i = 0; i < res.GetLength(0); i++)
                for (int j = 0; j < res.GetLength(1); j++)
                    objective += res[i, j] * res[i, j];
            return objective;
        }


        public static double ShrinkPositive(double value, double lambda)
        {
            value = Math.Max(value, 0.0) - lambda;
            return Math.Max(value, 0.0);
        }

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
            var IMG = FFT.Forward(img2, 1.0);
            var PSF = FFT.Forward(psf2, 1.0);
            var CONV = Common.Fourier2D.Multiply(IMG, PSF);
            var conv = FFT.Backward(CONV, (double)(img2.GetLength(0) * img2.GetLength(1)));
            FFT.Shift(conv);

            var convOut = new double[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    convOut[i, j] = conv[i + yHalf, j + xHalf];
                }

            return convOut;
        }

        #region deconvReplacement
        public static bool Deconvolve2(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, int maxIteration = 100, double[,] dirtyCopy = null)
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
                    psfPadded[y + psfYOffset +1, x + psfXOffset+1] = psf[psf.GetLength(0) - y -1, psf.GetLength(1) - x -1];

            FFT.Shift(psfPadded);
            var RES = FFT.Forward(resPadded, 1.0);
            var PSFPadded = FFT.Forward(psfPadded, 1.0);
            var B = Common.Fourier2D.Multiply(RES, PSFPadded);
            var b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
            

            double objective = 0;
            objective += CalcElasticNetObjective(xImage, res, integral, lambda, alpha, 0, 0);
            objective += CalcDataObjective(res);


            Console.WriteLine("Objective \t" + objective);
            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var maxImprov = 0.0;
                var xNew = 0.0;
                for (int y = 0; y < res.GetLength(0); y++)
                    for (int x = 0; x < res.GetLength(1); x++)
                    {
                        var currentA = QueryIntegral2(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[y, x];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;
                        var oImprov = EstimateObjectiveImprovement2(resPadded, res, psf, y, x, xDiff);
                        var lambdaA = lambda * 2 * currentA;
                        oImprov += lambdaA * GreedyCD.ElasticNetRegularization(old, alpha);
                        oImprov -= lambdaA * GreedyCD.ElasticNetRegularization(xTmp, alpha);

                        if (oImprov > maxImprov)
                        {
                            yPixel = y;
                            xPixel = x;
                            maxImprov = oImprov;
                            xNew = xTmp;
                        }
                    }

                var xOld = xImage[yPixel, xPixel];
                converged = maxImprov < epsilon;
                if (!converged)
                {
                    xImage[yPixel, xPixel] = xNew;
                    objective -= maxImprov;
                    Console.WriteLine(iter + "\t" + Math.Abs(xOld - xNew) + "\t" + yPixel + "\t" + xPixel + "\t" + objective);

                    var debug = EstimateObjectiveImprovement2(resPadded, res, psf, yPixel, xPixel, xOld - xNew);
                    UpdateResiduals2(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.Forward(resPadded, 1.0);
                    B = Common.Fourier2D.Multiply(RES, PSFPadded);
                    b = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
                    iter++;
                }
            }
            return converged;
        }

        public static double EstimateObjectiveImprovement2(double[,] resPadded, double[,] window, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var totalDiff = 0.0;
            var resOld = 0.0;
            var count = 0;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < window.GetLength(0) &
                        x >= 0 & x < window.GetLength(1))
                    {
                        count++;
                        var resVal = resPadded[y + yPsfHalf, x + xPsfHalf];
                        var newRes = resVal + psf[i, j] * xDiff;
                        resOld += (resVal * resVal);
                        totalDiff += (newRes * newRes);
                    }
                }
            }

            return resOld - totalDiff;
        }

        public static void UpdateResiduals2(double[,] resPadded, double[,] window, double[,] psf, int yPixel, int xPixel, double xDiff, int resYOffset, int resXOffset)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < window.GetLength(0) & x >= 0 & x < window.GetLength(1))
                    {
                        resPadded[y + yPsfHalf, x + xPsfHalf] += psf[i, j] * xDiff;
                    }
                }
        }

        public static double CalcElasticNetObjective(double[,] xImage, double[,] window, double[,] aMap, double lambda, double alpha, int yOffset, int xOffset)
        {
            double objective = 0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    var a = QueryIntegral2(aMap, i + yOffset, j + xOffset, window.GetLength(0), window.GetLength(1));
                    objective += lambda * 2 * a * ElasticNetRegularization(xImage[i, j], alpha);
                }
                    
            return objective;
        }

        public static double CalcDataObjective(double[,] resPadded, double[,] xImage, int yPsfOffset, int xPsfOffset)
        {
            double objective = 0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    objective += resPadded[i + yPsfOffset, j + xPsfOffset] * resPadded[i + yPsfOffset, j + xPsfOffset];
            return objective;
        }

        public static double QueryIntegral2(double[,] integral, int yPixel, int xPixel, int yLength, int xLength)
        {
            var yOverShoot = (yPixel + (integral.GetLength(0) - integral.GetLength(0) / 2)) - yLength;
            var xOverShoot = (xPixel + (integral.GetLength(1) - integral.GetLength(1) / 2)) - xLength;
            yOverShoot = Math.Max(0, yOverShoot);
            xOverShoot = Math.Max(0, xOverShoot);
            var yUnderShoot = (-1) * (yPixel - integral.GetLength(0) / 2);
            var xUnderShoot = (-1) * (xPixel - integral.GetLength(1) / 2);
            var yUnderShootIdx = Math.Max(1, yUnderShoot) - 1;
            var xUnderShootIdx = Math.Max(1, xUnderShoot) - 1;

            //PSF completely in picture
            if (yOverShoot <= 0 & xOverShoot <= 0 & yUnderShoot <= 0 & xUnderShoot <= 0)
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1];

            if (yUnderShoot > 0 & xUnderShoot > 0)
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1]
                       - integral[integral.GetLength(0) - 1, xUnderShootIdx]
                       - integral[yUnderShootIdx, integral.GetLength(1) - 1]
                       + integral[yUnderShootIdx, xUnderShootIdx];

            var correction = 0.0;
            if (yUnderShoot > 0)
                correction += integral[yUnderShootIdx, integral.GetLength(1) - xOverShoot - 1];
            if (xUnderShoot > 0)
                correction += integral[integral.GetLength(0) - yOverShoot - 1, xUnderShootIdx];

            return integral[integral.GetLength(0) - 1 - yOverShoot, integral.GetLength(1) - 1 - xOverShoot] - correction;
        }
        #endregion

        public static double ElasticNetRegularization(double value, double alpha) => 1.0 / 2.0 * (1 - alpha) * (value * value) + alpha * Math.Abs(value);
    }
}
