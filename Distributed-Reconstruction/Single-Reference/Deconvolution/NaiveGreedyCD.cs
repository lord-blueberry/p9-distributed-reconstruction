using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;
using static System.Math;
using System.Diagnostics;
using System.IO;

namespace Single_Reference.Deconvolution
{
    class IdiotCD
    {
        public static void Run()
        {
            //var frequencies = FitsIO.ReadFrequencies(@"C:\Users\Jon\github\p9-data\small\fits\simulation_point\freq.fits");
            //var uvw = FitsIO.ReadUVW(@"C:\Users\Jon\github\p9-data\small\fits\simulation_point\uvw.fits");
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset

            var visibilitiesCount = flags.Length;
            int gridSize = 64;
            int subgridsize = 16;
            int kernelSize = 8;
            int max_nr_timesteps = 64;
            double cellSize = 2.0 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf = FFT.GridIFFT(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);
            var maxPsf = psf[gridSize / 2, gridSize / 2];
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    psf[i, j] = psf[i, j] / maxPsf;
            FitsIO.Write(psf, "psf.fits");

            var truth = new double[64, 64];
            truth[40, 50] = 1.5;
            truth[0, 0] = 1.7;
            var dirty = ConvolveFFTPadded(truth, psf);
            FitsIO.Write(truth, "truth.fits");
            FitsIO.Write(dirty, "dirty.fits");

            var psf2 = ConvolveFFT(psf, psf);
            var b = ConvolveFFTPadded(dirty, psf);
            var a = psf2[gridSize / 2, gridSize / 2];

            /*
            var integral = CalcPSf2Integral(psf);
            FitsIO.Write(integral, "psfIntegral.fits");
            var c0 = new double[64, 64];
            var qY = 0;
            var qX = 0;
            c0[qY, qX] = 1.0;
            c0 = Convolve(c0, psf);
            FitsIO.Write(c0, "cx0.fits");
            var cx = ConvolveFFT(c0, psf);
            FitsIO.Write(cx, "cx1.fits");
            var a2 = cx[qY, qX];
            var res = QueryIntegral(integral, qY, qX);*/

            var x = new double[gridSize, gridSize];
            //Deconv(x, dirty, psf, psf2, a, 0.0);

            var dCopy = new double[gridSize, gridSize];
            for (int i = 0; i < b.GetLength(0); i++)
                for (int j = 0; j < b.GetLength(1); j++)
                    dCopy[i, j] = dirty[i, j];
            var x2 = new double[gridSize, gridSize];
            var converged = GreedyCD.Deconvolve(x2, dirty, psf, 0.0, 1.0, 500, dCopy);
        }


        public static void Deconv(double[,] xImage, double[,] dirty, double[,] psf, double[,] psf2, double a, double lambda, int maxIter = 10)
        {
            var aMap = CalcPSf2IntegralSquared(psf);
            var iter = 0;
            var converged = false;
            var FO = new double[xImage.GetLength(0), xImage.GetLength(1)];
            var XO = new double[xImage.GetLength(0), xImage.GetLength(1)];
            
            while (iter < maxIter & !converged)
            {
                var convolved = ConvolveFFTPadded(xImage, psf);
                var residuals = Subtract(dirty, convolved);
                FitsIO.Write(residuals, "deconv_residuals_" + iter + ".fits");
                var bMap = ConvolveFFTPadded(residuals, psf);
                FitsIO.Write(bMap, "deconv_bMap_" + iter + ".fits");
                var objectiveVal = CalcDataObjective(residuals);
                objectiveVal += CalcL1Objective(xImage, aMap, lambda);
                var minVal = Double.MaxValue;
                var yPixel = -1;
                var xPixel = -1;
                var xNew = 0.0;
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                    {
                        if (j == 43)
                            Console.Write("");
                        var currentB = bMap[i, j];
                        var currentA = QueryIntegral(aMap, i, j);
                        //var xDiff = currentB / a;
                        var xDiff = currentB / currentA;
                        var x = xImage[i, j] + xDiff;
                        x = ShrinkAbsolute(x, lambda);
                        x = Math.Max(x, 0);
                        XO[i, j] = x;
                        var currentO = EstimateObjective(xImage, dirty, psf, i, j, x, aMap, lambda);
                        if (Math.Abs(x - xImage[i, j]) > 1e-6)
                            if (currentO <= objectiveVal + 1e-6)
                                Console.Write("");
                            else
                                Console.Write("ERROR");
                        FO[i, j] = currentO;
                        if(minVal > currentO)
                        {
                            minVal = currentO;
                            xNew = x;
                            yPixel = i;
                            xPixel = j;
                        }
                    }
                FitsIO.Write(FO, "deconv_FO_" + iter+".fits");
                FitsIO.Write(XO, "deconv_XO_" + iter + ".fits");
                var old = xImage[yPixel, xPixel];
                if (Math.Abs(old - xNew) > 1e-2)
                    xImage[yPixel, xPixel] = xNew;
                else
                    converged = true;
                iter++;
                FitsIO.Write(xImage, "deconv_x_" + iter + ".fits");

            }
        }

        #region psf Integral
        public static double[,] CalcPSf2IntegralSquared(double[,] psf)
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

        public static double[,] CalcPSfIntegral(double[,] psf2)
        {
            var integral = new double[psf2.GetLength(0), psf2.GetLength(1)];
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                {
                    var iBefore = i > 0 ? integral[i - 1, j] : 0.0;
                    var jBefore = j > 0 ? integral[i, j - 1] : 0.0;
                    var ijBefore = i > 0 & j > 0 ? integral[i - 1, j - 1] : 0.0;
                    var current = psf2[i, j];
                    integral[i, j] = current + iBefore + jBefore - ijBefore;
                }

            return integral;
        }

        public static double QueryIntegral(double[,] integral, int yPixel, int xPixel)
        {
            var yPsfHalf = 32;
            var xPsfHalf = 32;
            var yOverShoot = integral.GetLength(0) * 2 - (yPixel + yPsfHalf) -1;
            var xOverShoot = integral.GetLength(1) * 2 - (xPixel + xPsfHalf) -1;

            var yCorrection = yOverShoot % integral.GetLength(0);
            var xCorrection = xOverShoot % integral.GetLength(1);

            if (yCorrection == yOverShoot & xCorrection == xOverShoot)
            {
                return integral[yCorrection, xCorrection];
            }
            else if(yCorrection == yOverShoot | xCorrection == xOverShoot)
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
        #endregion

        public static double EstimateObjective(double[,] xImage, double[,] dirty, double[,] psf, int yPixel, int xPixel, double newX, double[,] aMap, double lambda)
        {
            var xOld = xImage[yPixel, xPixel];

            xImage[yPixel, xPixel] = newX;
            var convolved = ConvolveFFTPadded(xImage, psf);
            var residuals = Subtract(dirty, convolved);
            var currentO = CalcDataObjective(residuals);
            currentO += CalcL1Objective(xImage, aMap, lambda);
            xImage[yPixel, xPixel] = xOld;

            return currentO;
        }

        public static double[,] Subtract(double[,] x, double[,]y)
        {
            var output = new double[x.GetLength(0), x.GetLength(1)];
            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    output[i, j] = x[i, j] - y[i, j];
            return output;
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
            var IMG = FFT.FFTDebug(img2, 1.0);
            var PSF = FFT.FFTDebug(psf2, 1.0);
            var CONV = IDG.Multiply(IMG, PSF);
            var conv = FFT.IFFTDebug(CONV, img2.GetLength(0) * img2.GetLength(1));
            FFT.Shift(conv);

            var convOut = new double[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    convOut[i, j] = conv[i + yHalf, j + xHalf];
                }

            return convOut;
        }

        public static double[,] ConvolveFFT(double[,] img, double[,] psf)
        {
            var IMG = FFT.FFTDebug(img, 1.0);
            var PSF = FFT.FFTDebug(psf, 1.0);
            var CONV = IDG.Multiply(IMG, PSF);
            var conv = FFT.IFFTDebug(CONV, img.GetLength(0) * img.GetLength(1));
            FFT.Shift(conv);
            return conv;
        }

        public static double[,] Convolve(double[,] image, double[,] kernel)
        {
            var output = new double[image.GetLength(0), image.GetLength(1)];
            for (int y = 0; y < image.GetLength(0); y++)
            {
                for (int x = 0; x < image.GetLength(1); x++)
                {
                    double sum = 0;
                    for (int yk = 0; yk < kernel.GetLength(0); yk++)
                    {
                        for (int xk = 0; xk < kernel.GetLength(1); xk++)
                        {
                            int ySrc = y + yk - ((kernel.GetLength(0) - 1) - kernel.GetLength(0) / 2);
                            int xSrc = x + xk - ((kernel.GetLength(1) - 1) - kernel.GetLength(1) / 2);
                            if (ySrc >= 0 & ySrc < image.GetLength(0) &
                                xSrc >= 0 & xSrc < image.GetLength(1))
                            {
                                sum += image[ySrc, xSrc] * kernel[kernel.GetLength(0) - 1 - yk, kernel.GetLength(1) - 1 - xk];
                            }
                        }
                    }
                    output[y, x] = sum;
                }
            }
            return output;
        }

        public static double CalcDataObjective(double[,] res)
        {
            var objective = 0.0;
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    objective += res[y, x] * res[y, x];

            return objective;
        }

        public static double CalcL1Objective(double[,] xImage, double[,] aMap, double lambda)
        {
            var objective = 0.0;
            for (int y = 0; y < xImage.GetLength(0); y++)
                for (int x = 0; x < xImage.GetLength(1); x++)
                    objective += Math.Abs(xImage[y, x]) * lambda * 2* QueryIntegral(aMap, y, x);
                    //objective += Math.Abs(xImage[y, x]) * lambda * 2 * aMap[aMap.GetLength(0) -1, aMap.GetLength(1) -1];
            return objective;
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
