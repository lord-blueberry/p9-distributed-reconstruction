using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using System.Numerics;
using static System.Math;
using System.Diagnostics;
using System.IO;

namespace Single_Reference
{
    class IdiotCD
    {
        public static void Run()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\Users\Jon\github\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\Users\Jon\github\p9-data\small\fits\simulation_point\uvw.fits");
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
            FitsIO.Write(psf, "psf.fits");

            var truth = new double[64, 64];
            truth[40, 25] = 1.5;
            truth[25, 35] = 2.5;
            var dirty = Convolve(truth, psf);
            FitsIO.Write(truth, "truth.fits");
            FitsIO.Write(dirty, "dirty.fits");

            var x = new double[gridSize, gridSize];
            Deconv(x, dirty, psf);
        }

        public static void Deconv(double[,] xImage, double[,] dirty, double[,] psf)
        {
            var iter = 0;
            var maxIter = 10;
            var converged = false;
            var lambda = 0.0;
            var fuckingA = CalcPSFSquared(psf);
            var FO = new double[xImage.GetLength(0), xImage.GetLength(1)];
            var XO = new double[xImage.GetLength(0), xImage.GetLength(1)];
            while (iter < maxIter & !converged)
            {
                var convolved = Convolve(xImage, psf);
                var residuals = Subtract(dirty, convolved);
                FitsIO.Write(residuals, "residuals_" + iter + ".fits");
                var bMAP = Convolve(residuals, psf);
                FitsIO.Write(bMAP, "bMap_" + iter + ".fits");
                var currentO = CalcDataObjective(residuals);
                currentO += CalcL1Objective(xImage, lambda);
                var fuckingMIN = Double.MaxValue;
                var yPixel = -1;
                var xPixel = -1;
                var xNew = 0.0;
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                    {
                        if (i == 25 & j == 35)
                            Console.Write("");
                        if (i == 40 & j == 25)
                            Console.Write("");
                        var fuckingB = bMAP[i, j];
                        var xDiff = fuckingB / fuckingA;
                        var x = xImage[i, j] + xDiff;
                        x = ShrinkAbsolute(x, lambda);
                        if (x < xImage[i, j])
                            Console.Write("OI");
                        XO[i, j] = x;
                        var fuckingO = EstimateObjective(xImage, dirty, psf, i, j, x, lambda);
                        FO[i, j] = fuckingO;
                        if(fuckingMIN > fuckingO)
                        {
                            fuckingMIN = fuckingO;
                            xNew = x;
                            yPixel = i;
                            xPixel = j;
                        }
                    }
                FitsIO.Write(FO, "FO_" + iter+".fits");
                FitsIO.Write(XO, "XO_" + iter + ".fits");
                var fuckingOld = xImage[yPixel, xPixel];
                if (Math.Abs(fuckingOld - xNew) > 1e-2)
                    xImage[yPixel, xPixel] = xNew;
                else
                    converged = true;
                iter++;
                FitsIO.Write(xImage, "x_" + iter + ".fits");

            }
        }

        public static double EstimateObjective(double[,] xImage, double[,] dirty, double[,] psf, int yPixel, int xPixel, double fuckingX, double fuckingLambda)
        {
            var xOld = xImage[yPixel, xPixel];

            xImage[yPixel, xPixel] = fuckingX;
            var convolved = Convolve(xImage, psf);
            var residuals = Subtract(dirty, convolved);
            var currentO = CalcDataObjective(residuals);
            currentO += CalcL1Objective(xImage, fuckingLambda);

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


        public static double[,] Convolve(double[,] img, double[,] psf)
        {
            var IMG = FFT.ForwardFFTDebug(img, 1.0);
            var PSF = FFT.ForwardFFTDebug(psf, 1.0);
            var CONV = IDG.Multiply(IMG, PSF);
            var conv = FFT.ForwardIFFTDebug(CONV, img.GetLength(0) * img.GetLength(1));
            FFT.Shift(conv);
            return conv;
        }
        public static double CalcDataObjective(double[,] res)
        {
            var objective = 0.0;
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    objective += res[y, x] * res[y, x];

            return objective;
        }

        public static double CalcL1Objective(double[,] xImage, double lambda)
        {
            var objective = 0.0;
            for (int y = 0; y < xImage.GetLength(0); y++)
                for (int x = 0; x < xImage.GetLength(1); x++)
                    objective += Math.Abs(xImage[y, x]) * lambda;
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
