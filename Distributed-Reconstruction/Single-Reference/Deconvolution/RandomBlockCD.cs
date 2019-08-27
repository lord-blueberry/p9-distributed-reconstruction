using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace Single_Reference.Deconvolution
{
    public class RandomBlockCD
    {

        public static void Run()
        {
            var psf = new double[32, 32];
            for (int i = 15; i < 18; i++)
                for (int j = 15; j < 18; j++)
                    psf[i, j] = 0.5;
            psf[16, 16] = 1.0;

            var A = new DenseMatrix(2, 2);
            A[0, 0] = CalcAMatrixEntry(psf, 0, 0);
            A[1, 1] = A[0, 0];
            A[1, 0] = CalcAMatrixEntry(psf, 0, 1);
            A[0, 1] = A[1, 0];
            var inv = A.Inverse();
            var arr = inv.ToArray();

            var xImage = new double[32, 32];
            var dirty = new double[32, 32];

            dirty[16, 16] = 1.0;
            dirty[16, 17] = 1.8;
            var IMG = FFT.Forward(dirty, 1.0);
            var PSF = FFT.Forward(psf, 1.0);
            var CONV = Common.Fourier2D.Multiply(IMG, PSF);
            var residuals = FFT.Backward(CONV, (double)(IMG.GetLength(0) * IMG.GetLength(1)));
            FFT.Shift(residuals);

            var RES = FFT.Forward(residuals, 1.0);
            var BMAP = Common.Fourier2D.Multiply(RES, PSF);
            var bMap = FFT.Backward(BMAP, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(bMap);

            FitsIO.Write(bMap, "dirtyConf.fits");
            var x0 = bMap[16, 16] * inv[0, 0] + bMap[16, 17] * inv[0, 1];
            var x1 = bMap[16, 16] * inv[1, 0] + bMap[16, 17] * inv[1, 1];

        }
        public static bool Deconvolve(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            //var bMap = 
            var A = new DenseMatrix(2, 2);
            A[0, 0] = CalcAMatrixEntry(psf, 0, 0);
            A[1, 1] = A[0, 0];
            A[1, 0] = CalcAMatrixEntry(psf, 0, 1);
            A[0, 1] = A[1, 0];

            var inv = A.Inverse();
            var arr = inv.ToArray();

            return false;
        }

        private static double CalcAMatrixEntry(double[,] psf, int yOffset, int xOffset)
        {
            var output = 0.0;

            for(int y = yOffset; y < psf.GetLength(0); y++)
                for(int x = xOffset; x < psf.GetLength(1); x++)
                {
                    output += psf[y - yOffset, x - xOffset] * psf[y, x];
                }

            return output;
        }

    }
}
