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

            /*
            var A = new DenseMatrix(4, 4);
            var corr = new double[3];
            corr[0] = 1/CalcAMatrixEntry(psf, 0, 1);
            corr[1] = 1/CalcAMatrixEntry(psf, 0, 2);
            corr[2] = 1/CalcAMatrixEntry(psf, 0, 3);
            var sum = 0.0;
            InfToZero(corr);
            for (int i = 0; i < corr.Length; i++)
                sum += corr[i];

            A[0, 0] =  CalcAMatrixEntry(psf, 0, 0);
            A[1, 1] = A[0, 0];
            A[2, 2] = A[0, 0];
            A[1, 0] = CalcAMatrixEntry(psf, 0, 1);
            A[0, 1] = A[1, 0];
            A[1, 2] = A[1, 0];
            A[2, 1] = A[1, 0];
            A[2, 3] = A[1, 0];
            A[3, 2] = A[1, 0];
            A[2, 0] = CalcAMatrixEntry(psf, 0, 2);
            A[0, 2] = A[2, 0];
            A[3, 1] = A[2, 0];
            A[1, 3] = A[2, 0];
            A[3, 0] = CalcAMatrixEntry(psf, 0, 3);
            A[0, 3] = A[3, 0];*/


            var A = new DenseMatrix(3, 3);
            /*A[0, 0] = CalcAMatrixEntry(psf, 0, 0);
            A[1, 1] = A[0, 0];
            A[1, 0] = CalcAMatrixEntry(psf, 0, 1);
            A[0, 1] = A[1, 0];*/

            A[0, 0] = CalcAMatrixEntry(psf, 0, 0);
            A[1, 1] = A[0, 0];
            A[2, 2] = A[0, 0];
            A[1, 0] = CalcAMatrixEntry(psf, 0, 1);
            A[0, 1] = A[1, 0];
            A[1, 2] = A[1, 0];
            A[2, 1] = A[1, 0];
            A[2, 0] = CalcAMatrixEntry(psf, 0, 2);
            A[0, 2] = A[2, 0];

            var blaA = A.ToArray();
            var inv = A.Inverse();
            var arr = inv.ToArray();

            var xImage = new double[32, 32];
            var dirty = new double[32, 32];

            dirty[16, 16] = 1.0;
            //dirty[16, 17] = 1.8;
            dirty[16, 18] = 1.1;
            //dirty[16, 19] = 0.5;
            var IMG = FFT.Forward(dirty, 1.0);
            var PSF = FFT.Forward(psf, 1.0);
            var CONV = Common.Fourier2D.Multiply(IMG, PSF);
            var residuals = FFT.Backward(CONV, (double)(IMG.GetLength(0) * IMG.GetLength(1)));
            FFT.Shift(residuals);

            var RES = FFT.Forward(residuals, 1.0);
            var BMAP = Common.Fourier2D.Multiply(RES, PSF);
            var bMap = FFT.Backward(BMAP, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(bMap);

            FitsIO.Write(residuals, "resConf.fits");
            var bVec = new DenseVector(3);
            bVec[0] = bMap[16, 16];
            bVec[1] = bMap[16, 17];
            bVec[2] = bMap[16, 18];
            //bVec[3] = bMap[16, 19];

 
            var res3 = (inv * bVec).ToArray();
            xImage[16, 16] = res3[0];
            xImage[16, 17] = res3[1];
            xImage[16, 18] = res3[2];
            //xImage[16, 19] = res3[3];
            var XIMG = FFT.Forward(xImage, 1.0);
            var RESCONV = Common.Fourier2D.Multiply(XIMG, PSF);
            var results = FFT.Backward(RESCONV, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(results);
            FitsIO.Write(results, "dirtyConf.fits");

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

        private static void InfToZero(double[] mat)
        {
            for (int j = 0; j < mat.Length; j++)
                {
                    if (!Double.IsFinite(mat[j]))
                        mat[j] = 0;
                }
            
        }
    }
}
