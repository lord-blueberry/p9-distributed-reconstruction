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

   
            var a = new DenseVector(4);
            a[0] = CalcAMatrixEntry(psf, 0, 0);
            a[1] = CalcAMatrixEntry(psf, 0, 1);
            a[2] = CalcAMatrixEntry(psf, 1, 0);
            a[3] = CalcAMatrixEntry(psf, 1, 1);

            var a01 = new DenseVector(3);
            a01[0] = a[0];
            a01[1] = a[3];
            a01[2] = a[2];

            var a10 = new DenseVector(2);
            a10[0] = a[0];
            a10[1] = a[1];
            var LA = new List<DenseVector>(3);
            LA.Add(a);
            LA.Add(a01);
            LA.Add(a10);

            var A2 = CreateA2(LA);

            //a[5] = CalcAMatrixEntry(psf, 6, 6);
            //a[4] = CalcAMatrixEntry(psf, 1, 0);
            /*a[4] = CalcAMatrixEntry(psf, 0, 2);
            a[5] = CalcAMatrixEntry(psf, 2, 0);
            a[6] = CalcAMatrixEntry(psf, 1, 2);
            a[7] = CalcAMatrixEntry(psf, 2, 1);
            a[8] = CalcAMatrixEntry(psf, 2, 2);*/

            var inv = A2.Inverse();
            var arr = inv.ToArray();

            var xImage = new double[32, 32];
            var dirty = new double[32, 32];

            dirty[16, 16] = 1.0;
            dirty[16, 17] = 1.8;
            dirty[17, 16] = 1.1;
            dirty[17, 17] = 0.5;
            //dirty[16, 20] = 0.5;
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
            var bVec = new DenseVector(4);
            bVec[0] = bMap[16, 16];
            bVec[1] = bMap[16, 17];
            bVec[2] = bMap[17, 16];
            bVec[3] = bMap[17, 17];

            //bVec[5] = bMap[22, 22];
            /*
            bVec[4] = bMap[16, 18];
            bVec[5] = bMap[18, 16];
            bVec[6] = bMap[17, 18];
            bVec[7] = bMap[18, 17];
            bVec[8] = bMap[18, 18];*/


            var res3 = (bVec * inv).ToArray();
            xImage[16, 16] = res3[0];
            xImage[16, 17] = res3[1];
            xImage[17, 16] = res3[2];
            xImage[17, 17] = res3[3];
            var XIMG = FFT.Forward(xImage, 1.0);
            var RESCONV = Common.Fourier2D.Multiply(XIMG, PSF);
            var results = FFT.Backward(RESCONV, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(results);
            FitsIO.Write(results, "dirtyConf.fits");
            FitsIO.Write(xImage, "dXXConf.fits");
            Console.Write(inv);

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

        private static DenseMatrix CreateA(DenseVector vec)
        {
            var A = new DenseMatrix(vec.Count, vec.Count);
            for(int i = 0; i < vec.Count; i++)
            {
                for(int j = 0; j < vec.Count; j++)
                {
                    A[j, i] = vec[Math.Abs((j - i) % vec.Count)];
                }
            }

            return A;
        }

        private static DenseMatrix CreateA2(List<DenseVector> vecs)
        {
            var A = new DenseMatrix(vecs[0].Count, vecs[0].Count);
            
            for(int i = 0; i < vecs.Count; i++)
            {
                var vec = vecs[i];
                for(int j = 0; j < vec.Count; j++)
                {
                    A[i, i + j] = vec[j];
                    A[i + j, i] = vec[j];
                }
            }
            var last = vecs[0].Count - 1;
            A[last, last] = vecs[0][0];

            return A;
        }
    }
}
