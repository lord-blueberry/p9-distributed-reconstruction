using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace Single_Reference.Deconvolution
{
    public class RandomBlockCD
    {

        public static bool Deconvolve(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            var yBlockSize = 16;
            var xBlockSize = 16;
            var psfCorrelated = Common.PSF.CalculateFourierCorrelation(psf, residuals.GetLength(0) - psf.GetLength(0), residuals.GetLength(1) - psf.GetLength(1));
            var residualsFourier = FFT.Forward(residuals);
            residualsFourier = Common.Fourier2D.Multiply(residualsFourier, psfCorrelated);
            var bMap = FFT.Backward(residualsFourier, residualsFourier.Length);
            FFT.Shift(bMap);

            var psf2Fourier = Common.Fourier2D.Multiply(psfCorrelated, psfCorrelated);

            var xDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];
            var blockInversion = CalcBlockInversion(psf, yBlockSize, xBlockSize);
            var random = new Random(123);

            var iter = 0;
            while(iter < maxIteration)
            {
                var yB = yBlockSize * random.Next(xImage.GetLength(0) / yBlockSize);
                var xB = xBlockSize * random.Next(xImage.GetLength(1) / xBlockSize);
                var block = CopyFrom(bMap, yB, xB, yBlockSize, xBlockSize);

                var optimized = block * blockInversion;

                //shrink
                for (int i = 0; i < optimized.Count; i++)
                    optimized[i] = Common.ShrinkElasticNet(optimized[i], lambda, alpha);
                AddInto(xDiff, optimized, yB, xB, yBlockSize, xBlockSize);
                AddInto(xImage, optimized, yB, xB, yBlockSize, xBlockSize);

                //update b-map
                var XDIFF = FFT.Forward(xDiff);
                XDIFF = Common.Fourier2D.Multiply(XDIFF, psf2Fourier);
                Common.Fourier2D.SubtractInPlace(residualsFourier, XDIFF);
                bMap = FFT.Backward(residualsFourier, residualsFourier.Length);
                FFT.Shift(bMap);

                //clear from xDiff
                AddInto(xDiff, -optimized, yB, xB, yBlockSize, xBlockSize);

                iter++;
            }

            return false;
        }

        /// <summary>
        /// Calculate correlation of the psf with itself, but shifted
        /// </summary>
        /// <param name="psf"></param>
        /// <param name="yShift"></param>
        /// <param name="xShift"></param>
        /// <returns></returns>
        private static double Correlate(double[,] psf, int yShift, int xShift)
        {
            var output = 0.0;
            for(int y = yShift; y < psf.GetLength(0); y++)
                for(int x = xShift; x < psf.GetLength(1); x++)
                    output += psf[y - yShift, x - xShift] * psf[y, x];
                

            return output;
        }

        private static Matrix<double> CalcBlockInversion(double[,] psf, int yBlockSize, int xBlockSize)
        {
            var size = yBlockSize * xBlockSize;

            var correlations = new double[yBlockSize, xBlockSize];
            var indices = new Tuple<int, int>[size];
            for (int i = 0; i < yBlockSize; i++)
                for (int j = 0; j < xBlockSize; j++)
                {
                    correlations[i, j]= Correlate(psf, i, j);
                    indices[i * yBlockSize + j] = new Tuple<int, int>(i, j);
                }

            var correlationMatrix = new DenseMatrix(size, size);
            for (int y = 0; y < size; y++)
            {
                var yIndex = indices[y];
                for (int x = 0; x < size; x++)
                {
                    var xIndex = indices[x];
                    var cX = Math.Abs(yIndex.Item1 - xIndex.Item1);
                    var cY = Math.Abs(yIndex.Item2 - xIndex.Item2);
                    correlationMatrix[x, y] = correlations[cX, cY];
                }
            }

            return correlationMatrix.Inverse();
        }

        private static Vector<double> CopyFrom(double[,] image, int yB, int xB, int yBlockSize, int xBlockSize)
        {
            var yOffset = yB * yBlockSize;
            var xOffset = xB * xBlockSize;

            int i = 0;
            var vec = new DenseVector(yBlockSize * xBlockSize);
            for (int y = 0; y < yBlockSize; y++)
                for (int x = 0; x < xBlockSize; x++)
                    vec[i++] = image[yOffset + y, xOffset + x];

            return vec;
        }

        private static void AddInto(double[,] image, Vector<double> vec, int yB, int xB, int yBlockSize, int xBlockSize)
        {
            var yOffset = yB * yBlockSize;
            var xOffset = xB * xBlockSize;

            int i = 0;
            for (int y = 0; y < yBlockSize; y++)
                for (int x = 0; x < xBlockSize; x++)
                    image[yOffset + y, xOffset + x] += vec[i++];
        }

        #region toDelete
        private static DenseMatrix CreateA2(List<DenseVector> vecs)
        {
            var A = new DenseMatrix(vecs[0].Count, vecs[0].Count);

            for (int i = 0; i < vecs.Count; i++)
            {
                var vec = vecs[i];
                for (int j = 0; j < vec.Count; j++)
                {
                    A[i, i + j] = vec[j];
                    A[i + j, i] = vec[j];
                }
            }
            var last = vecs[0].Count - 1;
            A[last, last] = vecs[0][0];

            return A;
        }

        public static void Run()
        {
            var psf = new double[32, 32];
            for (int i = 15; i < 18; i++)
                for (int j = 15; j < 18; j++)
                    psf[i, j] = 0.5;
            psf[16, 16] = 1.0;
            psf[17, 17] = 0.8;

            var aaa = CalcBlockInversion(psf, 2, 2);

            var a = new DenseVector(4);
            a[0] = Correlate(psf, 0, 0);
            a[1] = Correlate(psf, 0, 1);
            a[2] = Correlate(psf, 1, 0);
            a[3] = Correlate(psf, 1, 1);

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
            //TODO: WRONG WRONG WRONG, its PSFCorrelated. But does not matter for toy PSF
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
        #endregion
    }
}
