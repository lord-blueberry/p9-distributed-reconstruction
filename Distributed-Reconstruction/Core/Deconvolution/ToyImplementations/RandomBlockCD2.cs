using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using static Core.Common;

namespace Core.Deconvolution.ToyImplementations
{
    public class RandomBlockCD2
    {

        public static bool Deconvolve2(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, Random random, int maxIteration = 100, double epsilon = 1e-4)
        {
            var xImage2 = ToFloatImage(xImage);

            var PSFConvolution = CommonDeprecated.PSF.CalcPaddedFourierConvolution(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFSquared = Fourier2D.Multiply(PSFConvolution, PSFCorrelation);
            var bMapCalculator = new PaddedConvolver(PSFCorrelation, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var resUpdateCalculator = new PaddedConvolver(PSFConvolution, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var bMapUpdateCalculator = new PaddedConvolver(PSFSquared, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));

            var yBlockSize = 2;
            var xBlockSize = 2;
            var bMap = ToFloatImage(residuals);
            bMapCalculator.ConvolveInPlace(bMap);
            FitsIO.Write(bMap, "bmapFirst.fits");

            var xDiff = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var lipschitz = ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var iter = 0;
            while (iter < maxIteration)
            {
                var yB = random.Next(xImage.GetLength(0) / yBlockSize);
                var xB = random.Next(xImage.GetLength(1) / xBlockSize);
                //yB = 64 / yBlockSize;
                //xB = 64 / xBlockSize;
                var block = CopyFrom(bMap, yB, xB, yBlockSize, xBlockSize);

                //var optimized = block * blockInversion;

                var update = block / lipschitz;
                var xOld = CopyFrom(xImage2, yB, xB, yBlockSize, xBlockSize);
                var optimized = xOld + update;

                //shrink
                bool containsNonZero = false;
                for (int i = 0; i < optimized.Count; i++)
                {
                    optimized[i] = CommonDeprecated.ShrinkElasticNet(optimized[i], lambda, alpha);
                    containsNonZero  |=  (optimized[i] - xOld[i]) != 0.0;
                }

                var optDiff = optimized - xOld;
                if (containsNonZero)
                {
                    AddInto(xDiff, optDiff, yB, xB, yBlockSize, xBlockSize);
                    AddInto(xImage2, optDiff, yB, xB, yBlockSize, xBlockSize);
                    //FitsIO.Write(xImage2, "xImageBlock.fits");
                    //FitsIO.Write(xDiff, "xDiff.fits");

                    //update b-map
                    bMapUpdateCalculator.ConvolveInPlace(xDiff);
                    //FitsIO.Write(xDiff, "bMapUpdate.fits");
                    for (int i = 0; i < xDiff.GetLength(0); i++)
                        for (int j = 0; j < xDiff.GetLength(1); j++)
                        {
                            bMap[i, j] -= xDiff[i, j];
                            xDiff[i, j] = 0;
                        }
                    //FitsIO.Write(bMap, "bMap2.fits");

                    //calc residuals for debug purposes
                    AddInto(xDiff, optDiff, yB, xB, yBlockSize, xBlockSize);
                    resUpdateCalculator.ConvolveInPlace(xDiff);
                    //FitsIO.Write(xDiff, "residualsUpdate.fits");
                    for (int i = 0; i < xDiff.GetLength(0); i++)
                        for (int j = 0; j < xDiff.GetLength(1); j++)
                        {
                            residuals[i, j] -= xDiff[i, j];
                            xDiff[i, j] = 0;
                        }
                    //FitsIO.Write(residuals, "residuals2.fits");
                    var l2 = NaiveGreedyCD.CalcDataObjective(residuals);
                    Console.WriteLine(l2);
                }
                iter++;
            }

            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xImage[i, j] = xImage2[i, j];
            return false;
        }

        public static bool Deconvolve(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            FitsIO.Write(residuals, "res.fits");
            var yBlockSize = 2;
            var xBlockSize = 2;

            var PSFCorrelated = CommonDeprecated.PSF.CalculateFourierCorrelation(psf, psf.GetLength(0), psf.GetLength(1));
            var RES = FFT.Forward(residuals);
            var BMAPFourier = Common.Fourier2D.Multiply(RES, PSFCorrelated);
            var bMap = FFT.Backward(BMAPFourier, BMAPFourier.Length);
            //FFT.Shift(bMap);
            FitsIO.Write(bMap, "bMap.fits");
            var PSF = FFT.Forward(CommonDeprecated.Residuals.Pad(psf, psf.GetLength(0), psf.GetLength(1)), 1.0);

            var PSF2Fourier = Common.Fourier2D.Multiply(PSF, PSFCorrelated);

            var xDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];
            //var blockInversion = CalcBlock(psf, yBlockSize, xBlockSize).Inverse();
            var random = new Random(123);

            var lipschitz = ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var iter = 0;
            while (iter < maxIteration)
            {
                /*
                var yB = random.Next(xImage.GetLength(0) / yBlockSize);
                var xB = random.Next(xImage.GetLength(1) / xBlockSize);
                yB = 64 / yBlockSize;
                xB = 64 / xBlockSize;
                var block = CopyFrom(bMap, yB, xB, yBlockSize, xBlockSize);

                //var optimized = block * blockInversion;

                var optimized = block / lipschitz /4;
                var xOld = CopyFrom(xImage, yB, xB, yBlockSize, xBlockSize);
                optimized = xOld + optimized;

                //shrink
                for (int i = 0; i < optimized.Count; i++)
                    optimized[i] = Common.ShrinkElasticNet(optimized[i], lambda, alpha);
                var optDiff = optimized - xOld;
                //AddInto(xDiff, optDiff, yB, xB, yBlockSize, xBlockSize);
                AddInto(xImage, optDiff, yB, xB, yBlockSize, xBlockSize);
                FitsIO.Write(xImage, "xImageBlock.fits");
                FitsIO.Write(xDiff, "xDiff.fits");
                xDiff[64, 64] = 1.0;

                //update b-map
                var XDIFF = FFT.Forward(xDiff);
                XDIFF = Common.Fourier2D.Multiply(XDIFF, PSF2Fourier);
                Common.Fourier2D.SubtractInPlace(BMAPFourier, XDIFF);
                bMap = FFT.Backward(BMAPFourier, BMAPFourier.Length);
                //FFT.Shift(bMap);
                FitsIO.Write(bMap, "bMap2.fits");

                //calc residuals for debug purposes
                var XDIFF2 = FFT.Forward(xDiff);
                XDIFF2 = Common.Fourier2D.Multiply(XDIFF2, PSF);
                var RES2 = Common.Fourier2D.Subtract(RES, XDIFF2);
                var resDebug = FFT.Backward(RES2, RES2.Length);
                var L2 = NaiveGreedyCD.CalcDataObjective(resDebug);
                FitsIO.Write(resDebug, "resDebug.fits");
                var xDiff3 = FFT.Backward(XDIFF2, XDIFF2.Length);
                FitsIO.Write(xDiff3, "recDebug.fits");

                //clear from xDiff
                AddInto(xDiff, -optDiff, yB, xB, yBlockSize, xBlockSize);*/
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
            for (int y = yShift; y < psf.GetLength(0); y++)
                for (int x = xShift; x < psf.GetLength(1); x++)
                    output += psf[y - yShift, x - xShift] * psf[y, x];


            return output;
        }

        private static Matrix<double> CalcBlock(double[,] psf, int yBlockSize, int xBlockSize)
        {
            var size = yBlockSize * xBlockSize;

            var correlations = new double[yBlockSize, xBlockSize];
            var indices = new Tuple<int, int>[size];
            for (int i = 0; i < yBlockSize; i++)
                for (int j = 0; j < xBlockSize; j++)
                {
                    correlations[i, j] = Correlate(psf, i, j);
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

            return correlationMatrix;
        }

        public static Vector<double> CopyFrom(float[,] image, int yB, int xB, int yBlockSize, int xBlockSize)
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

        public static void AddInto(float[,] image, Vector<double> vec, int yB, int xB, int yBlockSize, int xBlockSize)
        {
            var yOffset = yB * yBlockSize;
            var xOffset = xB * xBlockSize;

            int i = 0;
            for (int y = 0; y < yBlockSize; y++)
                for (int x = 0; x < xBlockSize; x++)
                    image[yOffset + y, xOffset + x] += (float)vec[i++];
        }


        public static double ApproximateLipschitz(double[,] psf, int yBlockSize, int xBlockSize)
        {
            var a00 = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    a00 += psf[i, j] * psf[i,j];

            var lipschitz = a00 * yBlockSize * xBlockSize;
            return lipschitz;
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

        public static void RunToy()
        {
            var psf = new double[32, 32];
            for (int i = 15; i < 18; i++)
                for (int j = 15; j < 18; j++)
                    psf[i, j] = 0.5;
            psf[16, 16] = 1.0;
            //psf[17, 17] = 0.8;

            var AA = CalcAgain(psf);

            var yBSize = 8;
            var xBSize = 8;
            var A = CalcBlock(psf, yBSize, xBSize);
            var blockInv =A.Inverse();
            var inv = AA.Inverse();

            var lipschitz = 0.0;
            foreach (var cell in blockInv.Enumerate())
                lipschitz += cell * cell;

            var lipschitz2 = 0.0;
            foreach (var cell in A.Enumerate())
                lipschitz2 += cell * cell;
            lipschitz2 = Math.Sqrt(lipschitz2);

            var lipschitz3 = 0.0;
            var a00 = A[0, 0];
            for (int i = 0; i < yBSize * xBSize; i++)
                lipschitz3 += a00 * a00;
            lipschitz3 = Math.Sqrt(lipschitz3);

            var xImage = new double[32, 32];
            var groundTruth = new double[32, 32];

            groundTruth[16, 16] = 1.0;
            for (int i = 0; i < yBSize; i++)
                for (int j = 0; j < yBSize; j++)
                    groundTruth[16 + i, 16 + j] = 1 + i + j;
            //dirty[16, 17] = 1.8;
            //dirty[17, 16] = 1.1;
            //dirty[17, 17] = 0.5;
            //dirty[16, 20] = 0.5;
            /*
            var IMG = FFT.Forward(groundTruth, 1.0);
            var PSF = FFT.Forward(psf, 1.0);
            var DIRTY = Common.Fourier2D.Multiply(IMG, PSF);
            var residuals = FFT.Backward(DIRTY, (double)(IMG.GetLength(0) * IMG.GetLength(1)));
            FFT.Shift(residuals);

            var RES = FFT.Forward(residuals, 1.0);
            //TODO: WRONG WRONG WRONG, its PSFCorrelated. But does not matter for toy PSF
            var BMAP = Common.Fourier2D.Multiply(RES, PSF);
            var bMap = FFT.Backward(BMAP, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(bMap);

            FitsIO.Write(bMap, "bMapToy.fits");

            for(int i = 0; i < 10; i++)
            {
                var bVec = CopyFrom(bMap, 16 / yBSize, 16 / xBSize, yBSize, xBSize);

                var optimized = (blockInv * bVec);
                var tmp = optimized.ToArray();

                var optLipschitz = (bVec / lipschitz3);

                AddInto(xImage, optLipschitz, 16 / yBSize, 16 / xBSize, yBSize, xBSize);
                var XIMG = FFT.Forward(xImage, 1.0);
                var RECONDIRTY = Common.Fourier2D.Multiply(XIMG, PSF);
                var recon = FFT.Backward(RECONDIRTY, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
                FFT.Shift(recon);
                FitsIO.Write(recon, "recDirty.fits");
                FitsIO.Write(xImage, "xImageToy.fits");

                var RESIDUALS = Common.Fourier2D.Subtract(DIRTY, RECONDIRTY);
                residuals = FFT.Backward(RESIDUALS, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
                FFT.Shift(residuals);
                var l2 = 0.0;
                for (int i2 = 0; i2 < residuals.GetLength(0); i2++)
                    for (int j = 0; j < residuals.GetLength(1); j++)
                        l2 += residuals[i2, j] * residuals[i2, j];


                FitsIO.Write(residuals, "residualsToy.fits");
                BMAP = Common.Fourier2D.Multiply(RESIDUALS, PSF);
                bMap = FFT.Backward(BMAP, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
                //FFT.Shift(bMap);
                FitsIO.Write(bMap, "bMap2Toy.fits");
            }*/
            




            //Deconvolve(xImage, residuals, psf, 0.0, 1.0, 1);

            /*
            AddInto(xImage, res3, 16 / yBSize, 16 / xBSize, yBSize, xBSize);
            var XIMG = FFT.Forward(xImage, 1.0);
            var RESCONV = Common.Fourier2D.Multiply(XIMG, PSF);
            var results = FFT.Backward(RESCONV, (double)(BMAP.GetLength(0) * BMAP.GetLength(1)));
            FFT.Shift(results);
            FitsIO.Write(results, "dirtyConf.fits");
            FitsIO.Write(xImage, "dXXConf.fits");*/



        }

        private static Matrix<double> CalcAgain(double[,] psf)
        {
            var a0 = ToVector(psf);

            var a1 = ToVector(Shift(psf, 0, 1));
            var a2 = ToVector(Shift(psf, 1, 0));
            var a3 = ToVector(Shift(psf, 1, 1));

            var output = new DenseMatrix(a0.Count, 4);
            output.SetColumn(0, a0);
            output.SetColumn(1, a1);
            output.SetColumn(2, a2);
            output.SetColumn(3, a3);


            var A = output.Transpose() * output;

            return A;
        }

        private static Vector<double> ToVector(double[,] img)
        {
            var output = new DenseVector(img.Length);
            int index = 0;
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                    output[index++] = img[i, j];
            return output;
        }

        private static double[,] Shift(double[,] psf, int yShift, int xShift)
        {
            var output = new double[psf.GetLength(0), psf.GetLength(1)];

            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    if (i + yShift < psf.GetLength(0) & j + xShift < psf.GetLength(1))
                        output[i + yShift, j + xShift] = psf[i, j];

            return output;
        }
        #endregion
    }
}
