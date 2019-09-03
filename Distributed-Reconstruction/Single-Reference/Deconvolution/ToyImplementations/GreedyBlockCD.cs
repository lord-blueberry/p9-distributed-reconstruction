using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    public class GreedyBlockCD
    {

        public static bool Deconvolve2(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            var xImage2 = ToFloatImage(xImage);

            var PSFConvolution = CommonDeprecated.PSF.CalcPaddedFourierConvolution(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFSquared = Fourier2D.Multiply(PSFConvolution, PSFCorrelation);
            var bMapCalculator = new PaddedConvolver(PSFCorrelation, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var resUpdateCalculator = new PaddedConvolver(PSFConvolution, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var bMapUpdateCalculator = new PaddedConvolver(PSFSquared, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));

            var yBlockSize = 4;
            var xBlockSize = 4;
            lambda = lambda / (yBlockSize * xBlockSize);
            var bMap = ToFloatImage(residuals);
            bMapCalculator.ConvolveInPlace(bMap);
            FitsIO.Write(bMap, "bmapFirst.fits");

            var xDiff = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var lipschitz = ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var iter = 0;
            while (iter < maxIteration)
            {
                var maxBlock = GetMaxBlock(bMap, xImage2, lipschitz, (float)lambda, (float)alpha, yBlockSize, xBlockSize);
                var yB = maxBlock.Item1;
                var xB = maxBlock.Item2;
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
                    optimized[i] = Common.ShrinkElasticNet(optimized[i], lambda, alpha);
                    containsNonZero |= (optimized[i] - xOld[i]) != 0.0;
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
                    /*AddInto(xDiff, optDiff, yB, xB, yBlockSize, xBlockSize);
                    resUpdateCalculator.ConvolveInPlace(xDiff);
                    //FitsIO.Write(xDiff, "residualsUpdate.fits");
                    for (int i = 0; i < xDiff.GetLength(0); i++)
                        for (int j = 0; j < xDiff.GetLength(1); j++)
                        {
                            residuals[i, j] -= xDiff[i, j];
                            xDiff[i, j] = 0;
                        }
                    //FitsIO.Write(residuals, "residuals2.fits");*/
                    //var l2 = NaiveGreedyCD.CalcDataObjective(residuals);
                    if (maxBlock.Item3 < epsilon)
                        break;

                    Console.WriteLine(maxBlock.Item3 +"\t yB = " + yB + "\t xB =" + xB);
                }
                iter++;
            }

            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xImage[i, j] = xImage2[i, j];
            return false;
        }

        private static void AddInto(float[,] image, Vector<double> vec, int yB, int xB, int yBlockSize, int xBlockSize)
        {
            var yOffset = yB * yBlockSize;
            var xOffset = xB * xBlockSize;

            int i = 0;
            for (int y = 0; y < yBlockSize; y++)
                for (int x = 0; x < xBlockSize; x++)
                    image[yOffset + y, xOffset + x] += (float)vec[i++];
        }


        private static double ApproximateLipschitz(double[,] psf, int yBlockSize, int xBlockSize)
        {
            var a00 = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    a00 += psf[i, j] * psf[i, j];

            var lipschitz = a00 * yBlockSize * xBlockSize;
            return lipschitz;
        }

        private static Vector<double> CopyFrom(float[,] image, int yB, int xB, int yBlockSize, int xBlockSize)
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

        private static Tuple<int,int, double> GetMaxBlock(float[,] bMap, float[,] xImage, double lipschitz, float lambda, float alpha, int yBlockSize, int xBlockSize)
        {
            var yBlocks = bMap.GetLength(0) / yBlockSize;
            var xBlocks = bMap.GetLength(1) / xBlockSize;

            var maxSum = 0.0;
            var yBlockIdx = -1;
            var xBlockIdx = -1;
            for(int i = 0; i < yBlocks; i++)
                for(int j = 0; j < xBlocks; j++)
                {
                    int yIdx = i * yBlockSize;
                    int xIdx = j * xBlockSize;
                    var sum = 0.0;
                    for(int y = yIdx; y < yIdx + yBlockSize;y++)
                        for(int x = xIdx; x < xIdx + xBlockSize; x++)
                        {
                            var opt = bMap[y, x] / lipschitz;
                            var shrink = ShrinkElasticNet(xImage[y, x] + opt, lambda, alpha);
                            sum += Math.Abs(shrink - xImage[y, x]);
                        }
                    if(maxSum < sum)
                    {
                        maxSum = sum;
                        yBlockIdx = i;
                        xBlockIdx = j;
                    }
                }

            return new Tuple<int, int, double>(yBlockIdx, xBlockIdx, maxSum);
        }
    }
}
