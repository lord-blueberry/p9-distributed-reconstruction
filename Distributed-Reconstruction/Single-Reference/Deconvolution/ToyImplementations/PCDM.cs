using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    class PCDM
    {
        public static bool DeconvolveRandom(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, Random random, int maxIteration = 100, double epsilon = 1e-4)
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

            var xDiff = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var theta = 1; //2; //theta, also number of processors.
            var degreeOfSep = RandomCD.CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1) * (theta - 1) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = RandomBlockCD2.ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            lipschitz *= beta;
            lambda = lambda / (yBlockSize * xBlockSize * beta);

            var iter = 0;
            while (iter < maxIteration)
            {
                bool containsNonZero = false;
                var blockSamples = RandomCD.CreateSamples(blockCount, theta, random);
                for (int i = 0; i < blockSamples.Length; i++)
                {
                    var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                    var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);

                    var block = RandomBlockCD2.CopyFrom(bMap, yBlock, xBlock, yBlockSize, xBlockSize);

                    var update = block / (beta * lipschitz);

                    var xOld = RandomBlockCD2.CopyFrom(xImage2, yBlock, xBlock, yBlockSize, xBlockSize);
                    var optimized = xOld + update;

                    //shrink
                    for (int j = 0; j < optimized.Count; j++)
                    {
                        optimized[j] = Common.ShrinkElasticNet(optimized[j], lambda, alpha);
                        containsNonZero |= (optimized[j] - xOld[j]) != 0.0;
                    }

                    var optDiff = optimized - xOld;
                    RandomBlockCD2.AddInto(xDiff, optDiff, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(xImage2, optDiff, yBlock, xBlock, yBlockSize, xBlockSize);
                }


                if (containsNonZero)
                {
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
                }
                iter++;
            }

            var elasticNet = 0.0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    xDiff[i, j] = xImage2[i, j] - (float)xImage[i, j];
                    xImage[i, j] = xImage2[i, j];

                    elasticNet += lambda * 2 * lipschitz * GreedyBlockCD.ElasticNetPenalty(xImage2[i, j], (float)alpha);
                }


            resUpdateCalculator.ConvolveInPlace(xDiff);
            //FitsIO.Write(xDiff, "residualsUpdate.fits");
            for (int i = 0; i < xDiff.GetLength(0); i++)
                for (int j = 0; j < xDiff.GetLength(1); j++)
                {
                    residuals[i, j] -= xDiff[i, j];
                    xDiff[i, j] = 0;
                }
            var l2Penalty = NaiveGreedyCD.CalcDataObjective(residuals);
            Console.WriteLine("-------------------------");
            Console.WriteLine((l2Penalty + elasticNet));
            var io = System.IO.File.AppendText("penalty" + yBlockSize + ".txt");
            io.WriteLine("l2: " + l2Penalty + "\telastic: " + elasticNet + "\t " + (l2Penalty + elasticNet));
            io.Close();
            Console.WriteLine("-------------------------");


            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xImage[i, j] = xImage2[i, j];
            return false;
        }

        public static bool Deconvolve2(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, int blockSize, int maxIteration = 100, double epsilon = 1e-4)
        {
            var xImage2 = ToFloatImage(xImage);

            var PSFConvolution = CommonDeprecated.PSF.CalcPaddedFourierConvolution(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFSquared = Fourier2D.Multiply(PSFConvolution, PSFCorrelation);
            var bMapCalculator = new PaddedConvolver(PSFCorrelation, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var resUpdateCalculator = new PaddedConvolver(PSFConvolution, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var bMapUpdateCalculator = new PaddedConvolver(PSFSquared, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));

            var yBlockSize = blockSize;
            var xBlockSize = blockSize;

            var bMap = ToFloatImage(residuals);
            bMapCalculator.ConvolveInPlace(bMap);

            var xDiff = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var theta = 1; //2; //theta, also number of processors.
            var degreeOfSep = RandomCD.CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1) * (theta - 1) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = RandomBlockCD2.ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            lipschitz *= beta;
            lambda = lambda / (yBlockSize * xBlockSize * beta);

            var iter = 0;
            while (iter < maxIteration)
            {
                bool containsNonZero = false;
                var maxBlocks = GetMaxBlocks(bMap, xImage2, lipschitz, (float)lambda, (float)alpha, yBlockSize, xBlockSize, theta);
                foreach(var b in maxBlocks)
                {
                    var yBlock = b.Item1;
                    var xBlock = b.Item2;
                    var block = RandomBlockCD2.CopyFrom(bMap, yBlock, xBlock, yBlockSize, xBlockSize);

                    var update = block / (beta * lipschitz);

                    var xOld = RandomBlockCD2.CopyFrom(xImage2, yBlock, xBlock, yBlockSize, xBlockSize);
                    var optimized = xOld + update;

                    //shrink
                    for (int j = 0; j < optimized.Count; j++)
                    {
                        optimized[j] = Common.ShrinkElasticNet(optimized[j], lambda, alpha);
                        containsNonZero |= (optimized[j] - xOld[j]) != 0.0;
                    }

                    var optDiff = optimized - xOld;
                    RandomBlockCD2.AddInto(xDiff, optDiff, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(xImage2, optDiff, yBlock, xBlock, yBlockSize, xBlockSize);
                }

                if (containsNonZero)
                {
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

                    /*if (maxBlock.Item3 < epsilon)
                        break;*/

                    //Console.WriteLine(maxBlock.Item3 + "\t yB = " + yB + "\t xB =" + xB);
                }
                iter++;
            }
            var elasticNet = 0.0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    xDiff[i, j] = xImage2[i, j] - (float)xImage[i, j];
                    xImage[i, j] = xImage2[i, j];

                    elasticNet += lambda * 2 * lipschitz * GreedyBlockCD.ElasticNetPenalty(xImage2[i, j], (float)alpha);
                }


            resUpdateCalculator.ConvolveInPlace(xDiff);
            //FitsIO.Write(xDiff, "residualsUpdate.fits");
            for (int i = 0; i < xDiff.GetLength(0); i++)
                for (int j = 0; j < xDiff.GetLength(1); j++)
                {
                    residuals[i, j] -= xDiff[i, j];
                    xDiff[i, j] = 0;
                }
            var l2Penalty = NaiveGreedyCD.CalcDataObjective(residuals);
            Console.WriteLine("-------------------------");
            Console.WriteLine((l2Penalty + elasticNet));
            var io = System.IO.File.AppendText("penalty" + yBlockSize + ".txt");
            io.WriteLine("l2: " + l2Penalty + "\telastic: " + elasticNet + "\t " + (l2Penalty + elasticNet));
            io.Close();
            Console.WriteLine("-------------------------");

            return false;
        }


        public static Tuple<int, int, double>[] GetMaxBlocks(float[,] bMap, float[,] xImage, double lipschitz, float lambda, float alpha, int yBlockSize, int xBlockSize, int theta)
        {
            var yBlocks = bMap.GetLength(0) / yBlockSize;
            var xBlocks = bMap.GetLength(1) / xBlockSize;

            var tmp = new List<Tuple<int, int, double>>(yBlocks * xBlocks);
            for (int i = 0; i < yBlocks; i++)
                for (int j = 0; j < xBlocks; j++)
                {
                    int yIdx = i * yBlockSize;
                    int xIdx = j * xBlockSize;
                    var sum = 0.0;
                    for (int y = yIdx; y < yIdx + yBlockSize; y++)
                        for (int x = xIdx; x < xIdx + xBlockSize; x++)
                        {
                            var opt = bMap[y, x] / lipschitz;
                            var shrink = ShrinkElasticNet(xImage[y, x] + opt, lambda, alpha);
                            sum += Math.Abs(shrink - xImage[y, x]);
                        }
                    
                }
            tmp.Sort((x, y) => x.Item3.CompareTo(y.Item3));
            var output = new Tuple<int, int, double>[theta];
            for (int i = 0; i < theta; i++)
                output[i] = tmp[tmp.Count - i];
            return output;
        }

    }
}
