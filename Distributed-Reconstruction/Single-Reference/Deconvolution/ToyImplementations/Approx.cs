using System;
using System.Collections.Generic;
using System.Text;
using static Single_Reference.Common;
using System.IO;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    class Approx
    {
        public static bool DeconvolveRandom(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, Random random, int blockSize, int maxIteration = 100, double epsilon = 1e-4)
        {
            
            var xImage2 = ToFloatImage(xImage);
            var uImage = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var gradientUpdate = new float[xImage.GetLength(0), xImage.GetLength(1)];

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

            
            var startL2 = NaiveGreedyCD.CalcDataObjective(residuals);

            var tau = 1; //theta, also number of processors.
            var degreeOfSep = RandomCD.CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = RandomBlockCD2.ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            lipschitz *= beta;
            lambda = lambda / (yBlockSize * xBlockSize * beta);

            var omega = tau / (float)blockCount;

            var iter = 0;
            while (iter < maxIteration)
            {
                bool containsNonZero = false;
                var blockSamples = RandomCD.CreateSamples(blockCount, tau, random);
                for (int i = 0; i < blockSamples.Length; i++)
                {
                    var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                    var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);

                    var block = RandomBlockCD2.CopyFrom(bMap, yBlock, xBlock, yBlockSize, xBlockSize);

                    var update = block / (lipschitz * omega * blockCount/(double)(tau));

                    var xOld = RandomBlockCD2.CopyFrom(xImage2, yBlock, xBlock, yBlockSize, xBlockSize);
                    var xNew = xOld + update;

                    //shrink
                    for (int j = 0; j < xNew.Count; j++)
                    {
                        xNew[j] = CommonDeprecated.ShrinkElasticNet(xNew[j], lambda, alpha);
                        containsNonZero |= (xNew[j] - xOld[j]) != 0.0;
                    }

                    var xUpdate = xNew - xOld;
                    RandomBlockCD2.AddInto(xImage2, xUpdate, yBlock, xBlock, yBlockSize, xBlockSize);

                    var uUpdate = -(1.0 - blockCount / tau * omega) / (omega * omega) * xUpdate;
                    RandomBlockCD2.AddInto(uImage, uUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    
                    RandomBlockCD2.AddInto(gradientUpdate, xUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(gradientUpdate, omega*omega*uUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                }
                omega = (float)(Math.Sqrt((omega * omega * omega * omega) + 4 * (omega * omega)) - omega * omega) / 2.0f;


                if (containsNonZero)
                {
                    //FitsIO.Write(xImage2, "xImageBlock.fits");
                    //FitsIO.Write(xDiff, "xDiff.fits");

                    //update b-map
                    bMapUpdateCalculator.ConvolveInPlace(gradientUpdate);
                    //FitsIO.Write(xDiff, "bMapUpdate.fits");
                    for (int i = 0; i < gradientUpdate.GetLength(0); i++)
                        for (int j = 0; j < gradientUpdate.GetLength(1); j++)
                        {
                            bMap[i, j] -= gradientUpdate[i, j];
                            gradientUpdate[i, j] = 0;
                        }
                }
                iter++;
            }

            var elasticNet = 0.0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    gradientUpdate[i, j] = xImage2[i, j] - (float)xImage[i, j];
                    xImage[i, j] = xImage2[i, j];

                    elasticNet += lambda * 2 * lipschitz * GreedyBlockCD.ElasticNetPenalty(xImage2[i, j], (float)alpha);
                }


            resUpdateCalculator.ConvolveInPlace(gradientUpdate);
            //FitsIO.Write(xDiff, "residualsUpdate.fits");
            for (int i = 0; i < gradientUpdate.GetLength(0); i++)
                for (int j = 0; j < gradientUpdate.GetLength(1); j++)
                {
                    residuals[i, j] -= gradientUpdate[i, j];
                    gradientUpdate[i, j] = 0;
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
                    xImage[i, j] = (omega*omega)*uImage[i, j] + xImage2[i, j];
            return false;
        }

        public static bool DeconvolveRandom2(double[,] xImage, double[,] residuals, double[,] psf, double lambda, double alpha, Random random, int blockSize, StreamWriter writer, FastGreedyCD fastCD, int maxIteration = 100, double epsilon = 1e-4)
        {
            var lambdaOriginal = lambda;
            var xImage2 = ToFloatImage(xImage);
            var xImageExplore = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var xImageCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var bEUpdate = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var bCUpdate = new float[xImage.GetLength(0), xImage.GetLength(1)];

            for(int i = 0; i < xImage.GetLength(0);i++)
                for(int j = 0; j < xImage.GetLength(1);j++)
                    xImageExplore[i, j] = xImage2[i, j];

            var PSFConvolution = CommonDeprecated.PSF.CalcPaddedFourierConvolution(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psf, residuals.GetLength(0), residuals.GetLength(1));
            var PSFSquared = Fourier2D.Multiply(PSFConvolution, PSFCorrelation);
            var bMapCalculator = new PaddedConvolver(PSFCorrelation, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var bMapUpdateCalculator = new PaddedConvolver(PSFSquared, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));

            var yBlockSize = blockSize;
            var xBlockSize = blockSize;

            var bMapExplore = ToFloatImage(residuals);
            bMapCalculator.ConvolveInPlace(bMapExplore);
            var bMapCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];


            var tau = 1; //theta, also number of processors.
            var degreeOfSep = RandomCD.CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = RandomBlockCD2.ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            lipschitz *= beta;
            lambda = lambda / (yBlockSize * xBlockSize * beta);

            var theta = tau / (float)blockCount;
            var theta0 = theta;

            var testRestart = 0.0;
            var iter = 0;
            var test = 0.0;
            while (iter < maxIteration)
            {
                bool containsNonZero = false;
                var blockSamples = RandomCD.CreateSamples(blockCount, tau, random);
                for (int i = 0; i < blockSamples.Length; i++)
                {
                    var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                    var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);

                    var xE = RandomBlockCD2.CopyFrom(xImageExplore, yBlock, xBlock, yBlockSize, xBlockSize);
                    var xC = RandomBlockCD2.CopyFrom(xImageCorrection, yBlock, xBlock, yBlockSize, xBlockSize);
                    var bE = RandomBlockCD2.CopyFrom(bMapExplore, yBlock, xBlock, yBlockSize, xBlockSize);
                    var bC = RandomBlockCD2.CopyFrom(bMapCorrection, yBlock, xBlock, yBlockSize, xBlockSize);
                    test = theta / theta0;
                    var stepSize = lipschitz * theta / theta0;
                    // real:  var xNew = theta * theta * bC / stepSize + bE / stepSize + xE;
                    //var xNew = theta * theta * bC / stepSize + bE / stepSize + (xE + xC * theta * theta);
                    var xNew = theta * theta * bC / stepSize + bE / stepSize + xE;

                    //shrink
                    for (int j = 0; j < xNew.Count; j++)
                    {
                        //THIS IS WRONG: TODO: Actual proximal operator that does not cheekily decrease lambda with each iteration. As theta goes to zero, so does lambda
                        xNew[j] = CommonDeprecated.ShrinkElasticNet(xNew[j], lambda, alpha);
                        containsNonZero |= (xNew[j] - xE[j]) != 0.0;
                    }
                    var xEUpdate = xNew - xE;

                    RandomBlockCD2.AddInto(xImageExplore, xEUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(bEUpdate, xEUpdate, yBlock, xBlock, yBlockSize, xBlockSize);

                    var xCUpdate = -(1.0 - theta / theta0) / (theta * theta) * xEUpdate;
                    RandomBlockCD2.AddInto(xImageCorrection, xCUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(bCUpdate, xCUpdate, yBlock, xBlock, yBlockSize, xBlockSize);

                    var eta = 1.0 / blockCount;
                    testRestart = (1.0 - eta) * testRestart - eta * (xEUpdate) * (xNew - (theta * theta * xC + xE));
                }
                
                if (containsNonZero)
                {
                    //FitsIO.Write(xImage2, "xImageBlock.fits");
                    //FitsIO.Write(xDiff, "xDiff.fits");
                    bMapUpdateCalculator.ConvolveInPlace(bEUpdate);
                    bMapUpdateCalculator.ConvolveInPlace(bCUpdate);

                    for (int i = 0; i < xImage2.GetLength(0); i++)
                        for (int j = 0; j < xImage2.GetLength(1); j++)
                        {
                            bMapExplore[i, j] -= bEUpdate[i, j];
                            bMapCorrection[i, j] -= bCUpdate[i, j];
                            bEUpdate[i, j] = 0;
                            bCUpdate[i, j] = 0;
                        }

                }
                if (testRestart > 0)
                {
                    Console.Write("hello");
                    break;
                }
                    
                theta = (float)(Math.Sqrt((theta * theta * theta * theta) + 4 * (theta * theta)) - theta * theta) / 2.0f;
                iter++;
            }
            FitsIO.Write(bMapExplore, "bMapExplore.fits");
            FitsIO.Write(bMapCorrection, "bMapCorr.fits");
            FitsIO.Write(xImageExplore, "xExplore.fits");
            FitsIO.Write(xImageCorrection, "xCorr.fits");

            var xImageAcc = xImage2;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    xImageAcc[i, j] = Math.Max(tmpTheta * xImageCorrection[i, j] + xImageExplore[i, j], 0);
                    xImageCorrection[i, j] = tmpTheta * xImageCorrection[i, j];
                }
                    
            FitsIO.Write(xImageCorrection, "xCorrTheta.fits");

            var residualsExplore = bMapExplore;
            var residualsAcc = bMapCorrection;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    residualsExplore[i, j] = (float)(xImageExplore[i, j] - xImage[i, j]);
                    residualsAcc[i, j] = (float)(xImageAcc[i, j] - xImage[i, j]);
                }
            var residualsCalculator = new PaddedConvolver(CommonDeprecated.PSF.CalcPaddedFourierConvolution(psf, residuals.GetLength(0), residuals.GetLength(1)), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            residualsCalculator.ConvolveInPlace(residualsExplore);
            residualsCalculator.ConvolveInPlace(residualsAcc);

            FitsIO.Write(residualsExplore, "residualsExploreConvolved.fits");
            FitsIO.Write(residualsAcc, "residualsAccConvolved.fits");
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    residualsExplore[i, j] = (float)(residuals[i, j] - residualsExplore[i, j]);
                    residualsAcc[i, j] = (float)(residuals[i, j] - residualsAcc[i, j]);
                }
            FitsIO.Write(residualsExplore, "residualsExplore.fits");
            FitsIO.Write(residualsAcc, "residualsAcc.fits");
            FitsIO.Write(xImage2, "xAccelerated.fits");

            var l2penaltyExplore = FastGreedyCD.CalcDataPenalty(residualsExplore);
            var elasticPenaltyExplore = fastCD.CalcRegularizationPenalty(xImageExplore, (float)lambdaOriginal, (float)alpha);
            var l2PenaltyAcc = FastGreedyCD.CalcDataPenalty(residualsAcc);
            var elasticPenaltyAcc = fastCD.CalcRegularizationPenalty(xImage2, (float)lambdaOriginal, (float)alpha);

            //if (l2PenaltyAcc + elasticPenaltyAcc < l2penaltyExplore + elasticPenaltyExplore)
            if (l2PenaltyAcc + elasticPenaltyAcc < l2penaltyExplore + elasticPenaltyExplore)
            {
                //use accelerated result
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                        xImage[i, j] = xImageAcc[i, j];
            }
            else
            {
                //use non-accelerated result
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                        xImage[i, j] = xImageExplore[i, j];
            }

            var accelerated = l2PenaltyAcc + elasticPenaltyAcc < l2penaltyExplore + elasticPenaltyExplore;
            writer.WriteLine(accelerated + ";" + (l2penaltyExplore + elasticPenaltyExplore)  + ";" + (l2PenaltyAcc + elasticPenaltyAcc) +  ";\t" +l2penaltyExplore + ";" + elasticPenaltyExplore + ";\t" + l2PenaltyAcc + ";" + elasticPenaltyAcc);
            writer.Flush();
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

            var theta = 2;   //theta, also number of processors.
            var degreeOfSep = RandomCD.CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1.0) * (theta - 1.0) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = RandomBlockCD2.ApproximateLipschitz(psf, yBlockSize, xBlockSize);
            lipschitz *= beta;
            lambda = lambda / (yBlockSize * xBlockSize * beta);

            var iter = 0;
            while (iter < maxIteration)
            {
                bool containsNonZero = false;
                var maxBlocks = GetMaxBlocks(bMap, xImage2, lipschitz, (float)lambda, (float)alpha, yBlockSize, xBlockSize, theta);
                foreach (var b in maxBlocks)
                {
                    var yBlock = b.Item1;
                    var xBlock = b.Item2;
                    var block = RandomBlockCD2.CopyFrom(bMap, yBlock, xBlock, yBlockSize, xBlockSize);

                    var update = block / lipschitz;

                    var xOld = RandomBlockCD2.CopyFrom(xImage2, yBlock, xBlock, yBlockSize, xBlockSize);
                    var optimized = xOld + update;

                    //shrink
                    for (int j = 0; j < optimized.Count; j++)
                    {
                        optimized[j] = CommonDeprecated.ShrinkElasticNet(optimized[j], lambda, alpha);
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
                            var shrink = CommonDeprecated.ShrinkElasticNet(xImage[y, x] + opt, lambda, alpha);
                            sum += Math.Abs(shrink - xImage[y, x]);
                        }
                    tmp.Add(new Tuple<int, int, double>(i, j, sum));
                }
            tmp.Sort((x, y) => x.Item3.CompareTo(y.Item3));
            var output = new Tuple<int, int, double>[theta];
            for (int i = 0; i < theta; i++)
                output[i] = tmp[tmp.Count - i - 1];
            return output;
        }

    }
}
