using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    class ApproxSingle
    {
        int yBlockSize;
        int xBlockSize;
        int degreeOfSeperability;
        int tau;

        public class ApproxInfo
        {
            public bool CorrectionUsed;
            
        }

        public bool DeconvolveRandom(float[,] xImage, float[,] residuals, float[,] psf, float lambda, float alpha, Random random, int blockSize, int threadCount, int maxIteration = 100, double epsilon = 1e-4)
        {
            var xExplore = xImage;
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];

            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));

            //calculate gradients for each pixel
            var gExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var gCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];

            var psf2 = PSF.CalcPSFSquared(psf);

            yBlockSize = blockSize;
            xBlockSize = blockSize;
            degreeOfSeperability = CountNonZero(psf);
            tau = 1; //number of processors.
            
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = CalcESO(tau, degreeOfSeperability, blockCount);

            var lipschitz = (float)PSF.CalcMaxLipschitz(psf) * yBlockSize * xBlockSize;
            lipschitz *= (float)beta;
            var theta = tau / (float)blockCount;
            var theta0 = theta;
            float eta = 1.0f / blockCount;

            var testRestart = 0.0;
            var iter = 0;
            var blocks = new float[tau, yBlockSize, xBlockSize];
            var containsNonZero = new bool[tau];
            while (iter < maxIteration)
            {
                var stepSize = lipschitz * theta / theta0;
                var theta2 = theta * theta;
                var blockSamples = RandomCD.CreateSamples(blockCount, tau, random);

                //minimize blocks
                for (int i = 0; i < blockSamples.Length; i++)
                {
                    var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                    var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);

                    containsNonZero[i] = false;
                    for (int y = yBlock * yBlockSize; y < (yBlock * yBlockSize + yBlockSize); y++)
                        for (int x = xBlock * xBlockSize; x < (xBlock * xBlockSize + xBlockSize); x++)
                        {
                            var update = theta2 * gCorrection[y, x] + gExplore[y, x] + xExplore[y, x] * stepSize;
                            update = ElasticNet.ProximalOperator(update, stepSize, lambda, alpha) - xExplore[y, x];
                            blocks[i, y, x] = update;
                            if (update != 0.0)
                                containsNonZero[i] = true;
                        }
                }

                //update bMaps
                var correctionFactor = -(1.0f - theta / theta0) / (theta * theta);
                for (int i = 0; i < blockSamples.Length; i++)
                    if(containsNonZero[i])
                    {
                        var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                        var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);
                        UpdateBMaps(i, blocks, yBlock, xBlock, psf2, gExplore, gCorrection, correctionFactor);

                        //update reconstructed image
                        var yOffset = yBlock * yBlockSize;
                        var xOffset = xBlock * xBlockSize;
                        for(int y = 0; y < xExplore.GetLength(0); y++)
                            for(int x = 0; x < xExplore.GetLength(1);x++)
                            {
                                var update = blocks[i, y, x];
                                var oldExplore = xExplore[yOffset + y, xOffset + x];
                                var oldCorrection = xCorrection[yOffset + y, xOffset + x];
                                var newValue = oldExplore + update;
                                testRestart = (1.0 - eta) * testRestart - eta * (update) * (newValue - (theta * theta * oldCorrection + oldExplore));

                                xExplore[yOffset + y, xOffset + x] += update;
                                xCorrection[yOffset + y, xOffset + x] += update * correctionFactor;
                            }
                    }

                if (testRestart > 0)
                {
                    //restart acceleration
                    var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
                    for (int y = 0; y < xExplore.GetLength(0); y++)
                        for (int x = 0; x < xExplore.GetLength(1); x++)
                        {
                            xExplore[y, x] +=  tmpTheta* xCorrection[y, x];
                            xCorrection[y, x] = 0;
                            gExplore[y, x] += tmpTheta * gCorrection[y, x];
                            gCorrection[y, x] = 0;
                        }
                    theta = theta0;
                }

                theta = (float)(Math.Sqrt((theta * theta * theta * theta) + 4 * (theta * theta)) - theta * theta) / 2.0f;
                iter++;
            }

            return false;
        }

        public bool DeconvolveActiveSet(float[,] xImage, float[,] residuals, float[,] psf, float lambda, float alpha, Random random, int blockSize, int threadCount, int maxIteration = 100, float epsilon = 1e-4f)
        {
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];

            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));

            //calculate gradients for each pixel
            var gExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var gCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];
            var psf2 = PSF.CalcPSFSquared(psf);

            yBlockSize = blockSize;
            xBlockSize = blockSize;
            degreeOfSeperability = CountNonZero(psf);
            tau = 1; //number of processors
            var maxLipschitz = (float)PSF.CalcMaxLipschitz(psf);
            var activeSet = GetActiveSet(xExplore, gExplore, lambda, alpha, maxLipschitz);
            var theta = DeconvolveRandomActiveSet(xExplore, xCorrection, gExplore, gCorrection, psf2, ref activeSet, maxLipschitz, lambda, alpha, random, maxIteration, epsilon);

            //decide which version should be taken#
            var CONVKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var residualsCalculator = new PaddedConvolver(CONVKernel, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var theta0 = tau / (float)activeSet.Count;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;

            //calculate residuals
            var residualsExplore = Copy(xExplore);
            var residualsAccelerated = Copy(xExplore);
            for(int i = 0; i < xImage.GetLength(0);i++)
                for(int j = 0; j < xImage.GetLength(1);j++)
                {
                    residualsExplore[i, j] -= xImage[i, j];
                    residualsAccelerated[i, j] += tmpTheta * xCorrection[i,j] - xImage[i, j];
                    xCorrection[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i,j];
                }

            residualsCalculator.ConvolveInPlace(residualsExplore);
            residualsCalculator.ConvolveInPlace(residualsAccelerated);
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    residualsExplore[i, j] -= residuals[i, j];
                    residualsAccelerated[i, j] -= residuals[i, j];
                }

            var objectiveExplore = Residuals.CalcPenalty(residualsExplore) + ElasticNet.CalcPenalty(xExplore, lambda, alpha);
            var objectiveAcc = Residuals.CalcPenalty(residualsAccelerated) + ElasticNet.CalcPenalty(xCorrection, lambda, alpha);

            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    if (objectiveAcc < objectiveExplore)
                        xImage[i, j] = xCorrection[i, j];
                    else
                        xImage[i, j] = xExplore[i, j];

            return false;
        }

        public float DeconvolveRandomActiveSet(float[,] xExplore, float[,] xCorrection, float[,]gExplore, float[,] gCorrection, float[,] psf2, ref List<Tuple<int,int>> activeSet, float maxLipschitz, float lambda, float alpha, Random random, int maxIteration, float epsilon)
        {
            var blockCount = activeSet.Count;
            var beta = CalcESO(tau, degreeOfSeperability, blockCount);
            var lipschitz = maxLipschitz * yBlockSize * xBlockSize;
            lipschitz *= (float)beta;

            var theta = tau / (float)blockCount;
            var theta0 = theta;
            float eta = 1.0f / blockCount;

            var testRestart = 0.0;
            var iter = 0;
            var blocks = new float[tau, yBlockSize, xBlockSize];
            var containsNonZero = new bool[tau];
            var converged = false;
            while (iter < maxIteration & !converged)
            {
                var xDiffMax = 0.0f;
                for(int inner= 0; inner < (blockCount/tau); inner++)
                {
                    var stepSize = lipschitz * theta / theta0;
                    var theta2 = theta * theta;
                    var samples = CreateSamples(blockCount, tau, random);

                    //minimize blocks
                    for (int i = 0; i < samples.Length; i++)
                    {
                        var blockSample = activeSet[samples[i]];
                        var yBlock = blockSample.Item1;
                        var xBlock = blockSample.Item2;

                        containsNonZero[i] = false;
                        for (int y = yBlock * yBlockSize; y < (yBlock * yBlockSize + yBlockSize); y++)
                            for (int x = xBlock * xBlockSize; x < (xBlock * xBlockSize + xBlockSize); x++)
                            {
                                var update = theta2 * gCorrection[y, x] + gExplore[y, x] + xExplore[y, x] * stepSize;
                                update = ElasticNet.ProximalOperator(update, stepSize, lambda, alpha) - xExplore[y, x];
                                blocks[i, y, x] = update;
                                if (update != 0.0)
                                    containsNonZero[i] = true;
                            }
                    }

                    //update bMaps
                    var correctionFactor = -(1.0f - theta / theta0) / (theta * theta);
                    for (int i = 0; i < samples.Length; i++)
                        if (containsNonZero[i])
                        {
                            var blockSample = activeSet[samples[i]];
                            var yBlock = blockSample.Item1;
                            var xBlock = blockSample.Item2;
                            UpdateBMaps(i, blocks, yBlock, xBlock, psf2, gExplore, gCorrection, correctionFactor);

                            var currentDiff = 0.0f;
                            //update reconstructed image
                            var yOffset = yBlock * yBlockSize;
                            var xOffset = xBlock * xBlockSize;
                            for (int y = 0; y < xExplore.GetLength(0); y++)
                                for (int x = 0; x < xExplore.GetLength(1); x++)
                                {
                                    var update = blocks[i, y, x];
                                    var oldExplore = xExplore[yOffset + y, xOffset + x];
                                    var oldCorrection = xCorrection[yOffset + y, xOffset + x];
                                    var newValue = oldExplore + update;
                                    testRestart = (1.0 - eta) * testRestart - eta * (update) * (newValue - (theta * theta * oldCorrection + oldExplore));

                                    currentDiff += Math.Abs(update);

                                    xExplore[yOffset + y, xOffset + x] += update;
                                    xCorrection[yOffset + y, xOffset + x] += update * correctionFactor;
                                }
                            xDiffMax = Math.Max(xDiffMax, currentDiff);
                        }

                    theta = (float)(Math.Sqrt((theta * theta * theta * theta) + 4 * (theta * theta)) - theta * theta) / 2.0f;
                }
                
                if (testRestart > 0)
                {
                    //restart acceleration
                    var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
                    for (int y = 0; y < xExplore.GetLength(0); y++)
                        for (int x = 0; x < xExplore.GetLength(1); x++)
                        {
                            xExplore[y, x] += tmpTheta * xCorrection[y, x];
                            xCorrection[y, x] = 0;
                            gExplore[y, x] += tmpTheta * gCorrection[y, x];
                            gCorrection[y, x] = 0;
                        }

                    //new active set
                    activeSet = GetActiveSet(xExplore, gExplore, lambda, alpha, maxLipschitz);
                    blockCount = activeSet.Count;
                    theta = tau / (float)blockCount;
                    theta0 = theta;
                }

                if(xDiffMax < epsilon)
                {
                    converged = true;
                }

                iter++;
            }

            return theta;
        }



        private List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, float lambda, float alpha, float lipschitz)
        {
            var output = new List<Tuple<int, int>>();
            for (int i = 0; i < xExplore.GetLength(0)/yBlockSize; i++)
                for (int j = 0; j < xExplore.GetLength(1)/xBlockSize; i++)
                {
                    var yPixel = i * yBlockSize;
                    var xPixel = j * xBlockSize;
                    var nonZero = false;
                    for(int y = yPixel; y < yPixel+yBlockSize;y++)
                        for(int x = xPixel; x < xPixel+xBlockSize;x++)
                        {
                            var tmp = gExplore[y, x] + xExplore[y, x] * lipschitz;
                            tmp = ElasticNet.ProximalOperator(tmp, lipschitz, lambda, alpha);
                            if (tmp != xExplore[y, x])
                                nonZero = true;
                        }

                    if(nonZero)
                        output.Add(new Tuple<int, int>(i, j));
                }

            return output;
        }

        public static void UpdateBMaps(int blockId, float[,,] blocks, int yB, int xB, float[,] psf2, float[,] gExplore, float[,] gCorrection, float correctionFactor)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;
            var yPixelIdx = yB * blocks.GetLength(1);
            var xPixelIdx = xB * blocks.GetLength(2);

            var yMin = Math.Max(yPixelIdx - yPsf2Half, 0);
            var xMin = Math.Max(xPixelIdx - xPsf2Half, 0);
            var yMax = Math.Min(yPixelIdx + blocks.GetLength(1) - yPsf2Half + psf2.GetLength(0), gExplore.GetLength(0));
            var xMax = Math.Min(xPixelIdx + blocks.GetLength(2) - xPsf2Half + psf2.GetLength(1), gExplore.GetLength(1));
            for (int globalY = yMin; globalY < yMax; globalY++)
                for (int globalX = xMin; globalX < xMax; globalX++)
                {
                    var exploreUpdate = 0.0f;
                    var correctionUpdate = 0.0f;

                    var yPsfMin = Math.Max(globalY + yPsf2Half - yPixelIdx - blocks.GetLength(1) + 1, 0);
                    var xPsfMin = Math.Max(globalX + xPsf2Half - xPixelIdx - blocks.GetLength(2) + 1, 0);
                    var yPsfMax = Math.Min(globalY + yPsf2Half - yPixelIdx + 1, psf2.GetLength(0));
                    var xPsfMax = Math.Min(globalX + xPsf2Half - xPixelIdx + 1, psf2.GetLength(1));
                    for (int psfY = yPsfMin; psfY < yPsfMax; psfY++)
                        for (int psfX = xPsfMin; psfX < xPsfMax; psfX++)
                        {
                            var blockY = -1 * (psfY - yPsf2Half - globalY + yPixelIdx) ;
                            var blockX = -1 * (psfX - xPsf2Half - globalX + xPixelIdx);

                            var update = blocks[blockId, blockY, blockX] * psf2[psfY, psfX];
                            exploreUpdate += update;
                            correctionUpdate += update * correctionFactor; 
                        }
                    
                    gExplore[globalY, globalX] += exploreUpdate;
                    gCorrection[globalY, globalX] += correctionUpdate;
                }
        }

        /// <summary>
        /// Calculate estimated seperability overapproximation (ESO). Or: When we optimize two random pixels, how much do we expect the two values to correlate.
        /// We use tau-nice sampling. I.E. take tau (unique) blocks uniformly at random.
        /// </summary>
        /// <param name="tau"></param>
        /// <param name="degreeOfSep"></param>
        /// <param name="blockCount"></param>
        /// <returns></returns>
        private static double CalcESO(int tau, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1)));

        public static int[] CreateSamples(int length, int sampleCount, Random rand)
        {
            var samples = new HashSet<int>(sampleCount);
            while (samples.Count < sampleCount)
                samples.Add(rand.Next(0, length));
            int[] output = new int[samples.Count];
            samples.CopyTo(output);
            return output;
        }

        public static void DebugCray()
        {
            var bMap = new float[16, 16];
            var bMap2 = new float[16, 16];
            var psf2 = new float[8, 8];
            psf2[4, 4] = 1.0f; psf2[4, 5] = 0.5f; psf2[4, 6] = 0.5f;
            var blocks = new float[1, 2, 2];
            blocks[0, 0, 0] = 1.0f;
            blocks[0, 0, 1] = 2.0f;
            blocks[0, 1, 0] = 2.0f;
            blocks[0, 1, 1] = 3.0f;



            UpdateBMaps(0, blocks, 7, 7, psf2, bMap, bMap2, 3.0f);

            FitsIO.Write(bMap, "bMapCray.fits");
        }


    }
}
