using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    class ApproxParallel
    {
        const float ACTIVE_SET_CUTOFF = 1e-8f;
        const int MAX_ACTIVESET_ITER = 1000;
        int yBlockSize;
        int xBlockSize;
        int degreeOfSeperability;
        int tau;

        /// <summary>
        /// Calculate estimated seperability overapproximation (ESO). Or: When we optimize two random pixels, how much do we expect the two values to correlate.
        /// We use tau-nice sampling. I.E. take tau (unique) blocks uniformly at random.
        /// </summary>
        /// <param name="tau"></param>
        /// <param name="degreeOfSep"></param>
        /// <param name="blockCount"></param>
        /// <returns></returns>
        private static double CalcESO(int tau, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1)));


        public bool DeconvolveApprox(float[,] xImage, float[,] residuals, float[,] psf, float lambda, float alpha, Random random, int blockSize, int threadCount, int maxIteration = 100, float epsilon = 1e-4f)
        {
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];

            //calculate gradients for each pixel
            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var gExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var gCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];
            var psf2 = PSF.CalcPSFSquared(psf);

            yBlockSize = blockSize;
            xBlockSize = blockSize;
            degreeOfSeperability = CountNonZero(psf);
            tau = threadCount; //number of processors
            var maxLipschitz = (float)PSF.CalcMaxLipschitz(psf);
            var activeSet = GetActiveSet(xExplore, gExplore, lambda, alpha, maxLipschitz);

            var theta = DeconvolveAccelerated(xExplore, xCorrection, gExplore, gCorrection, psf2, ref activeSet, maxLipschitz, lambda, alpha, random, maxIteration, epsilon);

            var theta0 = tau / (float)activeSet.Count;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xCorrection[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];

            var objectives = CalcObjectives(xImage, residuals, psf, xExplore, xCorrection, lambda, alpha);

            //decide whether we take the correction or explore version
            if (objectives.Item2 < objectives.Item1)
            {
                //correction has the lower objective than explore
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                        xImage[i, j] = xCorrection[i, j];
            }
            else
            {
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                        xImage[i, j] = xExplore[i, j];
            }

            return objectives.Item2 < objectives.Item1;
        }

        public float DeconvolveAccelerated(float[,] xExplore, float[,] xCorrection, float[,] gExplore, float[,] gCorrection, float[,] psf2, ref List<Tuple<int, int>> activeSet, float maxLipschitz, float lambda, float alpha, Random random, int maxIteration, float epsilon)
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
            Console.WriteLine("Starting Active Set iterations with " + activeSet.Count + " blocks");
            while (iter < maxIteration & !converged)
            {
                var xDiffMax = new float[tau];
                var innerIterCount = Math.Min(activeSet.Count / tau, MAX_ACTIVESET_ITER / tau);
                for (int inner = 0; inner < innerIterCount; inner++)
                {
                    var stepSize = lipschitz * theta / theta0;
                    var theta2 = theta * theta;
                    var correctionFactor = -(1.0f - theta / theta0) / (theta * theta);
                    var samples = activeSet.Shuffle(random).Take(tau).ToList();

                    //values for calculating the restart parameter
                    var udpateXExplores = new float[tau];
                    var oldXExplores = new float[tau];
                    var newXExplores = new float[tau];
                    var oldXCorrections = new float[tau];
                    
                    Parallel.For(0, tau, (i) =>
                    {
                        udpateXExplores[i] = 0.0f;
                        oldXExplores[i] = 0.0f;
                        newXExplores[i] = 0.0f;
                        oldXCorrections[i] = 0.0f;

                        var blockSample = samples[i];
                        var yOffset = blockSample.Item1 * yBlockSize;
                        var xOffset = blockSample.Item2 * xBlockSize;

                        var blockUpdate = new float[yBlockSize, xBlockSize];
                        var updateAbsSum = 0.0f;
                        for (int y = yOffset; y < (yOffset + yBlockSize); y++)
                            for (int x = xOffset; x < (xOffset + xBlockSize); x++)
                            {
                                var update = theta2 * gCorrection[y, x] + gExplore[y, x] + xExplore[y, x] * stepSize;
                                update = ElasticNet.ProximalOperator(update, stepSize, lambda, alpha) - xExplore[y, x];
                                blockUpdate[y - yOffset, x - xOffset] = update;
                                udpateXExplores[i] += update;
                                updateAbsSum += Math.Abs(update);
                            }

                        //update gradients
                        if(0.0f != updateAbsSum)
                        {
                            xDiffMax[i] = Math.Max(xDiffMax[i], updateAbsSum);
                            UpdateBMaps(blockUpdate, blockSample.Item1, blockSample.Item2, psf2, gExplore, gCorrection, correctionFactor);
                            for (int y = yOffset; y < (yOffset + yBlockSize); y++)
                                for (int x = xOffset; x < (xOffset + xBlockSize); x++)
                                {
                                    var update = blockUpdate[y - yOffset, x - xOffset];
                                    var oldExplore = xExplore[y, x];
                                    var oldCorrection = xCorrection[y, x];

                                    oldXExplores[i] += xExplore[y, x];
                                    oldXCorrections[i] += xCorrection[y, x];
                                    newXExplores[i] += oldExplore + update;
   

                                    xExplore[y, x] += update;
                                    xCorrection[y, x] += update * correctionFactor;
                                }
                        }
                    });

                    //not 100% sure this is the correct generalization from single pixel/single thread rule to block/parallel rule
                    for (int i = 0; i < tau;i++)
                        testRestart = (1.0 - eta) * testRestart - eta * (udpateXExplores[i]) * (newXExplores[i] - (theta * theta * oldXCorrections[i] + oldXExplores[i]));
                    
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

                    Console.WriteLine("restarting");
                    //new active set
                    activeSet = GetActiveSet(xExplore, gExplore, lambda, alpha, maxLipschitz);
                    blockCount = activeSet.Count;
                    theta = tau / (float)blockCount;
                    theta0 = theta;
                    beta = CalcESO(tau, degreeOfSeperability, blockCount);
                    lipschitz = maxLipschitz * yBlockSize * xBlockSize;
                    lipschitz *= (float)beta;
                }

                if (xDiffMax.Sum() < epsilon)
                {
                    converged = true;
                }

                Console.WriteLine("Done Active Set iteration " + iter);
                iter++;
            }

            return theta;
        }

        public static void UpdateBMaps(float[,] updateBlock, int yB, int xB, float[,] psf2, float[,] gExplore, float[,] gCorrection, float correctionFactor)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;
            var yPixelIdx = yB * updateBlock.GetLength(0);
            var xPixelIdx = xB * updateBlock.GetLength(1);

            var yMin = Math.Max(yPixelIdx - yPsf2Half, 0);
            var xMin = Math.Max(xPixelIdx - xPsf2Half, 0);
            var yMax = Math.Min(yPixelIdx + updateBlock.GetLength(0) - yPsf2Half + psf2.GetLength(0), gExplore.GetLength(0));
            var xMax = Math.Min(xPixelIdx + updateBlock.GetLength(1) - xPsf2Half + psf2.GetLength(1), gExplore.GetLength(1));
            for (int globalY = yMin; globalY < yMax; globalY++)
                for (int globalX = xMin; globalX < xMax; globalX++)
                {
                    var exploreUpdate = 0.0f;
                    var correctionUpdate = 0.0f;

                    var yPsfMin = Math.Max(globalY + yPsf2Half - yPixelIdx - updateBlock.GetLength(0) + 1, 0);
                    var xPsfMin = Math.Max(globalX + xPsf2Half - xPixelIdx - updateBlock.GetLength(1) + 1, 0);
                    var yPsfMax = Math.Min(globalY + yPsf2Half - yPixelIdx + 1, psf2.GetLength(0));
                    var xPsfMax = Math.Min(globalX + xPsf2Half - xPixelIdx + 1, psf2.GetLength(1));
                    for (int psfY = yPsfMin; psfY < yPsfMax; psfY++)
                        for (int psfX = xPsfMin; psfX < xPsfMax; psfX++)
                        {
                            var blockY = -1 * (psfY - yPsf2Half - globalY + yPixelIdx);
                            var blockX = -1 * (psfX - xPsf2Half - globalX + xPixelIdx);

                            var update = updateBlock[blockY, blockX] * psf2[psfY, psfX];
                            exploreUpdate += update;
                            correctionUpdate += update * correctionFactor;
                        }

                    gExplore[globalY, globalX] -= exploreUpdate;
                    gCorrection[globalY, globalX] -= correctionUpdate;
                }
        }

        private List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, float lambda, float alpha, float lipschitz)
        {
            var debug = new float[xExplore.GetLength(0), xExplore.GetLength(1)];
            var output = new List<Tuple<int, int>>();
            for (int i = 0; i < xExplore.GetLength(0) / yBlockSize; i++)
                for (int j = 0; j < xExplore.GetLength(1) / xBlockSize; j++)
                {
                    var yPixel = i * yBlockSize;
                    var xPixel = j * xBlockSize;
                    var nonZero = false;
                    for (int y = yPixel; y < yPixel + yBlockSize; y++)
                        for (int x = xPixel; x < xPixel + xBlockSize; x++)
                        {
                            var tmp = gExplore[y, x] + xExplore[y, x] * lipschitz;
                            tmp = ElasticNet.ProximalOperator(tmp, lipschitz, lambda, alpha);
                            if (ACTIVE_SET_CUTOFF < Math.Abs(tmp - xExplore[y, x]))
                            {
                                nonZero = true;
                            }

                        }

                    if (nonZero)
                    {
                        output.Add(new Tuple<int, int>(i, j));
                        for (int y = yPixel; y < yPixel + yBlockSize; y++)
                            for (int x = xPixel; x < xPixel + xBlockSize; x++)
                                debug[y, x] = 1.0f;
                    }

                }
            FitsIO.Write(debug, "activeSet.fits");
            //can write max change for convergence purposes
            return output;
        }

        private static Tuple<double,double> CalcObjectives(float[,] xImage, float[,] residuals, float[,] psf, float[,] xExplore, float[,] xAccelerated, float lambda, float alpha)
        {
            Tuple<double, double> output = null;

            var CONVKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            using(var residualsCalculator = new PaddedConvolver(CONVKernel, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1))))
            {
                var residualsExplore = new float[xImage.GetLength(0), xImage.GetLength(1)];
                var residualsAccelerated = new float[xImage.GetLength(0), xImage.GetLength(1)];
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                    {
                        residualsExplore[i, j] = xExplore[i, j] - xImage[i, j];
                        residualsAccelerated[i, j] = xAccelerated[i, j] - xImage[i, j];
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
                var objectiveAccelerated = Residuals.CalcPenalty(residualsAccelerated) + ElasticNet.CalcPenalty(xAccelerated, lambda, alpha);

                output = new Tuple<double, double>(objectiveExplore, objectiveAccelerated);
            }

            return output;
        }
    }
}
