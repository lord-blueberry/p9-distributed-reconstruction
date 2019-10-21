using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class ApproxFast : IDeconvolver
    {
        private int threadCount;

        const float ACTIVE_SET_CUTOFF = 1e-8f;
        int yBlockSize;
        int xBlockSize;
        int degreeOfSeperability;
        int tau;

        public ApproxFast(int blockSize, int threadCount)
        {

        }

        

        #region IDeconvolver implementation
        public DeconvolutionResult Deconvolve(float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 0.0001F)
        {
            throw new NotImplementedException();
        }

        public DeconvolutionResult DeconvolvePath(float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001F)
        {
            throw new NotImplementedException();
        }
        #endregion


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

            var theta = 0.0f;//DeconvolveRandomActiveSet(xExplore, xCorrection, gExplore, gCorrection, psf2, ref activeSet, maxLipschitz, lambda, alpha, random, maxIteration, epsilon);

            //decide which version should be taken#
            var CONVKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var residualsCalculator = new PaddedConvolver(CONVKernel, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var theta0 = tau / (float)activeSet.Count;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;

            //calculate residuals
            var residualsExplore = Copy(xExplore);
            var residualsAccelerated = Copy(xExplore);
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                {
                    residualsExplore[i, j] -= xImage[i, j];
                    residualsAccelerated[i, j] += tmpTheta * xCorrection[i, j] - xImage[i, j];
                    xCorrection[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];
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

            if (objectiveAcc < objectiveExplore)
            {
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

            return objectiveAcc < objectiveExplore;
        }

        private class ApproxParams
        {
            public int YBlockSize { get; private set; }
            public int XBlockSize { get; private set; }
            public int DegreeOfSeperability { get; private set; }
            public int ThreadCount { get; private set; }
        }

        private class ApproxConcurrentThread
        {
            private readonly ApproxParams p;

            private float lambda;
            private float alpha;
            private readonly float[,] psf2;

            private readonly float[,] xExpl;
            private readonly float[,] xCorr;
            private readonly float[,] bExpl;
            private readonly float[,] bCorr;
            private List<Tuple<int, int>> activeSet;
            private int[] blockLock;
            private float restart; //not a reference, here is a problem

            private float[,] blockUpdate;


            public ApproxConcurrentThread(ApproxParams p, float lambda, float alpha, float[,] psf2, ref float[,] xExpl, ref float[,] xCorr, ref float[,] bExpl, ref float[,] bCorr, ref List<Tuple<int,int>> activeSet, ref int[] blockLock, ref float restart)
            {
                this.p = p;
                this.lambda = lambda;
                this.alpha = alpha;
                this.psf2 = psf2;

                this.xExpl = xExpl;
                this.xCorr = xCorr;
                this.bExpl = bExpl;
                this.bCorr = bCorr;
                this.activeSet = activeSet;
                this.blockLock = blockLock;
                this.restart = restart;
            }

            public void Run()
            {
                //get block
                var random = new Random();
                var block = GetRandomBlock(random, activeSet, blockLock);

                //calc update concurrently
                //update b
                //update restart parameter
                //update theta
                //do until iteration reached
                //unlockBlock
            }

            private static Tuple<int, int> GetRandomBlock(Random rand, List<Tuple<int, int>> activeSet, int[] blockLock)
            {
                var succesfulLock = false;
                var idx = -1;
                do
                {
                    idx = rand.Next(0, blockLock.Length);
                    var old = Interlocked.CompareExchange(ref blockLock[idx], 1, 0);
                    succesfulLock = old == 0;
                } while (!succesfulLock);

                return activeSet[idx];
            }

            private void UnlockBlock(Tuple<int,int> block)
            {

            }

            private void UpdateBMaps(int blockId, float[,,] blocks, int yB, int xB, float[,] psf2, float[,] gExplore, float[,] gCorrection, float correctionFactor)
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
                                var blockY = -1 * (psfY - yPsf2Half - globalY + yPixelIdx);
                                var blockX = -1 * (psfX - xPsf2Half - globalX + xPixelIdx);

                                var update = blocks[blockId, blockY, blockX] * psf2[psfY, psfX];
                                exploreUpdate += update;
                                correctionUpdate += update * correctionFactor;
                            }

                        ConcurrentSubtract(gExplore, globalY, globalX, exploreUpdate);
                        ConcurrentSubtract(gCorrection, globalY, globalX, correctionUpdate);
                        //gExplore[globalY, globalX] -= exploreUpdate;
                        //gCorrection[globalY, globalX] -= correctionUpdate;
                    }
            }

            private static void ConcurrentSubtract(float[,] map, int yIdx, int xIdx, float value)
            {
                var successfulWrite = value == 0.0f;    //skip write if value to subtract is zero
                while (!successfulWrite)
                {
                    var read = map[yIdx, xIdx];
                    var old = Interlocked.CompareExchange(ref map[yIdx, xIdx], read - value, read);
                    successfulWrite = old == read;
                } 
            }
        }
    }
}
