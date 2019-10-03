using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.ToyImplementations
{
    class ApproxSingle
    {
        public class ApproxInfo
        {
            public bool CorrectionUsed;
            
        }

        public static bool DeconvolveRandom2(float[,] xImage, float[,] residuals, float[,] psf, float lambda, float alpha, Random random, int blockSize, int threadCount, int maxIteration = 100, double epsilon = 1e-4)
        {
            var xImageExplore = xImage;
            var xImageCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];

            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var bMapExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var bMapCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];

            var psf2 = PSF.CalcPSFSquared(psf);

            var yBlockSize = blockSize;
            var xBlockSize = blockSize;

            var tau = 1; //theta, also number of processors.
            var degreeOfSep = CountNonZero(psf);
            var blockCount = xImage.Length / (yBlockSize * xBlockSize);
            var beta = 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling. Look at the original PCDM Paper for the explanation
            //Theta-nice sampling: take theta number of random pixels 

            var lipschitz = (float)PSF.CalcMaxLipschitz(psf) * yBlockSize * xBlockSize;
            lipschitz *= (float)beta;
            var theta = tau / (float)blockCount;
            var theta0 = theta;
            var eta = 1.0 / blockCount;

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
                            var update = theta2 * bMapCorrection[y, x] + bMapExplore[y, x] + xImageExplore[y, x] * stepSize;
                            update = ElasticNet.ProximalOperator(update, stepSize, lambda, alpha) - xImageExplore[y, x];
                            blocks[i, y, x] = update;
                            if (update != 0.0)
                                containsNonZero[i] = true;
                        }

                    /*
                    RandomBlockCD2.AddInto(xImageExplore, xEUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(bEUpdate, xEUpdate, yBlock, xBlock, yBlockSize, xBlockSize);

                    var xCUpdate = -(1.0 - theta / theta0) / (theta * theta) * xEUpdate;
                    RandomBlockCD2.AddInto(xImageCorrection, xCUpdate, yBlock, xBlock, yBlockSize, xBlockSize);
                    RandomBlockCD2.AddInto(bCUpdate, xCUpdate, yBlock, xBlock, yBlockSize, xBlockSize);*/


                    
                }

                //update bMaps
                var correctionFactor = -(1.0 - theta / theta0) / (theta * theta);
                for (int i = 0; i < blockSamples.Length; i++)
                    if(containsNonZero[i])
                    {
                        for (int y = 0; y < yBlockSize; y++)
                            for (int x = 0; x < xBlockSize; x++)
                            {
                                //xCorrectionUpdate[y, x] = 
                            }
                        var yBlock = blockSamples[i] / (xImage.GetLength(1) / xBlockSize);
                        var xBlock = blockSamples[i] % (xImage.GetLength(1) / xBlockSize);
                        

                        //testRestart = (1.0 - eta) * testRestart - eta * (xEUpdate) * (xNew - (theta * theta * xC + xE));
                    }

                if (testRestart > 0)
                {
                    Console.Write("hello");
                    break;
                }

                theta = (float)(Math.Sqrt((theta * theta * theta * theta) + 4 * (theta * theta)) - theta * theta) / 2.0f;
                iter++;
            }

            return false;
        }

        public static void UpdateBMaps(int blockId, float[,,] blocks, int yB, int xB, float[,] psf2, float[,] bE, float[,] bC, float corrFactor)
        {
            var yPsf2Half = psf2.GetLength(0) / 2;
            var xPsf2Half = psf2.GetLength(1) / 2;
            var yPixelIdx = yB * blocks.GetLength(1);
            var xPixelIdx = xB * blocks.GetLength(2);

            var yMin = Math.Max(yPixelIdx - yPsf2Half, 0);
            var xMin = Math.Max(xPixelIdx - xPsf2Half, 0);
            //TODO: debug index madness
            var yMax = Math.Min(yPixelIdx + blocks.GetLength(1) - yPsf2Half + psf2.GetLength(0), bE.GetLength(0));
            var xMax = Math.Min(xPixelIdx + blocks.GetLength(2) - xPsf2Half + psf2.GetLength(1), bE.GetLength(1));
            for (int globalY = yMin; globalY < yMax; globalY++)
                for (int globalX = xMin; globalX < xMax; globalX++)
                {
                    var exploreUpdate = 0.0f;
                    var correctionUpdate = 0.0f;

                    var yPsfMin = Math.Max(globalY + yPsf2Half - yPixelIdx - blocks.GetLength(1) + 1, 0);
                    var xPsfMin = Math.Max(globalX + xPsf2Half - xPixelIdx - blocks.GetLength(2) + 1, 0);
                    var yPsfMax = Math.Min(globalY + yPsf2Half - yPixelIdx + 1, psf2.GetLength(0));
                    var xPsfMax = Math.Min(globalX + xPsf2Half - xPixelIdx + 1, psf2.GetLength(1));
                    var xPsfIdx = globalX + xPsf2Half - yPixelIdx;
                    for (int psfY = yPsfMin; psfY < yPsfMax; psfY++)
                        for (int psfX = xPsfMin; psfX < xPsfMax; psfX++)
                        {
                            var blockY = -1 * (psfY - yPsf2Half - globalY) ;
                            var blockX = -1 * (psfX - xPsf2Half - globalX);

                            var b = blocks[blockId, blockY, blockX];
                            var p = psf2[psfY, psfX];
                            var update = blocks[blockId, blockY, blockX] * psf2[psfY, psfX];
                            exploreUpdate += update;
                            correctionUpdate += update * corrFactor; 
                        }
                    
                    bE[globalY, globalX] += exploreUpdate;
                    bC[globalY, globalX] += correctionUpdate;
                }



        }

    }
}
