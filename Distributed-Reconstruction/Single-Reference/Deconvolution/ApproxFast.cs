using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class ApproxFast 
    {

        private static double CalcESO(int processorCount, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (processorCount - 1.0) / (Math.Max(1.0, (blockCount - 1)));

        const float ACTIVE_SET_CUTOFF = 1e-8f;
        bool useCDColdStart = true;
        bool useAcceleration = true;
        int threadCount;
        int blockSize;


        public ApproxFast(int threadCount, int blockSize, bool useCDColdStart, bool useAcceleration = true)
        {
            this.threadCount = threadCount;
            this.blockSize = blockSize;
            this.useCDColdStart = useCDColdStart;
            this.useAcceleration = useAcceleration;
        }

        public bool Deconvolve(float[,] xImage, float[,] residuals, float[,] psf, float lambda, float alpha, Random random, int blockSize, int threadCount, int maxIteration = 100, float epsilon = 1e-4f)
        {
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var coldStart = true;
            for(int i = 0; i < xExplore.GetLength(0); i++)
                for(int j = 0; j < xExplore.GetLength(1); j++)
                    if (xExplore[i, j] != 0.0f)
                    {
                        coldStart = false;
                        break;
                    }
                        
            //calculate gradients for each pixel
            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var gExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var gCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];
            var psf2 = PSF.CalcPSFSquared(psf);
            var totalSize = new Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));

            if (coldStart & useCDColdStart)
            {
                var fastCD = new FastGreedyCD(totalSize, totalSize, psf, psf2);
                fastCD.Deconvolve(xExplore, gExplore, lambda, alpha, xImage.GetLength(0));
            }

            var maxLipschitz = (float)PSF.CalcMaxLipschitz(psf);

            var shared = new SharedData(lambda, alpha, blockSize, blockSize, threadCount,
                CountNonZero(psf), psf2, PSF.CalcAMap(psf, totalSize),
                xExplore, xCorrection, gExplore, gCorrection, random);
            shared.ActiveSet = GetActiveSet(xExplore, gExplore, shared.YBlockSize, shared.XBlockSize, lambda, alpha, maxLipschitz);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.maxLipschitz = maxLipschitz;
            shared.MaxConcurrentIterations = 1000;

            var theta = DeconvolveConcurrent(shared, maxIteration, epsilon);

            var theta0 = shared.ProcessorCount / (float)shared.ActiveSet.Count;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xCorrection[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];

            var objectives = EstimateObjectives(xImage, residuals, psf, xExplore, xCorrection, lambda, alpha);

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

        private float DeconvolveConcurrent(SharedData shared, int activeSetIterations, float epsilon)
        {
            var blockCount = shared.ActiveSet.Count;
            var theta0 = shared.ProcessorCount / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.testRestart = 0.0f;

            var xDiffs = new float[shared.ProcessorCount];
            var deconvolvers = new List<AsyncDeconvolver>(shared.ProcessorCount);
            for(int i = 0; i < shared.ProcessorCount; i++)
                deconvolvers.Add(new AsyncDeconvolver(shared, theta0));
            
            int iter = 0;
            var converged = false;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {
                //iterations
                Parallel.For(0, deconvolvers.Count, (i) =>
                {
                    xDiffs[i] = deconvolvers[i].Deconvolve(shared.MaxConcurrentIterations, theta0);
                });
                
                if (shared.testRestart > 0.0f)
                {
                    Console.WriteLine("restarting");
                    var currentTheta = deconvolvers[0].theta;
                    var tmpTheta = currentTheta < 1.0f ? ((currentTheta * currentTheta) / (1.0f - currentTheta)) : theta0;
                    for (int y = 0; y < shared.XExpl.GetLength(0); y++)
                        for (int x = 0; x < shared.XExpl.GetLength(1); x++)
                        {
                            shared.XExpl[y, x] += tmpTheta * shared.XCorr[y, x];
                            shared.XCorr[y, x] = 0;
                            shared.GExpl[y, x] += tmpTheta * shared.GCorr[y, x];
                            shared.GCorr[y, x] = 0;
                        }

                    //new active set
                    shared.ActiveSet = GetActiveSet(shared.XExpl, shared.GExpl, shared.YBlockSize, shared.XBlockSize, shared.Lambda, shared.Alpha, shared.maxLipschitz);
                    blockCount = shared.ActiveSet.Count;
                    theta0 = shared.ProcessorCount / (float)blockCount;
                    for (int i = 0; i < deconvolvers.Count; i++)
                        deconvolvers[i].theta = theta0;
                    shared.testRestart = 0.0f;
                }

                if (xDiffs.Sum() < epsilon)
                {
                    converged = true;
                }

                Console.WriteLine("Done Active Set iteration " + iter);
                iter++;
            }

            return deconvolvers[0].theta;
        }

        #region helper methods
        private static Tuple<double, double> EstimateObjectives(float[,] xImage, float[,] residuals, float[,] psf, float[,] xExplore, float[,] xAccelerated, float lambda, float alpha)
        {
            Tuple<double, double> output = null;

            var CONVKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            using (var residualsCalculator = new PaddedConvolver(CONVKernel, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1))))
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
                        residualsExplore[i, j] = residuals[i, j] - residualsExplore[i, j];
                        residualsAccelerated[i, j] = residuals[i, j] - residualsAccelerated[i, j];
                    }

                var objectiveExplore = Residuals.CalcPenalty(residualsExplore) + ElasticNet.CalcPenalty(xExplore, lambda, alpha);
                var objectiveAccelerated = Residuals.CalcPenalty(residualsAccelerated) + ElasticNet.CalcPenalty(xAccelerated, lambda, alpha);

                output = new Tuple<double, double>(objectiveExplore, objectiveAccelerated);
            }

            return output;
        }

        private static List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, int yBlockSize, int xBlockSize, float lambda, float alpha, float lipschitz)
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

            return output;
        }

        private static float GetAbsMax(float[,] xImage, float[,] bMap, float[,] aMap, float lambda, float alpha)
        {
            var maxPixels = new float[xImage.GetLength(0)];
            Parallel.For(0, xImage.GetLength(0), (y) =>
            {
                var yLocal = y;

                var currentMax = 0.0f;
                for (int x = 0; x < xImage.GetLength(1); x++)
                {
                    var xLocal = x;
                    var currentA = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    //var xTmp = old + bMap[y, x] / currentA;
                    //xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
                    var xTmp = ElasticNet.ProximalOperator(old * currentA + bMap[y, x], currentA, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax < Math.Abs(xDiff))
                        currentMax = Math.Abs(xDiff);
                }
                maxPixels[yLocal] = currentMax;
            });

            var maxPixel = 0.0f;
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel < maxPixels[i])
                    maxPixel = maxPixels[i];

            return maxPixel;
        }
        #endregion

        private class SharedData
        {
            public float Lambda { get; set; }
            public float Alpha { get; set; }

            public int YBlockSize { get; set; }
            public int XBlockSize { get; set; }
            public int ProcessorCount { get; private set; }

            public int DegreeOfSeperability { get; private set; }
            public float[,] Psf2 { get; private set; }
            public float[,] AMap { get; private set; }

            public float[,] XExpl { get; private set; }
            public float[,] XCorr { get; private set; }
            public float[,] GExpl { get; private set; }
            public float[,] GCorr { get; private set; }

            public Random Random { get; private set; }
            public List<Tuple<int, int>> ActiveSet { get; set; }
            public int[] BlockLock { get;  set; }

            public int MaxConcurrentIterations { get; set; }
            public float testRestart;
            public float maxLipschitz;

            public SharedData(
                float lambda,
                float alpha,
                int yBlockSize,
                int xBlockSize,
                int processorCount,

                int degreeOfSep,
                float[,] psf2,
                float[,] aMap,

                float[,] xExpl,
                float[,] xCorr,
                float[,] gExpl,
                float[,] gCorr,
                
                Random rand)
            {
                Lambda = lambda;
                Alpha = alpha;
                YBlockSize = yBlockSize;
                XBlockSize = xBlockSize;
                ProcessorCount = processorCount;

                DegreeOfSeperability = degreeOfSep;
                Psf2 = psf2;
                AMap = aMap;

                XExpl = xExpl;
                XCorr = xCorr;
                GExpl = gExpl;
                GCorr = gCorr;
                Random = rand;
            }
        }

        private class AsyncDeconvolver
        {
            private readonly SharedData shared;
            private readonly float[,] blockUpdate;
            private readonly bool useAcceleration;
            public float theta = 0.0f;
            
            public AsyncDeconvolver(SharedData shared, float theta0, bool useAcceleration = true)
            {
                this.shared = shared;
                theta = theta0;
                blockUpdate = new float[shared.YBlockSize, shared.XBlockSize];
                this.useAcceleration = useAcceleration;
            }

            public float Deconvolve(int maxIterations, float theta0)
            {
                var blockCount = shared.XExpl.Length;
                float eta = 1.0f / blockCount;

                var beta = CalcESO(shared.ProcessorCount, shared.DegreeOfSeperability, blockCount);
                var xDiffMax = 0.0f;
                for (int inner = 0; inner < shared.MaxConcurrentIterations; inner++)
                {
                    var blockIdx = GetRandomBlockIdx(shared.Random, shared.BlockLock);
                    var blockSample = shared.ActiveSet[blockIdx];
                    var yOffset = blockSample.Item1 * shared.YBlockSize;
                    var xOffset = blockSample.Item2 * shared.XBlockSize;

                    var blockLipschitz = GetBlockLipschitz(shared.AMap, yOffset, xOffset, shared.YBlockSize, shared.XBlockSize);
                    var step = blockLipschitz * (float)beta * theta / theta0;
                    var theta2 = theta * theta;
                    var correctionFactor = -(1.0f - theta / theta0) / theta2;

                    var updateSum = 0.0f;
                    var updateAbsSum = 0.0f;
                    for (int y = yOffset; y < (yOffset + shared.YBlockSize); y++)
                        for (int x = xOffset; x < (xOffset + shared.XBlockSize); x++)
                        {
                            var xExpl = Thread.VolatileRead(ref shared.XExpl[y, x]);
                            var update = theta2 * Thread.VolatileRead(ref shared.GCorr[y, x]) + Thread.VolatileRead(ref shared.GExpl[y, x]) + xExpl * step;
                            update = ElasticNet.ProximalOperator(update, step, shared.Lambda, shared.Alpha) - xExpl;
                            blockUpdate[y - yOffset, x - xOffset] = update;
                            updateSum = update;
                            updateAbsSum += Math.Abs(update);
                        }

                    //update gradients
                    if (0.0f != updateAbsSum)
                    {
                        xDiffMax = Math.Max(xDiffMax, updateAbsSum);
                        UpdateBMaps(blockUpdate, blockSample.Item1, blockSample.Item2, shared.Psf2, shared.GExpl, shared.GCorr, correctionFactor);
                        var newXExplore = 0.0f;
                        var oldXExplore = 0.0f;
                        var oldXCorr = 0.0f;
                        for (int y = yOffset; y < (yOffset + shared.YBlockSize); y++)
                            for (int x = xOffset; x < (xOffset + shared.XBlockSize); x++)
                            {
                                var update = blockUpdate[y - yOffset, x - xOffset];
                                var oldExplore = shared.XExpl[y, x];    //does not need to be volatile, this index is blocked until this process is finished, and we already made sure with a volatile read that the latest value is in the cache

                                oldXExplore += shared.XExpl[y, x];
                                oldXCorr += Thread.VolatileRead(ref shared.XCorr[y, x]);
                                newXExplore += oldExplore + update;

                                Thread.VolatileWrite(ref shared.XExpl[y, x], shared.XExpl[y, x] + update);
                                Thread.VolatileWrite(ref shared.XCorr[y, x], shared.XCorr[y, x] + update * correctionFactor);
                            }

                        //not 100% sure this is the correct generalization from single pixel thread rule to block rule
                        var testRestartUpdate = (updateSum) * (newXExplore - (theta2 * oldXCorr + oldXExplore));
                        ConcurrentUpdateTestRestart(ref shared.testRestart, eta, testRestartUpdate);
                    }

                    //unlockBlock
                    Thread.VolatileWrite(ref shared.BlockLock[blockIdx], 0);
                    
                    if(useAcceleration)
                        theta = (float)(Math.Sqrt((theta2 * theta2) + 4 * (theta2)) - theta2) / 2.0f;
                }

                return xDiffMax;
            }

            private static int GetRandomBlockIdx(Random rand, int[] blockLock)
            {
                var succesfulLock = false;
                var idx = -1;
                do
                {
                    idx = rand.Next(0, blockLock.Length);
                    var old = Interlocked.CompareExchange(ref blockLock[idx], 1, 0);
                    succesfulLock = old == 0;
                } while (!succesfulLock);

                return idx;
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

                        ConcurrentSubtract(gExplore, globalY, globalX, exploreUpdate);
                        ConcurrentSubtract(gCorrection, globalY, globalX, correctionUpdate);
                    }
            }

            private static void ConcurrentSubtract(float[,] map, int yIdx, int xIdx, float value)
            {
                var successfulWrite = value == 0.0f;    //skip write if value to subtract is zero
                while (!successfulWrite)
                {
                    var read = Thread.VolatileRead(ref map[yIdx, xIdx]);
                    var old = Interlocked.CompareExchange(ref map[yIdx, xIdx], read - value, read);
                    successfulWrite = old == read | !float.IsNormal(old);
                } 
            }

            private static void ConcurrentUpdateTestRestart(ref float test, float eta, float subtract)
            {
                var successfulWrite = subtract == 0.0f;    //skip write if value to subtract is zero
                while (!successfulWrite)
                {
                    var read = Thread.VolatileRead(ref test);
                    var update = (1.0f - eta) * read - subtract;
                    var old = Interlocked.CompareExchange(ref test, update, read);
                    successfulWrite = old == read;
                }
            }

            private static float GetBlockLipschitz(float[,] lipschitzMap, int yOffset, int xOffset, int yBlockSize, int xBlockSize)
            {
                var output = 0.0f;
                for (int y = yOffset; y < yOffset + yBlockSize; y++)
                    for (int x = xOffset; x < xOffset + xBlockSize; x++)
                        output += lipschitzMap[y, x];
                return output;
            }
        }

        #region Testing 
        public class TestingData
        {
            public List<Tuple<int, double, double, double>> Lines;
            public float lastTheta;
            public bool converged;
            public bool usingAccelerated;
            public StreamWriter writer;
            public int lineIdx = 0;
            public int processorCount;
            public int blockSize;

            public TestingData(StreamWriter writer)
            {
                Lines = new List<Tuple<int, double, double, double>>();
                this.writer = writer;
                writer.WriteLine("idx;cycle;seconds;objectiveNormal;objectiveAccelerated;processorCount;blockSize") ;
            }

            public void Write(Tuple<int, double, double, double> line)
            {
                Lines.Add(line);
                writer.WriteLine(lineIdx++ + ";" + line.Item1 + ";" + line.Item2 + ";" + line.Item3 + ";" + line.Item4 + ";" + processorCount + ";" + blockSize);
                writer.Flush();
            }
        }

        public void DeconvolveTest(TestingData data, int cycle, float[,] xImage, float[,] residuals, float[,] psf, float[,] psfFull, float lambda, float alpha, Random random, int maxIteration = 100, float epsilon = 1e-4f)
        {
            var xExplore = Copy(xImage);
            var xCorrection = new float[xImage.GetLength(0), xImage.GetLength(1)];
            var coldStart = true;
            for (int i = 0; i < xExplore.GetLength(0); i++)
                for (int j = 0; j < xExplore.GetLength(1); j++)
                    if (xExplore[i, j] != 0.0f)
                    {
                        coldStart = false;
                        break;
                    }

            //calculate gradients for each pixel
            var PSFCorr = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, residuals.GetLength(0), residuals.GetLength(1)));
            var gExplore = Residuals.CalcBMap(residuals, PSFCorr, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var gCorrection = new float[residuals.GetLength(0), residuals.GetLength(1)];
            var psf2 = PSF.CalcPSFSquared(psf);
            var totalSize = new Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));

            if (coldStart & useCDColdStart)
            {
                var fastCD = new FastGreedyCD(totalSize, totalSize, psf, psf2);
                fastCD.Deconvolve(xExplore, gExplore, lambda, alpha, xImage.GetLength(0)/2);
            }

            var maxLipschitz = (float)PSF.CalcMaxLipschitz(psf);

            var shared = new SharedData(lambda, alpha, blockSize, blockSize, threadCount,
                CountNonZero(psf), psf2, PSF.CalcAMap(psf, totalSize),
                xExplore, xCorrection, gExplore, gCorrection, random);
            shared.ActiveSet = GetActiveSet(xExplore, gExplore, shared.YBlockSize, shared.XBlockSize, lambda, alpha, maxLipschitz);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.maxLipschitz = maxLipschitz;
            shared.MaxConcurrentIterations = 50;

            var objectivesFirst = EstimateObjectives(xImage, residuals, psfFull, shared.XExpl, shared.XExpl, LAMBDA_TEST, ALPHA_TEST);
            var timeOffset = data.Lines.Count == 0 ? 0.0 : data.Lines.Last().Item2;
            var dataPoint = new Tuple<int, double, double, double>(cycle, timeOffset, objectivesFirst.Item1, objectivesFirst.Item2);
            data.Lines.Add(dataPoint);
            var theta = DeconvolveConcurrentTest(data, cycle, timeOffset, shared, maxIteration, epsilon, xImage, residuals, psf, psfFull);

            var theta0 = shared.ProcessorCount / (float)shared.ActiveSet.Count;
            var tmpTheta = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
            for (int i = 0; i < xImage.GetLength(0); i++)
                for (int j = 0; j < xImage.GetLength(1); j++)
                    xCorrection[i, j] = tmpTheta * xCorrection[i, j] + xExplore[i, j];

            var objectives = EstimateObjectives(xImage, residuals, psfFull, xExplore, xCorrection, lambda, alpha);
            var objectives2 = EstimateObjectives(xImage, residuals, psf, xExplore, xCorrection, lambda, alpha);

            if (objectives.Item1 < objectives.Item2 ^ objectives2.Item1 < objectives2.Item2)
                Console.WriteLine("Different objectives");

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

        }

        public static float LAMBDA_TEST;
        public static float ALPHA_TEST;

        private float DeconvolveConcurrentTest(TestingData data, int cycle, double timeOffset, SharedData shared, int activeSetIterations, float epsilon, float[,] xImage, float[,] residuals, float[,] psf, float[,] psfFull)
        {
            var watch = new Stopwatch();

            var blockCount = shared.XExpl.Length / (shared.YBlockSize * shared.XBlockSize); ;
            var theta0 = shared.ProcessorCount / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.testRestart = 0.0f;

            var xDiffs = new float[shared.ProcessorCount];
            var deconvolvers = new List<AsyncDeconvolver>(shared.ProcessorCount);
            for (int i = 0; i < shared.ProcessorCount; i++)
                deconvolvers.Add(new AsyncDeconvolver(shared, theta0, useAcceleration));
               
            int iter = 0;
            var converged = false;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {
                watch.Start();
                //async iterations
                Parallel.For(0, deconvolvers.Count, (i) =>
                {
                    xDiffs[i] = deconvolvers[i].Deconvolve(shared.MaxConcurrentIterations, theta0);
                });

                watch.Stop();


                Console.WriteLine("calculating objective");
                var xAccelerated = new float[xImage.GetLength(0), xImage.GetLength(1)];
                var theta = deconvolvers[0].theta;
                var tmpTheta2 = theta < 1.0f ? ((theta * theta) / (1.0f - theta)) : theta0;
                for (int i = 0; i < xImage.GetLength(0); i++)
                    for (int j = 0; j < xImage.GetLength(1); j++)
                        xAccelerated[i, j] = tmpTheta2 * shared.XCorr[i, j] + shared.XExpl[i, j];
                var objectives = EstimateObjectives(xImage, residuals, psfFull, shared.XExpl, xAccelerated, LAMBDA_TEST, ALPHA_TEST);
                var dataPoint = new Tuple<int, double, double, double>(cycle, timeOffset + watch.Elapsed.TotalSeconds, objectives.Item1, objectives.Item2);
                data.Write(dataPoint);


                if (shared.testRestart > 0.0f)
                {
                    Console.WriteLine("restarting");
                    var currentTheta = deconvolvers[0].theta;
                    var tmpTheta = currentTheta < 1.0f ? ((currentTheta * currentTheta) / (1.0f - currentTheta)) : theta0;
                    for (int y = 0; y < shared.XExpl.GetLength(0); y++)
                        for (int x = 0; x < shared.XExpl.GetLength(1); x++)
                        {
                            shared.XExpl[y, x] += tmpTheta * shared.XCorr[y, x];
                            shared.XCorr[y, x] = 0;
                            shared.GExpl[y, x] += tmpTheta * shared.GCorr[y, x];
                            shared.GCorr[y, x] = 0;
                        }

                    //new active set
                    shared.ActiveSet = GetActiveSet(shared.XExpl, shared.GExpl, shared.YBlockSize, shared.XBlockSize, shared.Lambda, shared.Alpha, shared.maxLipschitz);
                    blockCount = shared.XExpl.Length / (shared.YBlockSize * shared.XBlockSize);
                    theta0 = shared.ProcessorCount / (float)blockCount;
                    for (int i = 0; i < deconvolvers.Count; i++)
                        deconvolvers[i].theta = theta0;
                    shared.testRestart = 0.0f;
                }

                if (xDiffs.Sum() < epsilon)
                {
                    if(GetAbsMax(shared.XExpl, shared.GExpl, shared.AMap, shared.Lambda, shared.Alpha) < epsilon)
                    {
                        converged = true;
                        data.converged = true;
                    }
                }

                Console.WriteLine("Done Active Set iteration " + iter);
                iter++;
            }

            return deconvolvers[0].theta;
        }

        #endregion
    }
}
