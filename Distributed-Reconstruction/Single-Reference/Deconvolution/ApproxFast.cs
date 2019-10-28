using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class ApproxFast : IDeconvolver
    {
        private static double CalcESO(int tau, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (tau - 1.0) / (Math.Max(1.0, (blockCount - 1)));

        const float ACTIVE_SET_CUTOFF = 1e-8f;
        bool useCDStart = true;

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

            if (coldStart & useCDStart)
            {
                var rec = new Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));
                var fastCD = new FastGreedyCD(rec, rec, psf, psf2);
                fastCD.Deconvolve(xExplore, gExplore, lambda, alpha, xImage.GetLength(0));
            }

            var maxLipschitz = (float)PSF.CalcMaxLipschitz(psf);

            var shared = new SharedData(lambda, alpha, blockSize, blockSize, threadCount,
                CountNonZero(psf), psf2, null,
                xExplore, xCorrection, gExplore, gCorrection, random);
            shared.ActiveSet = GetActiveSet(xExplore, gExplore, shared.YBlockSize, shared.XBlockSize, lambda, alpha, maxLipschitz);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.maxLipschitz = maxLipschitz;
            shared.MaxConcurrentIterations = 1000;

            var theta = DeconvolveConcurrent(shared, maxIteration, epsilon);

            var theta0 = shared.Tau / (float)shared.ActiveSet.Count;
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

        private static Tuple<double, double> CalcObjectives(float[,] xImage, float[,] residuals, float[,] psf, float[,] xExplore, float[,] xAccelerated, float lambda, float alpha)
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
                        residualsExplore[i, j] -= residuals[i, j];
                        residualsAccelerated[i, j] -= residuals[i, j];
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

        private float DeconvolveConcurrent(SharedData shared, int activeSetIterations, float epsilon)
        {
            var blockCount = shared.ActiveSet.Count;
            var theta0 = shared.Tau / (float)blockCount;
            float eta = 1.0f / blockCount;
            shared.testRestart = 0.0f;

            var deconvolvers = new List<ConcurrentDeconvolver>(shared.Tau);
            
            for(int i = 0; i < shared.Tau; i++)
            {
                deconvolvers.Add(new ConcurrentDeconvolver(shared, theta0));
            }

            var tasks = new Task[shared.Tau];
            int iter = 0;
            var converged = false;
            Console.WriteLine("Starting Active Set iterations with " + shared.ActiveSet.Count + " blocks");
            while (iter < activeSetIterations & !converged)
            {

                //iterations
                for (int i = 0; i < deconvolvers.Count; i++)
                {
                    tasks[i] = new Task(deconvolvers[i].Run);
                    tasks[i].Start();
                }

                for (int i = 0; i < deconvolvers.Count; i++)
                    tasks[i].Wait();
                
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
                    theta0 = shared.Tau / (float)blockCount;
                    for (int i = 0; i < deconvolvers.Count; i++)
                        deconvolvers[i].theta = theta0;
                    shared.testRestart = 0.0f;
                }

                if (deconvolvers.Sum(x => x.xDiffMax) < epsilon)
                {
                    converged = true;
                }

                for(int i = 0; i < deconvolvers.Count; i++)
                    deconvolvers[i].xDiffMax = 0.0f;

                Console.WriteLine("Done Active Set iteration " + iter);
                iter++;
            }

            return deconvolvers[0].theta;
        }

        private class SharedData
        {
            public float Lambda { get; set; }
            public float Alpha { get; set; }

            public int YBlockSize { get; set; }
            public int XBlockSize { get; set; }
            public int Tau { get; private set; }

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
                int tau,

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
                Tau = tau;

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

        private class ConcurrentDeconvolver
        {
            private readonly SharedData shared;
            private readonly float[,] blockUpdate;
            public float theta = 0.0f;
            public float xDiffMax = 0.0f;
            
            public ConcurrentDeconvolver(SharedData shared, float theta0)
            {
                this.shared = shared;
                theta = theta0;
                blockUpdate = new float[shared.YBlockSize, shared.XBlockSize];
            }

            public void Run()
            {
                var blockCount = shared.ActiveSet.Count;
                var theta0 = shared.Tau / (float)blockCount;
                float eta = 1.0f / blockCount;

                var beta = CalcESO(shared.Tau, shared.DegreeOfSeperability, blockCount);
                var lipschitz = shared.maxLipschitz * shared.YBlockSize * shared.XBlockSize;
                lipschitz *= (float)beta;
                
                for (int inner = 0; inner < shared.MaxConcurrentIterations; inner++)
                {
                    var stepSize = lipschitz * theta / theta0;
                    var theta2 = theta * theta;
                    var correctionFactor = -(1.0f - theta / theta0) / theta2;

                    var blockIdx = GetRandomBlockIdx(shared.Random, shared.BlockLock);
                    var blockSample = shared.ActiveSet[blockIdx];
                    var yOffset = blockSample.Item1 * shared.YBlockSize;
                    var xOffset = blockSample.Item2 * shared.XBlockSize;

                    var updateSum = 0.0f;
                    var updateAbsSum = 0.0f;
                    for (int y = yOffset; y < (yOffset + shared.YBlockSize); y++)
                        for (int x = xOffset; x < (xOffset + shared.XBlockSize); x++)
                        {
                            var update = theta2 * shared.GCorr[y, x] + shared.GExpl[y, x] + shared.XExpl[y, x] * stepSize;
                            update = ElasticNet.ProximalOperator(update, stepSize, shared.Lambda, shared.Alpha) - shared.XExpl[y, x];
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
                                var oldExplore = shared.XExpl[y, x];
                                var oldCorrection = shared.XCorr[y, x];

                                oldXExplore += shared.XExpl[y, x];
                                oldXCorr += shared.XCorr[y, x];
                                newXExplore += oldExplore + update;

                                shared.XExpl[y, x] += update;
                                shared.XCorr[y, x] += update * correctionFactor;
                            }

                        //not 100% sure this is the correct generalization from single pixel thread rule to block rule
                        var testRestartUpdate = (updateSum) * (newXExplore - (theta2 * oldXCorr + oldXExplore));
                        ConcurrentUpdateTestRestart(ref shared.testRestart, eta, testRestartUpdate);
                    }

                    //unlockBlock
                    shared.BlockLock[blockIdx] = 0;
                    
                    theta = (float)(Math.Sqrt((theta2 * theta2) + 4 * (theta2)) - theta2) / 2.0f;
                }  
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
                    var read = map[yIdx, xIdx];
                    var old = Interlocked.CompareExchange(ref map[yIdx, xIdx], read - value, read);
                    successfulWrite = old == read;
                } 
            }

            private static void ConcurrentUpdateTestRestart(ref float test, float eta, float subtract)
            {
                var successfulWrite = subtract == 0.0f;    //skip write if value to subtract is zero
                while (!successfulWrite)
                {
                    var read = test;
                    var update = (1.0f - eta) * read - subtract;
                    var old = Interlocked.CompareExchange(ref test, update, read);
                    successfulWrite = old == read;
                }
            }
        }
    }
}
