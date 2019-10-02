using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Numerics;
using static System.Math;

using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;



namespace Single_Reference.Experiments
{
    class PSFSize
    {
        public const double REFERENCE_L2_PENALTY = 5.64324894802577;
        public const double REFERENCE_ELASTIC_PENALTY = 29.3063615047876;
        private class InputData
        {
            public GriddingConstants c;
            public List<List<SubgridHack>> metadata;
            public double[] frequencies;
            public Complex[,,] visibilities;
            public double[,,] uvw;
            public bool[,,] flags;
            public float[,] fullPsf;

            public InputData(GriddingConstants c, List<List<SubgridHack>> metadata, double[] frequencies, Complex[,,] vis, double[,,] uvw, bool[,,] flags, float[,] fullPsf)
            {
                this.c = c;
                this.metadata = metadata;
                this.frequencies = frequencies;
                this.visibilities = vis;
                this.uvw = uvw;
                this.flags = flags;
                this.fullPsf = fullPsf;
            }

        }
        private class ReconstructionInfo
        {
            public Stopwatch totalDeconv;
            public double lastDataPenalty;
            public double lastRegPenalty;

            public ReconstructionInfo()
            {
                totalDeconv = new Stopwatch();
            }
        }

        private static ReconstructionInfo ReconstructGradientApprox(InputData input, string folder, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            FitsIO.Write(psfCut, folder + cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var lambdaTrue = 0.4f * Common.PSF.CalcMaxLipschitz(input.fullPsf);
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
            var minimumReached = false;
            for (int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, folder + dirtyPrefix + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = Residuals.CalcPenalty(dirtyImage);
                var regPenalty = ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                var regPenaltyCurrent = ElasticNet.CalcPenalty(xImage, lambda, alpha);
                info.lastDataPenalty = dataPenalty;
                info.lastRegPenalty = regPenalty;

                var bMap = bMapCalculator.Convolve(dirtyImage);
                FitsIO.Write(bMap, folder + dirtyPrefix + "bmap_" + cycle + ".fits");
                var currentSideLobe = Residuals.GetMax(bMap) * maxSidelobe;
                var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                writer.Write(cycle + ";" + currentLambda + ";" + currentSideLobe + ";" + dataPenalty + ";" + regPenalty + ";" + regPenaltyCurrent + ";");
                writer.Flush();

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < objectiveCutoff;
                minimumReached = (lastResult != null && lastResult.IterationCount < 1000 && lastResult.Converged && currentLambda == lambda);
                if (!objectiveReached & !minimumReached)
                {
                    info.totalDeconv.Start();
                    lastResult = fastCD.Deconvolve(xImage, bMap, currentLambda, alpha, 10000, epsilon);
                    info.totalDeconv.Stop();

                    FitsIO.Write(xImage, xImagePrefix + cycle + ".fits");
                    writer.Write(lastResult.Converged + ";" + lastResult.IterationCount + ";" + lastResult.ElapsedTime.TotalSeconds + "\n");
                    writer.Flush();

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGrid(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                }
                else
                {
                    writer.Write(false + ";0;0\n");
                    writer.Flush();
                    if(!objectiveReached & minimumReached)
                    {
                        //do one last run, starting with the bMap of Full gradients
                        fastCD.ResetAMap(input.fullPsf);
                        bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(input.fullPsf, totalSize), new Rectangle(0, 0, input.fullPsf.GetLength(0), input.fullPsf.GetLength(1)));
                        bMap = bMapCalculator.Convolve(dirtyImage);
                        FitsIO.Write(bMap, folder + dirtyPrefix + "bmap_" + cycle + "_full.fits");
                        currentSideLobe = Residuals.GetMax(bMap) * maxSidelobe;
                        currentLambda = Math.Max(currentSideLobe / alpha, lambdaTrue);

                        info.totalDeconv.Start();
                        lastResult = fastCD.Deconvolve(xImage, bMap, currentLambda, alpha, 10000, epsilon);
                        info.totalDeconv.Stop();

                        FFT.Shift(xImage);
                        var xGrid = FFT.Forward(xImage);
                        FFT.Shift(xImage);
                        var modelVis = IDG.DeGrid(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                        residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                        dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                        dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);

                        dataPenalty = Residuals.CalcPenalty(dirtyImage);
                        regPenalty = ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                        regPenaltyCurrent = ElasticNet.CalcPenalty(xImage, lambda, alpha);
                        info.lastDataPenalty = dataPenalty;
                        info.lastRegPenalty = regPenalty;
                    }

                    break;
                }

            }

            return info;
        }

        private static ReconstructionInfo ReconstructSimple(InputData input, string folder, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon, bool startWithFullPSF)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = startWithFullPSF ? input.fullPsf : psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            if(!startWithFullPSF)
                fastCD.ResetAMap(input.fullPsf);
            FitsIO.Write(psfCut, folder + cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var alpha = 0.1f;
            var lambdaTrue = 55.20486f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
            for(int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, folder + dirtyPrefix + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = Residuals.CalcPenalty(dirtyImage);
                var regPenalty = ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                var regPenaltyCurrent = ElasticNet.CalcPenalty(xImage, lambda, alpha);
                info.lastDataPenalty = dataPenalty;
                info.lastRegPenalty = regPenalty;

                bMapCalculator.ConvolveInPlace(dirtyImage);
                FitsIO.Write(dirtyImage, folder + dirtyPrefix + "bmap_" + cycle + ".fits");
                var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                writer.Write(cycle + ";" + currentLambda + ";" + currentSideLobe + ";" + dataPenalty + ";" + regPenalty + ";" + regPenaltyCurrent + ";");
                writer.Flush();

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < objectiveCutoff;
                var minimumReached = (lastResult != null && lastResult.IterationCount < 20 && lastResult.Converged);
                if (!objectiveReached & !minimumReached)
                {
                    info.totalDeconv.Start();
                    lastResult = fastCD.Deconvolve(xImage, dirtyImage, currentLambda, alpha, 10000, epsilon);
                    info.totalDeconv.Stop();

                    FitsIO.Write(xImage, xImagePrefix + cycle + ".fits");
                    writer.Write(lastResult.Converged + ";" + lastResult.IterationCount + ";" + lastResult.ElapsedTime.TotalSeconds + "\n");
                    writer.Flush();

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGrid(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                }
                else
                {
                    writer.Write(false + ";0;0");
                    writer.Flush();
                    break;
                }

            }

            return info;
        }

        public static void Run()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw0.fits"));
            var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            for (int i = 1; i < 8; i++)
            {
                var uvw0 = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + i + ".fits"));
                var flags0 = FitsIO.ReadFlags(Path.Combine(folder, "flags" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, norm);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            var visCount2 = 0;
            for (int i = 0; i < flags.GetLength(0); i++)
                for (int j = 0; j < flags.GetLength(1); j++)
                    for (int k = 0; k < flags.GetLength(2); k++)
                        if (!flags[i, j, k])
                            visCount2++;
            var visibilitiesCount = visCount2;

            int gridSize = 2048;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 512;
            double scaleArcSec = 2.5 / 3600.0 * PI / 180.0;

            Console.WriteLine("Gridding PSF");

            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
            var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
            
            FFT.Shift(psf);
            FitsIO.Write(psf, "psfFull.fits");

            var input = new InputData(c, metadata, frequencies, visibilities, uvw, flags, psf);
            //---------------End file reading----------------------------------------------------------

            //reconstruct with full psf and find reference objective value
            var fileHeader = "cycle;dataPenalty;regPenalty;currentRegPenalty;converged;iterCount;ElapsedTime";
            var objectiveCutoff = REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY;
            var recalculateFullPSF = false;
            if (recalculateFullPSF)
            {
                ReconstructionInfo referenceInfo = null;
                using (var writer = new StreamWriter("1Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    referenceInfo = ReconstructSimple(input, "", 1, 10, "dirtyReference", "xReference", writer, 0.0, 1e-5f, false);
                    File.WriteAllText("1PsfTotal.txt", referenceInfo.totalDeconv.Elapsed.ToString());
                }
                objectiveCutoff = referenceInfo.lastDataPenalty + referenceInfo.lastRegPenalty;
            }

            //tryout with simply cutting the PSF
            ReconstructionInfo experimentInfo = null;
            var psfCuts = new int[] {/*2, 4, 8, 16,*/ 32, /*64*/};
            var outFolder = "cutPsf";
            Directory.CreateDirectory(folder);
            foreach(var cut in psfCuts)
            {
                using (var writer = new StreamWriter(cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructSimple(input, outFolder + @"\", cut, 16, cut+"dirty", cut+"x", writer, objectiveCutoff, 1e-5f, false);
                    File.WriteAllText(cut+"PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }
            
            //Tryout with cutting the PSF, but starting from the true bMap
            outFolder = "changeLipschitz";
            Directory.CreateDirectory(folder);
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructSimple(input, outFolder + @"\", cut, 16, cut + "dirty", cut + "x", writer, objectiveCutoff, 1e-5f, false);
                    File.WriteAllText(cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }

            //combined, final solution. Cut the psf in half, optimize until convergence, and then do one more major cycle with the second method
            outFolder = "properSolution";
            Directory.CreateDirectory(folder);
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructGradientApprox(input, outFolder + @"\", cut, 16, cut + "dirty", cut + "x", writer, objectiveCutoff, 1e-5f);
                    File.WriteAllText(cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }
        }

        public static void DebugConvergence()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw0.fits"));
            var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            for (int i = 1; i < 8; i++)
            {
                var uvw0 = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + i + ".fits"));
                var flags0 = FitsIO.ReadFlags(Path.Combine(folder, "flags" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, norm);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            var visCount2 = 0;
            for (int i = 0; i < flags.GetLength(0); i++)
                for (int j = 0; j < flags.GetLength(1); j++)
                    for (int k = 0; k < flags.GetLength(2); k++)
                        if (!flags[i, j, k])
                            visCount2++;
            var visibilitiesCount = visCount2;

            int gridSize = 2048;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 512;
            double scaleArcSec = 2.5 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var xImage = FitsIO.ReadImage("32x5-noAcorr.fits");
            var psf = FitsIO.ReadImage("psfFull.fits");


            var input = new InputData(c, metadata, frequencies, visibilities, uvw, flags, psf);

            var cutFactor = 32;
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = psf;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            fastCD.ResetAMap(psf);
            var lambda = (0.4f) * fastCD.MaxLipschitz;
            var lambda2 = 55.20486f;
            var alpha = 0.1f;


            FFT.Shift(xImage);
            var xGrid2 = FFT.Forward(xImage);
            FFT.Shift(xImage);
            var modelVis2 = IDG.DeGrid(input.c, input.metadata, xGrid2, input.uvw, input.frequencies);
            var residualVis = IDG.Substract(input.visibilities, modelVis2, input.flags);
            DeconvolutionResult lastResult = null;
            for (int cycle = 0; cycle < 10; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "debugDirty" + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = Residuals.CalcPenalty(dirtyImage);
                var regPenalty = ElasticNet.CalcPenalty(xImage, lambda2, alpha);
                var regPenalty_corr = ElasticNet.CalcPenalty(xImage, lambda, alpha);

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < (REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY);
                var minimumReached = (lastResult != null && lastResult.IterationCount < 20 && lastResult.Converged);
                if (!objectiveReached)
                {
                    var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                    var currentLambda = Math.Max(currentSideLobe / alpha, lambda);
                    bMapCalculator.ConvolveInPlace(dirtyImage);
                    FitsIO.Write(dirtyImage, "bMapDebug" + cycle + ".fits");
                    lastResult = fastCD.Deconvolve(xImage, dirtyImage, currentLambda, alpha, 10000, 1e-5f);

                    FitsIO.Write(xImage, "xDebug" + cycle + ".fits");

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGrid(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                }

                if(objectiveReached)
                {
                    Console.WriteLine("objective");
                }

            }
            FitsIO.Write(xImage, "xImageDebug.fits");
        }

        public static void DebugConvergence2()
        {
            var xImage = FitsIO.ReadImage("32x5-noAcorr.fits");
            var psf = FitsIO.ReadImage("psfFull.fits");
            var dirty = FitsIO.ReadImage("debugDirty0-noAcorr.fits");

            int gridSize = 2048;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 512;
            double scaleArcSec = 2.5 / 3600.0 * PI / 180.0;
            var visCount = 13922040;
            var c = new GriddingConstants(visCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);

            var cutFactor = 32;
            var psfCut = PSF.Cut(psf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, cutFactor);
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfBMap = psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            //fastCD.ResetAMap(psf);
            var lambda = (0.4f + maxSidelobe) * fastCD.MaxLipschitz;
            var lambda2 = 55.20486f;
            var alpha = 0.1f;

            var dataPenalty = Residuals.CalcPenalty(dirty);
            var regPenalty = ElasticNet.CalcPenalty(xImage, lambda2, alpha);
            var regPenalty_corr = ElasticNet.CalcPenalty(xImage, lambda, alpha);

            bMapCalculator.ConvolveInPlace(dirty);
            FitsIO.Write(dirty, "bMapDebug" + 0 + ".fits");
            var currentSideLobe = Residuals.GetMax(dirty) * maxSidelobe;
            var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

            var lastResult = fastCD.Deconvolve(xImage, dirty, currentLambda, alpha, 10000, 1e-5f);

            FitsIO.Write(xImage, "xDebug" + 0 + ".fits");
            var xImageLast = FitsIO.ReadImage("32x5-noAcorr.fits");
            var xDiff = new float[xImage.GetLength(0), xImage.GetLength(1)];
            for (int i = 0; i < xDiff.GetLength(0); i++)
                for (int j = 0; j < xDiff.GetLength(1); j++)
                    xDiff[i, j] = xImage[i, j] - xImageLast[i, j];
            FitsIO.Write(xDiff, "xDiffebug" + 0 + ".fits");
        }
    }
}
