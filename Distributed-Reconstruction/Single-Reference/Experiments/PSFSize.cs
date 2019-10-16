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
using static Single_Reference.Experiments.DataLoading;


namespace Single_Reference.Experiments
{
    static class PSFSize
    {
        public const double REFERENCE_L2_PENALTY = 7.21195378695326;
        public const double REFERENCE_ELASTIC_PENALTY = 44.1032910858257;
        //51,31524487277896

        public class InputData
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
        public class ReconstructionInfo
        {
            public Stopwatch totalDeconv;
            public double lastDataPenalty;
            public double lastRegPenalty;

            public ReconstructionInfo()
            {
                totalDeconv = new Stopwatch();
            }
        }

        private static ReconstructionInfo ReconstructGradientApprox(Data input, float[,] fullPsf, string folder, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var bMapCalculator2 = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(fullPsf, totalSize), new Rectangle(0, 0, fullPsf.GetLength(0), fullPsf.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            var fastCD2 = new FastGreedyCD(totalSize, psfCut);
            fastCD2.ResetAMap(fullPsf, cutFactor);
            FitsIO.Write(psfCut, folder + cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var lambdaTrue = (float)(0.4f * PSF.CalcMaxLipschitz(fullPsf));
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
            var minimumReached = false;
            var firstTimeConverged = false;
            var lastLambda = 0.0f;
            for (int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
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
                minimumReached = (lastResult != null && lastResult.Converged && lastResult.IterationCount < 20 && currentLambda == lambda);
                if (lambda == lastLambda)
                    firstTimeConverged = true;

                if (!objectiveReached & !minimumReached)
                {
                    writer.Write(firstTimeConverged + ";");
                    writer.Flush();
                    info.totalDeconv.Start();
                    if (!firstTimeConverged)
                    {
                        lastResult = fastCD.Deconvolve(xImage, bMap, currentLambda, alpha, 50000, epsilon);
                    } else
                    {
                        bMap = bMapCalculator2.Convolve(dirtyImage);
                        //FitsIO.Write(bMap, folder + dirtyPrefix + "bmap_" + cycle + "_full.fits");
                        currentSideLobe = Residuals.GetMax(bMap) * maxSidelobe;
                        currentLambda = Math.Max(currentSideLobe / alpha, lambdaTrue);
                        info.totalDeconv.Start();
                        lastResult = fastCD.Deconvolve(xImage, bMap, currentLambda, alpha, 50000, epsilon);
                        info.totalDeconv.Stop();
                    }
                   
                    info.totalDeconv.Stop();

                    FitsIO.Write(xImage, folder + xImagePrefix + cycle + ".fits");
                    writer.Write(lastResult.Converged + ";" + lastResult.IterationCount + ";" + lastResult.ElapsedTime.TotalSeconds + "\n");
                    writer.Flush();

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
                    residualVis = IDG.Substract(input.visibilities, modelVis, input.flags);
                }
                else
                {
                    writer.Write(false + ";0;0\n");
                    writer.Flush();

                    break;
                }

                lastLambda = currentLambda;

            }

            return info;
        }

        private static ReconstructionInfo ReconstructSimple(Data input, float[,] fullPsf, string folder, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon, bool startWithFullPSF)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = startWithFullPSF ? fullPsf : psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            if(startWithFullPSF)
                fastCD.ResetAMap(fullPsf, cutFactor);
            FitsIO.Write(psfCut, folder + cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var lambdaTrue =(float)( 0.4f * PSF.CalcMaxLipschitz(fullPsf));
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
            for(int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, folder + dirtyPrefix + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = Residuals.CalcPenalty(dirtyImage);
                var regPenalty = ElasticNet.CalcPenalty(xImage, lambdaTrue, alpha);
                var regPenaltyCurrent = ElasticNet.CalcPenalty(xImage, lambda, alpha);
                info.lastDataPenalty = dataPenalty;
                info.lastRegPenalty = regPenalty;

                bMapCalculator.ConvolveInPlace(dirtyImage);
                //FitsIO.Write(dirtyImage, folder + dirtyPrefix + "bmap_" + cycle + ".fits");
                var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                writer.Write(cycle + ";" + currentLambda + ";" + currentSideLobe + ";" + dataPenalty + ";" + regPenalty + ";" + regPenaltyCurrent + ";");
                writer.Flush();

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < objectiveCutoff;
                var minimumReached = (lastResult != null && lastResult.IterationCount < 100 && lastResult.Converged);
                if (!objectiveReached & !minimumReached)
                {
                    info.totalDeconv.Start();
                    lastResult = fastCD.Deconvolve(xImage, dirtyImage, currentLambda, alpha, 50000, epsilon);
                    info.totalDeconv.Stop();

                    FitsIO.Write(xImage, folder+xImagePrefix + cycle + ".fits");
                    writer.Write(lastResult.Converged + ";" + lastResult.IterationCount + ";" + lastResult.ElapsedTime.TotalSeconds + "\n");
                    writer.Flush();

                    FFT.Shift(xImage);
                    var xGrid = FFT.Forward(xImage);
                    FFT.Shift(xImage);
                    var modelVis = IDG.DeGridW(input.c, input.metadata, xGrid, input.uvw, input.frequencies);
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
            var data = LMC.Load(folder);

            var maxW = 0.0;
            for (int i = 0; i < data.uvw.GetLength(0); i++)
                for (int j = 0; j < data.uvw.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.uvw[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.frequencies[data.frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < data.flags.GetLength(0); i++)
                for (int j = 0; j < data.flags.GetLength(1); j++)
                    for (int k = 0; k < data.flags.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            visCount2++;
            var visibilitiesCount = visCount2;
            int gridSize = 2048;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 2.0 / 3600.0 * PI / 180.0;
            int wLayerCount = 32;
            double wStep = maxW / (wLayerCount);
            data.c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
            data.metadata = Partitioner.CreatePartition(data.c, data.uvw, data.frequencies);
            data.visibilitiesCount = visibilitiesCount;

            var psfVis = new Complex[data.uvw.GetLength(0), data.uvw.GetLength(1), data.frequencies.Length];
            for (int i = 0; i < data.visibilities.GetLength(0); i++)
                for (int j = 0; j < data.visibilities.GetLength(1); j++)
                    for (int k = 0; k < data.visibilities.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            Console.WriteLine("gridding psf");
            var psfGrid = IDG.GridW(data.c, data.metadata, psfVis, data.uvw, data.frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, data.c.VisibilitiesCount);
            FFT.Shift(psf);

            Directory.CreateDirectory("PSFSizeExperiment");
            Directory.SetCurrentDirectory("PSFSizeExperiment");
            FitsIO.Write(psf, "psfFull.fits");

            //reconstruct with full psf and find reference objective value
            var fileHeader = "cycle;lambda;sidelobe;dataPenalty;regPenalty;currentRegPenalty;converged;iterCount;ElapsedTime";
            var objectiveCutoff = REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY;
            var recalculateFullPSF = true;
            if (recalculateFullPSF)
            {
                ReconstructionInfo referenceInfo = null;
                using (var writer = new StreamWriter("1Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    referenceInfo = ReconstructSimple(data, psf, "", 1, 10, "dirtyReference", "xReference", writer, 0.0, 1e-5f, false);
                    File.WriteAllText("1PsfTotal.txt", referenceInfo.totalDeconv.Elapsed.ToString());
                }
                objectiveCutoff = referenceInfo.lastDataPenalty + referenceInfo.lastRegPenalty;
            }
            
            //tryout with simply cutting the PSF
            ReconstructionInfo experimentInfo = null;
            var psfCuts = new int[] { 32};
            var outFolder = "cutPsf";
            Directory.CreateDirectory(outFolder);
            outFolder += @"\";
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(outFolder + cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructSimple(data, psf, outFolder, cut, 10, cut+"dirty", cut+"x", writer, objectiveCutoff, 1e-5f, false);
                    File.WriteAllText(outFolder + cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }

            //Tryout with cutting the PSF, but starting from the true bMap
            outFolder = "cutPsf2";
            Directory.CreateDirectory(outFolder);
            outFolder += @"\";
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(outFolder + cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructSimple(data, psf, outFolder, cut, 10, cut + "dirty", cut + "x", writer, objectiveCutoff, 1e-5f, true);
                    File.WriteAllText(outFolder + cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }

            //combined, final solution. Cut the psf in half, optimize until convergence, and then do one more major cycle with the second method
            outFolder = "properSolution";
            Directory.CreateDirectory(outFolder);
            outFolder += @"\";
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(outFolder + cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructGradientApprox(data, psf, outFolder, cut, 12, cut + "dirty", cut + "x", writer, objectiveCutoff, 1e-5f);
                    File.WriteAllText(outFolder + cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }
        }

        public static void RunSpeed()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var data = LMC.Load(folder);
            int gridSize = 2048;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 2.0 / 3600.0 * PI / 180.0;
            int wLayerCount = 32;

            var maxW = 0.0;
            for (int i = 0; i < data.uvw.GetLength(0); i++)
                for (int j = 0; j < data.uvw.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.uvw[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.frequencies[data.frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < data.flags.GetLength(0); i++)
                for (int j = 0; j < data.flags.GetLength(1); j++)
                    for (int k = 0; k < data.flags.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            visCount2++;

            data.c = new GriddingConstants(data.visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, maxW);
            data.metadata = Partitioner.CreatePartition(data.c, data.uvw, data.frequencies);

            var psfVis = new Complex[data.uvw.GetLength(0), data.uvw.GetLength(1), data.frequencies.Length];
            for (int i = 0; i < data.visibilities.GetLength(0); i++)
                for (int j = 0; j < data.visibilities.GetLength(1); j++)
                    for (int k = 0; k < data.visibilities.GetLength(2); k++)
                        if (!data.flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            Console.WriteLine("gridding psf");
            var psfGrid = IDG.GridW(data.c, data.metadata, psfVis, data.uvw, data.frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, data.c.VisibilitiesCount);
            FFT.Shift(psf);
            var objectiveCutoff = REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY;

            Directory.CreateDirectory("PSFSpeedExperiment");
            FitsIO.Write(psf, "psfFull.fits");


            //tryout with simply cutting the PSF
            ReconstructionInfo experimentInfo = null;
            var psfCuts = new int[] { 8, 16, 32, 64, 128 };
            var outFolder = "PSFSpeedExperiment";
            outFolder += @"\";
            var fileHeader = "cycle;lambda;sidelobe;dataPenalty;regPenalty;currentRegPenalty;converged;iterCount;ElapsedTime";
            foreach (var cut in psfCuts)
            {
                using (var writer = new StreamWriter(outFolder + cut + "Psf.txt", false))
                {
                    writer.WriteLine(fileHeader);
                    experimentInfo = ReconstructSimple(data, psf, outFolder, cut, 10, cut+"dirty", cut+"x", writer, objectiveCutoff, 1e-5f, false);
                    File.WriteAllText(outFolder + cut + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }



        }


    }
}
