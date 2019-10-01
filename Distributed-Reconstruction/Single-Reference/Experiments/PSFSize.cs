﻿using System;
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
        public const double REFERENCE_L2_PENALTY = 5.59695057707117;
        public const double REFERENCE_ELASTIC_PENALTY = 17.7968081988002;
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

        private static ReconstructionInfo Reconstruct(InputData input, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(input.fullPsf, totalSize), new Rectangle(0, 0, input.fullPsf.GetLength(0), input.fullPsf.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            fastCD.ResetAMap(input.fullPsf);
            FitsIO.Write(psfCut, cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
            for(int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, dirtyPrefix + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = Residuals.CalculatePenalty(dirtyImage);
                var regPenalty = ElasticNet.CalculatePenalty(xImage, lambda, alpha);
                info.lastDataPenalty = dataPenalty;
                info.lastRegPenalty = regPenalty;

                bMapCalculator.ConvolveInPlace(dirtyImage);
                FitsIO.Write(dirtyImage, dirtyPrefix + "bmap_" + cycle + ".fits");
                var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                writer.Write(cycle + ";" + currentLambda + ";" + currentSideLobe + ";" + dataPenalty + ";" + regPenalty + ";" + ";");
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

            //reconstruct with full psf
            var objectiveCutoff = REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY;
            
            ReconstructionInfo referenceInfo = null;
            using (var writer = new StreamWriter("1Psf.txt", false))
            {
                writer.WriteLine("cycle;dataPenalty;regPenalty;converged;iterCount;ElapsedTime");
                referenceInfo = Reconstruct(input, 1, 10, "dirtyReference", "xReference", writer, 0.0, 1e-5f);
                File.WriteAllText("1PsfTotal.txt", referenceInfo.totalDeconv.Elapsed.ToString());
            }
            objectiveCutoff = referenceInfo.lastDataPenalty + referenceInfo.lastRegPenalty;
            
            ReconstructionInfo experimentInfo = null;
            var psfCuts = new int[] { 2, 4, 8, 16, 32, 64};
            foreach(var cut in psfCuts)
            {
                using (var writer = new StreamWriter(cut + "Psf.txt", false))
                {
                    writer.WriteLine("cycle;currentLambda;dataPenalty;regPenalty;converged;iterCount;ElapsedTime");
                    experimentInfo = Reconstruct(input, cut, 16, cut+"dirty", cut+"x", writer, objectiveCutoff, 1e-5f);
                    File.WriteAllText(cut+"PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
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

            var xImage = ToFloatImage(FitsIO.ReadImageDouble("32x7.fits"));
            var psf = ToFloatImage(FitsIO.ReadImageDouble("psfFull.fits"));


            var input = new InputData(c, metadata, frequencies, visibilities, uvw, flags, psf);
            var lambda = 0.5f;
            var alpha = 0.02f;
            var cutFactor = 32;
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            fastCD.ResetAMap(psf);


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
                var dataPenalty = FastGreedyCD.CalcDataPenalty(dirtyImage);
                var regPenalty = fastCD.CalcRegularizationPenalty(xImage, lambda, alpha);
                var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                var currentLambda = Math.Max(currentSideLobe / alpha, lambda);

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < (REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY);
                var minimumReached = (lastResult != null && lastResult.IterationCount < 20 && lastResult.Converged);
                if (!objectiveReached)
                {
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
    }
}