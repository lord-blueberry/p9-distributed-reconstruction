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
        private class InputData
        {
            public GriddingConstants c;
            public List<List<SubgridHack>> metadata;
            public double[] frequencies;
            public Complex[,,] visibilities;
            public double[,,] uvw;
            public bool[,,] flags;
            public float[,] fullPsf;
            public FastGreedyCD fullCD;

            public InputData(GriddingConstants c, List<List<SubgridHack>> metadata, double[] frequencies, Complex[,,] vis, double[,,] uvw, bool[,,] flags, float[,] fullPsf)
            {
                this.c = c;
                this.metadata = metadata;
                this.frequencies = frequencies;
                this.visibilities = vis;
                this.uvw = uvw;
                this.flags = flags;
                this.fullPsf = fullPsf;
                this.fullCD = new FastGreedyCD(new Rectangle(0, 0, c.GridSize, c.GridSize), fullPsf);
            }

        }
        private class ReconstructionInfo
        {
            public Stopwatch totalDeconv;
            public double lastDataPenalty;
            public double lastRegPenalty;
            public double lastRegPenaltyFull;

            public ReconstructionInfo()
            {
                totalDeconv = new Stopwatch();
            }
        }

        private static ReconstructionInfo Reconstruct(InputData input, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveVal)
        {
            var info = new ReconstructionInfo();
            var lambda = 0.4f;
            var alpha = 0.1f;

            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            for(int cycle = 0; cycle < maxMajor; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.Grid(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, dirtyPrefix + cycle + ".fits");

                //calc data and reg penalty
                var dataPenalty = FastGreedyCD.CalcDataPenalty(dirtyImage);
                var regPenalty = fastCD.CalcRegularizationPenalty(xImage, lambda, alpha);
                var regPenaltyFull = input.fullCD.CalcRegularizationPenalty(xImage, lambda, alpha);
                info.lastDataPenalty = dataPenalty;
                info.lastRegPenalty = regPenalty;
                info.lastRegPenaltyFull = regPenaltyFull;
                var currentSideLobe = Residuals.GetMax(dirtyImage) * maxSidelobe;
                var currentLambda = Math.Max(1.0f / alpha * currentSideLobe, lambda);

                writer.Write(cycle + ";" + currentLambda + ";" + dataPenalty + ";" + regPenalty + ";" + regPenaltyFull + ";");
                writer.Flush();

                //not below objective, go further
                if (objectiveVal < dataPenalty + regPenaltyFull)
                {
                    bMapCalculator.ConvolveInPlace(dirtyImage);

                    info.totalDeconv.Start();
                    var result = fastCD.Deconvolve(xImage, dirtyImage, currentLambda, alpha, 10000, 1e-5f);
                    info.totalDeconv.Stop();

                    FitsIO.Write(xImage, xImagePrefix + cycle + ".fits");

                    writer.Write(result.Converged + ";" + result.IterationCount + ";" + result.ElapsedTime.TotalSeconds + "\n");
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

            int gridSize = 1024;
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

            //reconstruct with half the psf
            ReconstructionInfo referenceInfo = null;
            using (var writer = new StreamWriter("1Psf.txt", false))
            {
                writer.WriteLine("cycle;dataPenalty;regPenalty;regPenaltyFull;converged;iterCount;ElapsedTime");
                referenceInfo = Reconstruct(input, 1, 5, "dirtyReference", "xReference", writer, 0.0);
                File.WriteAllText("1PsfTotal.txt", referenceInfo.totalDeconv.Elapsed.ToString());
            }

            var objectiveCutoff = referenceInfo.lastDataPenalty + referenceInfo.lastRegPenaltyFull;
            var psfCuts = new int[] { 2, 4, 8};
            foreach(var cut in psfCuts)
            {
                using (var writer = new StreamWriter(cut + "Psf.txt", false))
                {
                    writer.WriteLine("cycle;dataPenalty;regPenalty;regPenaltyFull;converged;iterCount;ElapsedTime");
                    referenceInfo = Reconstruct(input, cut, 15, cut+"dirtyReference", cut+"xReference", writer, objectiveCutoff);
                    File.WriteAllText(cut+"PsfTotal.txt", referenceInfo.totalDeconv.Elapsed.ToString());
                }
            }
            


        }
    }
}
