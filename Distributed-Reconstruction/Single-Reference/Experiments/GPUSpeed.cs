using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Single_Reference.Deconvolution;
using Single_Reference.IDGSequential;
using static Single_Reference.Common;
using static Single_Reference.Experiments.PSFSize;




namespace Single_Reference.Experiments
{
    class GPUSpeed
    {

        private static ReconstructionInfo ReconstructSimple(InputData input, string folder, int cutFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon, bool startWithFullPSF)
        {
            var info = new ReconstructionInfo();
            var psfCut = PSF.Cut(input.fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(input.fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var psfBMap = startWithFullPSF ? input.fullPsf : psfCut;
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfBMap, totalSize), new Rectangle(0, 0, psfBMap.GetLength(0), psfBMap.GetLength(1)));
            var fastCD = new FastGreedyCD(totalSize, psfCut);
            if (startWithFullPSF)
                fastCD.ResetAMap(input.fullPsf, cutFactor);
            FitsIO.Write(psfCut, folder + cutFactor + "psf.fits");

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var lambdaTrue = (float)(0.4f * Common.PSF.CalcMaxLipschitz(input.fullPsf));
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
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

                    FitsIO.Write(xImage, folder + xImagePrefix + cycle + ".fits");
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
            var folder = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var data = DataLoading.SimulatedPoints.Load(folder);
            var gridSizes = new int[] { 512, 1024, 2048, 4096};
            Directory.CreateDirectory("GPUSpeedup");
            var writer = new StreamWriter("GPUSpeedup/GPUSpeedup.txt", false);
            foreach (var gridSize in gridSizes)
            {
                var visibilitiesCount = data.visibilitiesCount;
                int subgridsize = 8;
                int kernelSize = 4;
                int max_nr_timesteps = 1024;
                double cellSize = (1.0 * 256/gridSize) / 3600.0 * Math.PI / 180.0;
                var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, data.uvw, data.frequencies);

                var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
                var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw.fits"));
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
                double norm = 2.0;
                var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

                var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
                var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
                FFT.Shift(psf);

                var residualVis = data.visibilities;
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, data.uvw, data.frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);

                var totalSize = new Rectangle(0, 0, gridSize, gridSize);
                var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
                var bMapCPU = bMapCalculator.Convolve(dirtyImage);
                var bMapGPU = bMapCalculator.Convolve(dirtyImage);
                var fastCD = new FastGreedyCD(totalSize, psf);
                var gpuCD = new GPUGreedyCD(totalSize, psf, 1000);
                var lambda = 0.5f * fastCD.MaxLipschitz;
                var alpha = 0.5f;

                var xCPU = new float[gridSize, gridSize];
                var cpuResult = fastCD.Deconvolve(xCPU, bMapCPU, lambda, alpha, 10000, 1e-8f);
                FitsIO.Write(xCPU, "GPUSpeedup/cpuResult" + gridSize + ".fits");

                var xGPU = new float[gridSize, gridSize];
                var gpuResult = gpuCD.Deconvolve(xGPU, bMapGPU, lambda, alpha, 10000, 1e-8f);
                FitsIO.Write(xCPU, "GPUSpeedup/gpuResult" + gridSize + ".fits");

                writer.WriteLine(gridSize + ";" + cpuResult.IterationCount + ";" + cpuResult.ElapsedTime.TotalSeconds + ";" + gpuResult.IterationCount + ";" + gpuResult.ElapsedTime.TotalSeconds);
                writer.Flush();
            }

            writer.Close();
            
        }
    }
}
