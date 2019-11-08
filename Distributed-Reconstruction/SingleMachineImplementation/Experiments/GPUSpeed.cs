using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Single_Reference;
using Single_Reference.Deconvolution;
using Single_Reference.IDGSequential;
using static Single_Reference.Common;
using static SingleMachineRuns.Experiments.PSFSize;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns.Experiments
{
    class GPUSpeed
    {

        public static void Run()
        {
            var folder = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var data = DataLoading.SimulatedPoints.Load(folder);
            var gridSizes = new int[] { 256, 512, 1024, 2048, 4096};
            Directory.CreateDirectory("GPUSpeedup");
            var writer = new StreamWriter("GPUSpeedup/GPUSpeedup.txt", false);
            writer.WriteLine("imgSize;iterCPU;timeCPU;iterGPU;timeGPU");
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
