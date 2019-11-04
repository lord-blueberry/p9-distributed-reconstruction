using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;
using static System.Math;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns.ImageGeneration.ApproxImages
{
    class Generator
    {
        public static void LoadLMCData()
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";

            var data = LMC.Load(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * PI / 180.0;
            int wLayerCount = 24;

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
        }

        private static void GenerateApproxRandomProblem()
        {

        }


    }
}
