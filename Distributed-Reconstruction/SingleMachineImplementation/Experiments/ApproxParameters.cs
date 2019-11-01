using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Numerics;
using static System.Math;

using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns.Experiments
{
    class ApproxParameters
    {
        static float LAMBDA = 0.4f;
        static float ALPHA = 0.1f;

        private static void Reconstruct(Data input, int cutFactor, float[,] fullPsf, string folder, int threads, int blockSize, bool accelerated)
        {
            var approx = new ApproxFast(threads, blockSize, true, accelerated);
            var psfCut = PSF.Cut(fullPsf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(fullPsf, cutFactor);
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var random = new Random(123);

            var maxLipschitzCut = PSF.CalcMaxLipschitz(psfCut);
            var lambda = (float)(LAMBDA * PSF.CalcMaxLipschitz(psfCut));
            var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(fullPsf));
            var alpha = ALPHA;

            var data = new ApproxFast.TestingData(new StreamWriter(folder+ "/" + folder + ".txt"));
            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            for (int cycle = 0; cycle < 3; cycle++)
            {
                Console.WriteLine("cycle " + cycle);
                var dirtyGrid = IDG.GridW(input.c, input.metadata, residualVis, input.uvw, input.frequencies);
                var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, input.c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, folder + "dirty" + cycle + ".fits");

                var maxDirty = Residuals.GetMax(dirtyImage);
                var bMap = bMapCalculator.Convolve(dirtyImage);
                var maxB = Residuals.GetMax(bMap);
                var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitzCut), 1.0f);
                var currentSideLobe = maxB * maxSidelobe * correctionFactor;
                var currentLambda = (float)Math.Max(currentSideLobe / alpha, lambda);

                approx.DeconvolveTest(data, cycle, xImage, dirtyImage, psfCut, fullPsf, currentLambda, alpha, random, 100);
                FitsIO.Write(xImage, folder + "xImage_" + cycle + ".fits");

            }
        }


        public static void Run()
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
            FFT.Shift(psf);
            

            Directory.CreateDirectory("ApproxParams");
            FitsIO.Write(psf, "psfFull.fits");


            //tryout with simply cutting the PSF
            var outFolder = "ApproxTest";
            Directory.CreateDirectory(outFolder);
            Reconstruct(data, 4, psf, outFolder, 4, 1, true);
            
        }
    }
}
