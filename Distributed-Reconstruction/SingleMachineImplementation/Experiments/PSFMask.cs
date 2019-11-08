using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.IO;
using static System.Math;

using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;
using static SingleMachineRuns.Experiments.PSFSize;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns.Experiments
{
    class PSFMask
    {
        private static ReconstructionInfo Reconstruct(InputData input, float fullLipschitz, float[,] maskedPsf, string folder, float maskFactor, int maxMajor, string dirtyPrefix, string xImagePrefix, StreamWriter writer, double objectiveCutoff, float epsilon, bool maskPsf2)
        {
            var info = new ReconstructionInfo();
            var totalSize = new Rectangle(0, 0, input.c.GridSize, input.c.GridSize);

            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(maskedPsf, totalSize), new Rectangle(0, 0, maskedPsf.GetLength(0), maskedPsf.GetLength(1)));
            var maskedPsf2 = PSF.CalcPSFSquared(maskedPsf);
            if (maskPsf2)
                Mask(maskedPsf2, 1e-5f);
            writer.WriteLine((CountNonZero(maskedPsf2) - maskedPsf2.Length)/ (double)maskedPsf2.Length);
            var fastCD = new FastGreedyCD(totalSize, totalSize, maskedPsf, maskedPsf2);
            FitsIO.Write(maskedPsf, folder + maskFactor + "psf.fits");

            

            var lambda = 0.4f * fastCD.MaxLipschitz;
            var lambdaTrue = 0.4f * fullLipschitz;
            var alpha = 0.1f;

            var xImage = new float[input.c.GridSize, input.c.GridSize];
            var residualVis = input.visibilities;
            DeconvolutionResult lastResult = null;
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

                bMapCalculator.ConvolveInPlace(dirtyImage);
                //FitsIO.Write(dirtyImage, folder + dirtyPrefix + "bmap_" + cycle + ".fits");
                var currentLambda = lambda;

                writer.Write(cycle + ";" + currentLambda + ";"  + Residuals.GetMax(dirtyImage) + ";" + dataPenalty + ";" + regPenalty + ";" + regPenaltyCurrent + ";");
                writer.Flush();

                //check wether we can minimize the objective further with the current psf
                var objectiveReached = (dataPenalty + regPenalty) < objectiveCutoff;
                var minimumReached = (lastResult != null && lastResult.IterationCount < 100 && lastResult.Converged);
                if (!objectiveReached & !minimumReached)
                {
                    info.totalDeconv.Start();
                    lastResult = fastCD.Deconvolve(xImage, dirtyImage, currentLambda, alpha, 50000, epsilon);
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
            var rootFolder = Directory.GetCurrentDirectory();

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
            var objectiveCutoff = REFERENCE_L2_PENALTY + REFERENCE_ELASTIC_PENALTY;
            var actualLipschitz = (float)PSF.CalcMaxLipschitz(psf);

            Console.WriteLine("Calc Histogram");
            var histPsf = GetHistogram(psf, 256, 0.05f);
            var experiments = new float[] { 0.5f, /*0.4f, 0.2f, 0.1f, 0.05f*/};
            Console.WriteLine("Done Histogram");

            Directory.CreateDirectory("PSFMask");
            Directory.SetCurrentDirectory("PSFMask");
            FitsIO.Write(psf, "psfFull.fits");

            //reconstruct with full psf and find reference objective value
            ReconstructionInfo experimentInfo = null;
            var outFolder = "";
            var fileHeader = "cycle;lambda;sidelobe;dataPenalty;regPenalty;currentRegPenalty;converged;iterCount;ElapsedTime";
            foreach (var maskPercent in experiments)
            {
                using (var writer = new StreamWriter(outFolder + maskPercent + "Psf.txt", false))
                {
                    var maskedPSF = Common.Copy(psf);
                    var maskedPixels = MaskPSF(maskedPSF, histPsf, maskPercent);
                    writer.WriteLine(maskedPixels + ";" + maskedPixels / (double)maskedPSF.Length);
                    FitsIO.Write(maskedPSF, outFolder + maskPercent + "Psf.fits");
                   
                    writer.WriteLine(fileHeader);
                    writer.Flush();
                    experimentInfo = Reconstruct(data, actualLipschitz, maskedPSF, outFolder, 1, 10, maskPercent + "dirty", maskPercent + "x", writer, objectiveCutoff, 1e-5f, false);
                    File.WriteAllText(outFolder + maskPercent + "PsfTotal.txt", experimentInfo.totalDeconv.Elapsed.ToString());
                }
            }

            Directory.SetCurrentDirectory(rootFolder);

        }

        private static Tuple<int[], float[]> GetHistogram(float[,] psf, int bins, float max)
        {
            var min = float.MaxValue;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    min = Math.Min(Math.Abs(psf[i, j]), min);
                }

            var histogram = new int[bins];
            var breaks = new float[bins];
            for (int i = 0; i < bins; i++)
                breaks[i] = i * (max - min) / bins + min;

            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var value = Math.Abs(psf[i, j]);

                    //binary search for bin
                    int start = 0;
                    int end = histogram.Length - 1;
                    while (start != end)
                    {
                        var idx = start + (end - start + 1) / 2;
                        if (value < breaks[idx])
                            end = idx - 1;
                        else
                            start = idx;

                    }
                    histogram[start]++;
                }

            return new Tuple<int[], float[]>(histogram, breaks);
        }

        private static int MaskPSF(float[,] psf, Tuple<int[], float[]> histPsf, float percent)
        {
            var sum = 0;
            var cutoff = 0.0f;
            var histogram = histPsf.Item1;
            for (int i = histogram.Length - 1; i >= 0; i--)
            {
                sum += histogram[i];
                var percentageOfValues = sum / (float)psf.Length;
                if (percentageOfValues > percent)
                {
                    cutoff = (1 + percentageOfValues - percent) * histPsf.Item2[i];
                    break;
                }
            }

            return Mask(psf, cutoff);
        }

        public static int Mask(float[,] psf, float cutOff)
        {
            var zeroCount = 0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    if (Math.Abs(psf[y, x]) < cutOff)
                    {
                        psf[y, x] = 0.0f;
                        zeroCount++;
                    }
            return zeroCount;
        }
    }
}
