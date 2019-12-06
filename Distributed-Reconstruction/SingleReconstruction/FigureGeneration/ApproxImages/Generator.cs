using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.IO;
using Core;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using static Core.Common;
using static System.Math;


namespace SingleReconstruction.FigureGeneration.ApproxImages
{
    class Generator
    {
        static string INPUT_DIR = "./FigureGeneration/ApproxImages";
        const float LAMBDA = 1.0f;
        const float ALPHA = 0.01f;

        public static void Generate(string outputFolder)
        {
            GenerateApproxRandomProblem(outputFolder);
            GeneratePCDMComparison(outputFolder);
        }

        #region Generate approx random problem
        private static void GenerateApproxRandomProblem(string outputFolder)
        {
            var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";
            Console.WriteLine("Generating approx random images");

            var data = MeasurementData.LoadLMC(folder);
            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.UVW.GetLength(0); i++)
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.UVW[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.Frequencies[data.Frequencies.Length - 1]);
            double wStep = maxW / (wLayerCount);

            var c = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
            var metadata = Partitioner.CreatePartition(c, data.UVW, data.Frequencies);

            var psfVis = new Complex[data.UVW.GetLength(0), data.UVW.GetLength(1), data.Frequencies.Length];
            for (int i = 0; i < data.Visibilities.GetLength(0); i++)
                for (int j = 0; j < data.Visibilities.GetLength(1); j++)
                    for (int k = 0; k < data.Visibilities.GetLength(2); k++)
                        if (!data.Flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            Console.WriteLine("gridding psf");
            var psfGrid = IDG.GridW(c, metadata, psfVis, data.UVW, data.Frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            Directory.CreateDirectory(outputFolder);
            ReconstructRandom(data, c, psf, 1, 200, outputFolder + "/random__block1");
            ReconstructRandom(data, c, psf, 8, 50, outputFolder + "/random_10k_block8");
        }

        private static void ReconstructRandom(MeasurementData input, GriddingConstants c, float[,] psf, int blockSize, int iterCount, string file)
        {
            var cutFactor = 8;
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var psfCut = PSF.Cut(psf, cutFactor);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, cutFactor);

            var maxLipschitzCut = PSF.CalcMaxLipschitz(psfCut);
            var lambda = (float)(LAMBDA * PSF.CalcMaxLipschitz(psfCut));
            var lambdaTrue = (float)(LAMBDA * PSF.CalcMaxLipschitz(psf));
            var alpha = ALPHA;
            ApproxFast.LAMBDA_TEST = lambdaTrue;
            ApproxFast.ALPHA_TEST = alpha;

            var metadata = Partitioner.CreatePartition(c, input.UVW, input.Frequencies);

            var random = new Random(123);
            var approx = new ApproxFast(totalSize, psfCut, 8, blockSize, 0.0f, 0.0f, false, true, false);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var data = new ApproxFast.TestingData(new StreamWriter(file + "_tmp.txt"));
            var xImage = new float[c.GridSize, c.GridSize];
            var xCorr = Copy(xImage);
            var residualVis = input.Visibilities;

            var dirtyGrid = IDG.GridW(c, metadata, residualVis, input.UVW, input.Frequencies);
            var dirtyImage = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
            FFT.Shift(dirtyImage);

            var maxDirty = Residuals.GetMax(dirtyImage);
            var bMap = bMapCalculator.Convolve(dirtyImage);
            var maxB = Residuals.GetMax(bMap);
            var correctionFactor = Math.Max(maxB / (maxDirty * maxLipschitzCut), 1.0f);
            var currentSideLobe = maxB * maxSidelobe * correctionFactor;
            var currentLambda = (float)Math.Max(currentSideLobe / alpha, lambda);

            var gCorr = new float[c.GridSize, c.GridSize];
            var shared = new ApproxFast.SharedData(currentLambda, alpha, 1, 1, 8, CountNonZero(psfCut), approx.psf2, approx.aMap, xImage, xCorr, bMap, gCorr, new Random());
            shared.ActiveSet = ApproxFast.GetActiveSet(xImage, bMap, shared.YBlockSize, shared.XBlockSize, lambda, alpha, shared.AMap);
            shared.BlockLock = new int[shared.ActiveSet.Count];
            shared.maxLipschitz = (float)PSF.CalcMaxLipschitz(psfCut);
            shared.MaxConcurrentIterations = 1000;
            approx.DeconvolveConcurrentTest(data, 0, 0, 0.0, shared, 1, 1e-5f, Copy(xImage), dirtyImage, psfCut, psf);
            var output = Tools.LMC.CutN132Remnant(xImage);
            Tools.WriteToMeltCSV(output.Item1, file + "_1k.csv", output.Item2, output.Item3);
            FitsIO.Write(output.Item1, file + "_1k.fits");
            FitsIO.Write(xImage, file + "_1k2.fits");

            approx.DeconvolveConcurrentTest(data, 0, 0, 0.0, shared, iterCount, 1e-5f, Copy(xImage), dirtyImage, psfCut, psf);
            output = Tools.LMC.CutN132Remnant(xImage);
            Tools.WriteToMeltCSV(output.Item1, file + "_10k.csv", output.Item2, output.Item3);
            FitsIO.Write(output.Item1, file + "_10k.fits");
            FitsIO.Write(xImage, file + "_10k2.fits");
        }
        #endregion

        private static void GeneratePCDMComparison(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var serial = FitsIO.ReadImage(Path.Combine(INPUT_DIR, "xImage_serial_4.fits"));
            var n132 = Tools.LMC.CutN132Remnant(serial);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "SerialCD-N132.csv"), n132.Item2, n132.Item3);
            var calibration = Tools.LMC.CutCalibration(serial);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "SerialCD-Calibration.csv"), calibration.Item2, calibration.Item3);


            var pcdm = FitsIO.ReadImage(Path.Combine(INPUT_DIR, "xImage_pcdm_5.fits"));
            var n132_2 = Tools.LMC.CutN132Remnant(pcdm);
            Tools.WriteToMeltCSV(n132_2.Item1, Path.Combine(outputFolder, "PCDM-N132.csv"), n132.Item2, n132.Item3);
            var calibration2 = Tools.LMC.CutCalibration(pcdm);
            Tools.WriteToMeltCSV(calibration2.Item1, Path.Combine(outputFolder, "PCDM-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    }
}
