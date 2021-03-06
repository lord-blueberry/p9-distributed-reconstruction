﻿using System;
using System.Collections.Generic;
using System.Text;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using Core.Deconvolution.ToyImplementations;
using System.Numerics;
using static System.Math;
using static Core.Common;
using System.Diagnostics;
using System.IO;


namespace Core
{
    public class DebugMethods
    {
        #region full
        public static void MeerKATFull()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
            var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            for(int i = 1; i < 8; i++)
            {
                var uvw0 = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw" + i + ".fits");
                var flags0 = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, norm);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            /*
            var frequencies = FitsIO.ReadFrequencies(@"freq.fits");
            var uvw = FitsIO.ReadUVW("uvw0.fits");
            var flags = FitsIO.ReadFlags("flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities("vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);
            */
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

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();
            watchTotal.Start();

            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
            var psf = IDG.CalculatePSF(c, metadata, uvw, flags, frequencies);
            FitsIO.Write(psf, "psf.fits");
            var psfCut = CutImg(psf, 2);
            FitsIO.Write(psfCut, "psfCut.fits");
            var maxSidelobe = CommonDeprecated.PSF.CalcMaxSidelobe(psf);
            var psfCorrelated = CommonDeprecated.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);

            var xImage = new double[gridSize, gridSize];
            var residualVis = visibilities;
            var maxCycle = 2;
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                watchForward.Start();
                var dirtyImage = IDG.ToImage(c, metadata, residualVis, uvw, frequencies);
                watchForward.Stop();
                FitsIO.Write(dirtyImage, "dirty" + cycle + ".fits");

                watchDeconv.Start();
                var sideLobe = maxSidelobe * GetMax(dirtyImage);
                Console.WriteLine("sideLobeLevel: " + sideLobe);
                var b = CommonDeprecated.Residuals.CalculateBMap(dirtyImage, psfCorrelated, psfCut.GetLength(0), psfCut.GetLength(1));
                var lambda = 0.8;
                var alpha = 0.05;
                var currentLambda = Math.Max(1.0 / alpha * sideLobe, lambda);
                var converged = SerialCDReference.DeconvolvePath(xImage, b, psfCut, currentLambda, 4.0, alpha, 5, 1000, 2e-5);
                //var converged = GreedyCD2.Deconvolve(xImage, b, psfCut, currentLambda, alpha, 5000);
                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!! with lambda " + currentLambda + "------------------------");
                else
                    Console.WriteLine("-------------------------------not converged with lambda " + currentLambda + "----------------------");
                
                watchDeconv.Stop();
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");

                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = Visibilities.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();
            }
            watchBackwards.Stop();
            watchTotal.Stop();

            var timetable = "total elapsed: " + watchTotal.Elapsed;
            timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
            timetable += "\n" + "idg backwards elapsed: " + watchBackwards.Elapsed;
            timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
            File.WriteAllText("watches_single.txt", timetable);
        }

        public static void DebugSimulatedMixed()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_mixed\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_mixed\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_mixed\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);
            var visibilitiesCount = visibilities.Length;

            int gridSize = 1024;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 512;
            double scaleArcSec = 0.5 / 3600.0 * PI / 180.0;

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();
            watchTotal.Start();

            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
            var psf = IDG.CalculatePSF(c, metadata, uvw, flags, frequencies);
            FitsIO.Write(psf, "psf.fits");
            var psfCut = CutImg(psf, 2);
            FitsIO.Write(psfCut, "psfCut.fits");
            var maxSidelobe = CommonDeprecated.PSF.CalcMaxSidelobe(psf);

            var xImage = new double[gridSize, gridSize];
            var residualVis = visibilities;
            var maxCycle = 10;
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                watchForward.Start();
                var dirtyImage = IDG.ToImage(c, metadata, residualVis, uvw, frequencies);
                watchForward.Stop();
                FitsIO.Write(dirtyImage, "dirty" + cycle + ".fits");

                watchDeconv.Start();
                var sideLobe = maxSidelobe * GetMax(dirtyImage);
                Console.WriteLine("sideLobeLevel: " + sideLobe);
                var PsfCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);
                var b = CommonDeprecated.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));
                var lambda = 100.0;
                var alpha = 0.95;
                var currentLambda = Math.Max(1.0 / alpha * sideLobe, lambda);
                var converged = SerialCDReference.DeconvolvePath(xImage, b, psfCut, currentLambda, 5.0, alpha, 5, 6000, 1e-3);
                //var converged = GreedyCD2.Deconvolve(xImage, b, psfCut, currentLambda, alpha, 5000);
                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!! with lambda " + currentLambda + "------------------------");
                else
                    Console.WriteLine("-------------------------------not converged with lambda " + currentLambda + "----------------------");

                watchDeconv.Stop();
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");

                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = Visibilities.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();
            }
            watchBackwards.Stop();
            watchTotal.Stop();

            var timetable = "total elapsed: " + watchTotal.Elapsed;
            timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
            timetable += "\n" + "idg backwards elapsed: " + watchBackwards.Elapsed;
            timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
            File.WriteAllText("watches_single.txt", timetable);
        }

        public static void DebugSimulatedApprox()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var visibilitiesCount = visibilities.Length;
            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();

            watchTotal.Start();
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);
            var psfCut = PSF.Cut(psf);
            FitsIO.Write(psfCut, "psfCut.fits");

            var random = new Random(123);
            var totalSize = new Rectangle(0, 0, gridSize, gridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize), new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var fastCD = new FastSerialCD(totalSize, psfCut);
            //fastCD.ResetAMap(psf);
            var lambda = 0.5f * fastCD.MaxLipschitz;
            var alpha = 0.8f;
            var approx = new ApproxParallel();
            var approx2 = new ApproxFast(totalSize, psfCut, 4, 8, 0f, 0.25f,false,true);

            var xImage = new float[gridSize, gridSize];
            var residualVis = visibilities;

            /*var truth = new double[gridSize, gridSize];
            truth[30, 30] = 1.0;
            truth[35, 36] = 1.5;
            var truthVis = IDG.ToVisibilities(c, metadata, truth, uvw, frequencies);
            visibilities = truthVis;
            var residualVis = truthVis;*/
            var data = new ApproxFast.TestingData(new StreamWriter("approxConvergence.txt"));
            for (int cycle = 0; cycle < 4; cycle++)
            {
                //FORWARD
                watchForward.Start();
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "dirty_" + cycle + ".fits");
                watchForward.Stop();

                //DECONVOLVE
                watchDeconv.Start();
                //approx.ISTAStep(xImage, dirtyImage, psf, lambda, alpha);
                //FitsIO.Write(xImage, "xIsta.fits");
                //FitsIO.Write(dirtyImage, "dirtyFista.fits");
                //bMapCalculator.ConvolveInPlace(dirtyImage);
                //FitsIO.Write(dirtyImage, "bMap_" + cycle + ".fits");
                //var result = fastCD.Deconvolve(xImage, dirtyImage, 0.5f * fastCD.MaxLipschitz, 0.8f, 1000, 1e-4f);
                //var converged = approx.DeconvolveActiveSet(xImage, dirtyImage, psfCut, lambda, alpha, random, 8, 1, 1);
                //var converged = approx.DeconvolveGreedy(xImage, dirtyImage, psfCut, lambda, alpha, random, 4, 4, 500);
                //var converged = approx.DeconvolveApprox(xImage, dirtyImage, psfCut, lambda, alpha, random, 1, threads, 500, 1e-4f, cycle == 0);
                
                approx2.DeconvolveTest(data, cycle, 0, xImage, dirtyImage, psfCut, psf, lambda, alpha, random, 10, 1e-4f);


                if (data.converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                else
                    Console.WriteLine("-------------------------------not converged----------------------");
                FitsIO.Write(xImage, "xImageApprox_" + cycle + ".fits");
                watchDeconv.Stop();

                //BACKWARDS
                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = Visibilities.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();
            }
            

            var dirtyGridCheck = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
            var dirtyCheck = FFT.Backward(dirtyGridCheck, c.VisibilitiesCount);
            FFT.Shift(dirtyCheck);

            var l2Penalty = Residuals.CalcPenalty(ToFloatImage(dirtyCheck));
            var elasticPenalty = ElasticNet.CalcPenalty(xImage, (float)lambda, (float)alpha);
            var sum = l2Penalty + elasticPenalty;

            data.writer.Close();
        }

        public static void DebugILGPU()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var visibilitiesCount = visibilities.Length;
            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();

            watchTotal.Start();
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf = FFT.Backward(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            var psfCutDouble = CutImg(psf);
            var psfCut = ToFloatImage(psfCutDouble);
            FitsIO.Write(psfCut, "psfCut.fits");

            
            var totalSize = new Rectangle(0, 0, gridSize, gridSize);
            var imageSection = new Rectangle(0, 128, gridSize, gridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psfCut, totalSize) , new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            var fastCD = new FastSerialCD(totalSize, psfCut);
            fastCD.ResetLipschitzMap(ToFloatImage(psf));
            var gpuCD = new GPUSerialCD(totalSize, psfCut, 100);
            var lambda = 0.5f * fastCD.MaxLipschitz;
            var alpha = 0.8f;

            var xImage = new float[gridSize, gridSize];
            var residualVis = visibilities;

            /*var truth = new double[gridSize, gridSize];
            truth[30, 30] = 1.0;
            truth[35, 36] = 1.5;
            var truthVis = IDG.ToVisibilities(c, metadata, truth, uvw, frequencies);
            visibilities = truthVis;
            var residualVis = truthVis;*/
            for (int cycle = 0; cycle < 4; cycle++)
            {
                //FORWARD
                watchForward.Start();
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "dirty_" + cycle + ".fits");
                watchForward.Stop();

                //DECONVOLVE
                watchDeconv.Start();
                bMapCalculator.ConvolveInPlace(dirtyImage);
                FitsIO.Write(dirtyImage, "bMap_" + cycle + ".fits");
                //var result = fastCD.Deconvolve(xImage, dirtyImage, lambda, alpha, 1000, 1e-4f);
                var result = gpuCD.Deconvolve(xImage, dirtyImage, lambda, alpha, 1000, 1e-4f);

                if (result.Converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                else
                    Console.WriteLine("-------------------------------not converged----------------------");
                FitsIO.Write(xImage, "xImageGreedy" + cycle + ".fits");
                FitsIO.Write(dirtyImage, "residualDebug_" + cycle + ".fits");
                watchDeconv.Stop();

                //BACKWARDS
                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = Visibilities.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();

                var hello = FFT.Forward(xImage, 1.0);
                hello = Common.Fourier2D.Multiply(hello, psfGrid);
                var hImg = FFT.Backward(hello, (double)(128 * 128));
                //FFT.Shift(hImg);
                FitsIO.Write(hImg, "modelDirty_FFT.fits");

                var imgRec = IDG.ToImage(c, metadata, modelVis, uvw, frequencies);
                FitsIO.Write(imgRec, "modelDirty" + cycle + ".fits");
            }
        }

        public static void DebugdWStack()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
            var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            for (int i = 1; i < 8; i++)
            {
                var uvw0 = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw" + i + ".fits");
                var flags0 = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, norm);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            var maxW = 0.0;
            for (int i = 0; i < uvw.GetLength(0); i++)
                for (int j = 0; j < uvw.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(uvw[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, frequencies[frequencies.Length - 1]);

            var visCount2 = 0;
            for (int i = 0; i < flags.GetLength(0); i++)
                for (int j = 0; j < flags.GetLength(1); j++)
                    for (int k = 0; k < flags.GetLength(2); k++)
                        if (!flags[i, j, k])
                            visCount2++;
            var visibilitiesCount = visCount2;
            int gridSize = 4096;
            int subgridsize = 16;
            int kernelSize = 8;
            int max_nr_timesteps = 1024;
            double cellSize = 1.6 / 3600.0 * PI / 180.0;
            int wLayerCount = 32;
            double wStep = maxW / (wLayerCount);
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
            var c2 = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psfVis = new Complex[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            for (int i = 0; i < visibilities.GetLength(0); i++)
                for (int j = 0; j < visibilities.GetLength(1); j++)
                    for (int k = 0; k < visibilities.GetLength(2); k++)
                        if (!flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);

            var psfGrid = IDG.GridW(c, metadata, psfVis, uvw, frequencies);
            var psf = FFT.WStackIFFTFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            FitsIO.Write(psf, "psfWStack.fits");

            var totalSize = new Rectangle(0, 0, gridSize, gridSize);
            var bMapCalculator = new PaddedConvolver(PSF.CalcPaddedFourierCorrelation(psf, totalSize), new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
            var fastCD = new FastSerialCD(totalSize, psf);
            var lambda = 0.4f * fastCD.MaxLipschitz;
            var alpha = 0.1f;

            var xImage = new float[gridSize, gridSize];
            var residualVis = visibilities;
            for (int cycle = 0; cycle < 8; cycle++)
            {
                var dirtyGrid = IDG.GridW(c, metadata, residualVis, uvw, frequencies);
                var dirty = FFT.WStackIFFTFloat(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirty);

                FitsIO.Write(dirty, "dirty_" + cycle + ".fits");
                bMapCalculator.ConvolveInPlace(dirty);
                FitsIO.Write(dirty, "bMap_" + cycle + ".fits");
                var result = fastCD.Deconvolve(xImage, dirty, lambda, alpha, 10000, 1e-4f);

                FitsIO.Write(xImage, "xImageGreedy" + cycle + ".fits");

                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGridW(c, metadata, xGrid, uvw, frequencies);
                var modelGrid = IDG.GridW(c, metadata, modelVis, uvw, frequencies);
                var model = FFT.WStackIFFTFloat(modelGrid, c.VisibilitiesCount);
                FFT.Shift(model);
                FitsIO.Write(model, "model_" + cycle + ".fits");
                residualVis = Visibilities.Substract(visibilities, modelVis, flags);
            }
        }
        #endregion




        #region helpers
        private static double[,] CutImg(double[,] image, int factor = 2)
        {
            var output = new double[image.GetLength(0) / factor, image.GetLength(1) / factor];
            var yOffset = image.GetLength(0) / 2 - output.GetLength(0) / 2;
            var xOffset = image.GetLength(1) / 2 - output.GetLength(1) / 2;

            for (int y = 0; y < output.GetLength(0); y++)
                for (int x = 0; x < output.GetLength(0); x++)
                    output[y, x] = image[yOffset + y, xOffset + x];
            return output;
        }

        public static double GetMax(double[,] image)
        {
            double max = 0.0;
            for (int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    max = Math.Max(max, image[y, x]);
            return max;
        }

        public static double CalcAvgSidelobe(double[,] fullPsf, int cutFactor = 2)
        {
            var yOffset = fullPsf.GetLength(0) / 2 - (fullPsf.GetLength(0) / cutFactor) / 2;
            var xOffset = fullPsf.GetLength(1) / 2 - (fullPsf.GetLength(1) / cutFactor) / 2;

            double output = 0.0;
            int count = 0;
            for (int y = 0; y < fullPsf.GetLength(0); y++)
                for (int x = 0; x < fullPsf.GetLength(1); x++)
                    if (!(y >= yOffset & y < (yOffset + fullPsf.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + fullPsf.GetLength(1) / cutFactor)))
                    {
                        output += Math.Abs(fullPsf[y, x]);
                        count++;
                    }

            return output / count;
        }

        public static int MaskPixels(double[,] image, double cutOff)
        {
            var zeroCount = 0;
            for(int y = 0; y < image.GetLength(0);y++)
                for(int x = 0; x < image.GetLength(1);x++)
                    if(Math.Abs(image[y,x]) < cutOff)
                    {
                        image[y, x] = 0.0;
                        zeroCount++;
                    }
            return zeroCount;
        }
        #endregion


    }
}
