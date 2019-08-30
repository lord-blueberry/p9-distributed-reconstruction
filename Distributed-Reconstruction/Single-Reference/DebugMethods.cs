using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using Single_Reference.Deconvolution.ToyImplementations;
using System.Numerics;
using static System.Math;
using static Single_Reference.Common;
using System.Diagnostics;
using System.IO;


namespace Single_Reference
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
            var maxSidelobe = PSF.CalcMaxSidelobe(psf);
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
                var b = Common.Residuals.CalculateBMap(dirtyImage, psfCorrelated, psfCut.GetLength(0), psfCut.GetLength(1));
                var lambda = 0.8;
                var alpha = 0.05;
                var currentLambda = Math.Max(1.0 / alpha * sideLobe, lambda);
                var converged = GreedyCD2.DeconvolvePath(xImage, b, psfCut, currentLambda, 4.0, alpha, 5, 1000, 2e-5);
                //var converged = GreedyCD2.Deconvolve(xImage, b, psfCut, currentLambda, alpha, 5000);
                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!! with lambda " + currentLambda + "------------------------");
                else
                    Console.WriteLine("-------------------------------not converged with lambda " + currentLambda + "----------------------");
                
                watchDeconv.Stop();
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");

                watchBackwards.Start();
                var modelVis = IDG.ToVisibilities(c, metadata, xImage, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
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
            var maxSidelobe = Common.PSF.CalcMaxSidelobe(psf);

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
                var b = Common.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));
                var lambda = 100.0;
                var alpha = 0.95;
                var currentLambda = Math.Max(1.0 / alpha * sideLobe, lambda);
                var converged = GreedyCD2.DeconvolvePath(xImage, b, psfCut, currentLambda, 5.0, alpha, 5, 6000, 1e-3);
                //var converged = GreedyCD2.Deconvolve(xImage, b, psfCut, currentLambda, alpha, 5000);
                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!! with lambda " + currentLambda + "------------------------");
                else
                    Console.WriteLine("-------------------------------not converged with lambda " + currentLambda + "----------------------");

                watchDeconv.Stop();
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");

                watchBackwards.Start();
                var modelVis = IDG.ToVisibilities(c, metadata, xImage, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
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

        public static void DebugSimulatedGreedy2()
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
            
            var psfCut = CutImg(psf);
            FitsIO.Write(psf, "psf.fits");
            FitsIO.Write(psfCut, "psfCut.fits");

            var xImage = new double[gridSize, gridSize];
            var residualVis = visibilities;
            /*var truth = new double[gridSize, gridSize];
            truth[30, 30] = 1.0;
            truth[35, 36] = 1.5;
            var truthVis = IDG.ToVisibilities(c, metadata, truth, uvw, frequencies);
            visibilities = truthVis;
            var residualVis = truthVis;*/
            for (int cycle = 0; cycle < 1; cycle++)
            {
                //FORWARD
                watchForward.Start();
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
                var dirtyImage = FFT.Backward(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "dirty_" + cycle + ".fits");
                watchForward.Stop();

                //DECONVOLVE
                watchDeconv.Start();

                var PsfCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);
                var b = Common.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));
                var converged = GreedyCD2.Deconvolve(xImage, b, psfCut, 0.0, 1.0, 10000);

                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                else
                    Console.WriteLine("-------------------------------not converged----------------------");
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");
                FitsIO.Write(dirtyImage, "residualDebug_" + cycle + ".fits");
                watchDeconv.Stop();

                //BACKWARDS
                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();

                var imgRec = IDG.ToImage(c, metadata, modelVis, uvw, frequencies);
                FitsIO.Write(imgRec, "modelDirty" + cycle + ".fits");
            }
        }

        public static void DebugSimulatedPCDM()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var visibilitiesCount = visibilities.Length;
            int gridSize = 128;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 2.0 / 3600.0 * PI / 180.0;
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
            var maxSidelobe = Common.PSF.CalcMaxSidelobe(psf);
            var avgSidelobe = CalcAvgSidelobe(psf);

            var psfCut = CutImg(psf);
            //MaskPixels(psfCut, avgSidelobe);
            

            FitsIO.Write(psf, "psf.fits");
            FitsIO.Write(psfCut, "psfCut.fits");

            var xImage = new double[gridSize, gridSize];
            //var residualVis = visibilities;

            var truth = new double[gridSize, gridSize];
            truth[64, 64] = 1.0;
            //truth[64, 65] = 1.5;
            var truthVis = IDG.ToVisibilities(c, metadata, truth, uvw, frequencies);
            visibilities = truthVis;
            var residualVis = truthVis;
            for (int cycle = 0; cycle < 5; cycle++)
            {
                //FORWARD
                watchForward.Start();
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
                var dirtyImage = FFT.Backward(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "dirty_" + cycle + ".fits");
                watchForward.Stop();

                //DECONVOLVE
                watchDeconv.Start();

                var PsfCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);
                var b = Common.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));
                //var converged = GradientDebug.Deconvolve(xImage, b, psfCut, 0.0, 1.0, 10000);
                var converged = RandomBlockCD.Deconvolve(xImage, dirtyImage, psf, 0.0, 1.0, 1);

                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                else
                    Console.WriteLine("-------------------------------not converged----------------------");
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");
                FitsIO.Write(b, "bMap_" + cycle + ".fits");
                watchDeconv.Stop();

                //BACKWARDS
                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();

                var imgRec = IDG.ToImage(c, metadata, modelVis, uvw, frequencies);
                FitsIO.Write(imgRec, "modelDirty" + cycle + ".fits");
            }
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

            var psfCut = CutImg(psf);
            FitsIO.Write(psf, "psf.fits");
            FitsIO.Write(psfCut, "psfCut.fits");

            var xImage = new double[gridSize, gridSize];
            var residualVis = visibilities;
            /*var truth = new double[gridSize, gridSize];
            truth[30, 30] = 1.0;
            truth[35, 36] = 1.5;
            var truthVis = IDG.ToVisibilities(c, metadata, truth, uvw, frequencies);
            visibilities = truthVis;
            var residualVis = truthVis;*/
            for (int cycle = 0; cycle < 1; cycle++)
            {
                //FORWARD
                watchForward.Start();
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, uvw, frequencies);
                var dirtyImage = FFT.Backward(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                FitsIO.Write(dirtyImage, "dirty_" + cycle + ".fits");
                watchForward.Stop();

                //DECONVOLVE
                watchDeconv.Start();

                var PsfCorrelation = CommonDeprecated.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);
                var b = Common.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));

                //var converged = GPUDeconvolution.GreedyCD2.Deconvolve(xImage, b, psfCut, 0.5, 0.20);

                //var deconvolver = new GPUDeconvolution.GPUGreedyCD();
                //deconvolver.DeconvolvePath(null, null, null, null, 0.5f, 0.5f, 1, 10, 100);

                var converged = GPUDeconvolution.StupidGreedy.Deconvolve(xImage, b, psfCut, 0.5, 0.20);

                if (converged)
                    Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                else
                    Console.WriteLine("-------------------------------not converged----------------------");
                FitsIO.Write(xImage, "xImage_" + cycle + ".fits");
                FitsIO.Write(dirtyImage, "residualDebug_" + cycle + ".fits");
                watchDeconv.Stop();

                //BACKWARDS
                watchBackwards.Start();
                FFT.Shift(xImage);
                var xGrid = FFT.Forward(xImage);
                FFT.Shift(xImage);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
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

        public static void DebugSimulatedWStack()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var maxW = 0.0;
            for (int i = 0; i < uvw.GetLength(0); i++)
                for (int j = 0; j < uvw.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(uvw[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, frequencies[frequencies.Length - 1]);

            var visibilitiesCount = visibilities.Length;
            int gridSize = 2048;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 5 / 3600.0 * PI / 180.0;
            int wLayerCount = 8;
            double wStep = maxW / (wLayerCount);
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);
            var c2 = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0);
            var metadata2 = Partitioner.CreatePartition(c2, uvw, frequencies);

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();

            var psfVis = new Complex[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            for (int i = 0; i < visibilities.GetLength(0); i++)
                for (int j = 0; j < visibilities.GetLength(1); j++)
                    for (int k = 0; k < visibilities.GetLength(2); k++)
                        if (!flags[i, j, k])
                            psfVis[i, j, k] = new Complex(1.0, 0);
                        else
                            psfVis[i, j, k] = new Complex(0, 0);
            
            watchTotal.Start();
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
            var psfGrid = IDG.GridW(c, metadata, psfVis, uvw, frequencies);
            var psf = FFT.GridIFFT(psfGrid, c.VisibilitiesCount);
            //var psfGridAdded = FFT.GridFFT(psf);
            FFT.Shift(psf);

            var psf2Grid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf2 = FFT.Backward(psf2Grid, c.VisibilitiesCount);
            FFT.Shift(psf2);
            FitsIO.Write(psf2, "psf2222.fits");
            FitsIO.Write(psf, "psf");

            var psfSum = new double[gridSize, gridSize];
            for (int k = 0; k < wLayerCount; k++)
                for (int i = 0; i < gridSize; i++)
                    for (int j = 0; j < gridSize; j++)
                        psfSum[i, j] += psf[k, i, j];

            FitsIO.Write(psfSum, "psfSum.fits");

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
