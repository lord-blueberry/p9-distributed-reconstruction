using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using System.Numerics;
using static System.Math;
using System.Diagnostics;
using System.IO;


namespace Single_Reference
{
    class DebugMethods
    {
        #region IDG test
        public static void DebugIDG()
        {
            int max_nr_timesteps = 256;
            int gridSize = 16;
            int subgridsize = 8;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingConstants(1, gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

            double v = -50;
            double wavelength = -4 / imageSize / v;
            double u = 4 / imageSize / wavelength;

            double[] frequency = { 857000000 };
            double u1 = 10 / imageSize / wavelength;
            double u2 = 9 / imageSize / wavelength;

            var visibilities = new Complex[1, 4, 1];
            visibilities[0, 0, 0] = new Complex(3.9, 0);
            visibilities[0, 1, 0] = new Complex(5.2, 0);
            visibilities[0, 2, 0] = new Complex(2.3, 0);
            visibilities[0, 3, 0] = new Complex(1.8, 0);
            var uvw = new double[1, 4, 3];
            uvw[0, 0, 0] = -9.3063146568965749;
            uvw[0, 0, 1] = 35.529046011622995;
            uvw[0, 0, 2] = -0.36853532493114471;
            uvw[0, 1, 0] = -9.308945244341885;
            uvw[0, 1, 1] = 35.528369050938636;
            uvw[0, 1, 2] = -0.36735843494534492;
            uvw[0, 2, 0] = -9.3063146568965749;
            uvw[0, 2, 1] = 30.529046011622995;
            uvw[0, 2, 2] = -0.36735843494534492;
            uvw[0, 3, 0] = -9.308945244341885;
            uvw[0, 3, 1] = 30.528369050938636;
            uvw[0, 3, 2] = -0.36735843494534492;

            var I = new Complex(0, 1);
            var subgr = 16;
            var wavenr = MathFunctions.FrequencyToWavenumber(frequency);
            wavenr[0] = wavenr[0] ;
            var image = new Complex[subgr, subgr];
            for(int i = 0; i < visibilities.GetLength(1); i++)
                for (int y = 0; y < subgr; y++)
                    for (int x = 0; x < subgr; x++)
                    {
                        var l = ComputeL(x, subgr, imageSize);
                        var m = ComputeL(y, subgr, imageSize);
                        double phaseindex = uvw[0, i, 0] * l + uvw[0, i, 1] * m;
                        double phase = -(phaseindex * wavenr[0]*2*PI);
                        var cpl = new Complex(Cos(phase), Sin(phase));
                        var tmp =  Complex.Exp(-2*PI*I * wavenr[0] * (uvw[0,i,0] *l + uvw[0, i, 1] * m));
                        tmp = visibilities[0, i, 0] * tmp;
                        image[y, x] += tmp;
                    }

            var visi2 = new Complex[visibilities.GetLength(1)];
            var norm = 1.0 /(subgr * subgr);
            for (int i = 0; i < visibilities.GetLength(1); i++)
                for (int y = 0; y < subgr; y++)
                    for (int x = 0; x < subgr; x++)
                    {
                        var l = ComputeL(x, subgr, imageSize);
                        var m = ComputeL(y, subgr, imageSize);
                        double phase = ((uvw[0, i, 0] * l + uvw[0, i, 1] * m) * wavenr[0]);
                        var cpl = new Complex(Cos(phase), Sin(phase));
                        var tmp = Complex.Exp(2 * PI * I * wavenr[0] * (uvw[0, i, 0] * l + uvw[0, i, 1] * m));
                        tmp = image[y, x] * tmp;
                        visi2[i] += tmp;
                    }
            for (int i = 0; i < visibilities.GetLength(1); i++)
                visi2[i] = visi2[i] * norm;
        }
        #endregion

        #region full
        public static void DebugFullMeerKAT()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
            var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);
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
            var psf2 = CutImg(psf);

            var reconstruction = new double[gridSize, gridSize];
            var residualVis = visibilities;
            var majorCycles = 7;
            for (int cycle = 0; cycle < majorCycles; cycle++)
            {
                watchForward.Start();
                var dirtyImage = IDG.ToImage(c, metadata, residualVis, uvw, frequencies);
                watchForward.Stop();
                FitsIO.Write(dirtyImage, "dirty" + cycle + ".fits");

                watchDeconv.Start();
                CDClean.Deconvolve(reconstruction, dirtyImage, psf2, 0.1 / (10*(cycle + 1)), 5);
                int nonzero = CountNonZero(reconstruction);
                Console.WriteLine("number of nonzeros in reconstruction: " + nonzero);
                watchDeconv.Stop();
                FitsIO.Write(reconstruction, "reconstruction" + cycle + ".fits");

                watchBackwards.Start();
                var modelVis = IDG.ToVisibilities(c, metadata, reconstruction, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();

                var imgRec = IDG.ToImage(c, metadata, modelVis, uvw, frequencies);
                FitsIO.Write(imgRec, "model" + cycle + ".fits");
            }
            watchBackwards.Stop();
            watchTotal.Stop();

            var timetable = "total elapsed: " + watchTotal.Elapsed;
            timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
            timetable += "\n" + "idg backwards elapsed: " + watchBackwards.Elapsed;
            timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
            File.WriteAllText("watches_single.txt", timetable);
        }

        public static void DebugSimulated2()
        {
            /*var frequencies = FitsIO.ReadFrequencies(@"freq.fits");
            var uvw = FitsIO.ReadUVW(@"uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var visibilitiesCount = visibilities.Length;
            int gridSize = 256;
            int subgridsize = 16;
            int kernelSize = 8;
            //cell = image / grid
            int max_nr_timesteps = 256;
            double cellSize = 0.5 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();
            watchTotal.Start();

            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psf = IDG.CalculatePSF(c, metadata, uvw, flags, frequencies);
            var psf2 = psf;//CutImg(psf);
            FitsIO.Write(psf2, "psf.fits");*/

            var psf = new double[4, 4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    psf[i, j] = 1 / 32.0;
            psf[2, 2] = 17 / 32.0;

            var b = new double[16, 16];
            var resTmp = new double[4, 4];
            GreedyCD.UpdateB(b, resTmp, psf, 14, 2, 0.5);
            FitsIO.Write(b, "bOutput.fits");
            FitsIO.Write(resTmp, "resTmp.fits");

            var convTest = new double[16, 16];
            convTest[14, 2] = 0.5;
            var conv1 = Convolve(convTest, psf);
            FitsIO.Write(conv1, "conv1Output.fits");
            var conv2 = Convolve(conv1, psf);
            FitsIO.Write(conv2, "conv2Output.fits");
        }

        public static void DebugFullPipeline()
        {
            var frequencies = FitsIO.ReadFrequencies(@"freq.fits");
            var uvw = FitsIO.ReadUVW(@"uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(@"vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            var nrBaselines = uvw.GetLength(0);
            var nrFrequencies = frequencies.Length;
            var uvwtmp = new double[nrBaselines, uvw.GetLength(1), 3];
            var vistmp = new Complex[nrBaselines, uvw.GetLength(1), nrFrequencies];
            var freqtmp = new double[nrFrequencies];
            for(int i = 0; i < nrBaselines; i++)
            {
                for (int j = 0; j < uvw.GetLength(1); j++)
                {
                    for(int k = 0; k < nrFrequencies; k++)
                    {
                        vistmp[i, j, k] = visibilities[i, j, k];
                    }
                    uvwtmp[i, j, 0] = uvw[i, j, 0];
                    uvwtmp[i, j, 1] = uvw[i, j, 1];
                    uvwtmp[i, j, 2] = uvw[i, j, 2];
                }
            }

            for(int i = 0; i < nrFrequencies; i++)
            {
                freqtmp[i] = frequencies[i];
            }

            uvw = uvwtmp;
            visibilities = vistmp;
            frequencies = freqtmp;
            var visibilitiesCount = visibilities.Length;

            int gridSize = 256;
            int subgridsize = 16;
            int kernelSize = 8;
            //cell = image / grid
            int max_nr_timesteps = 256;
            double cellSize = 0.5 / 3600.0 * PI / 180.0;

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackwards = new Stopwatch();
            var watchDeconv = new Stopwatch();
            watchTotal.Start();

            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psf = IDG.CalculatePSF(c, metadata, uvw, flags, frequencies);
            var psf2 = psf;//CutImg(psf);
            FitsIO.Write(psf2, "psf.fits");

            var reconstruction = new double[gridSize, gridSize];
            var residualVis = visibilities;
            for(int cycle = 0; cycle < 7; cycle++)
            {
                watchForward.Start();
                var dirtyImage = IDG.ToImage(c, metadata, residualVis, uvw, frequencies);
                watchForward.Stop();
                FitsIO.Write(dirtyImage, "dirty"+cycle+".fits");

                watchDeconv.Start();
                CDClean.Deconvolve(reconstruction, dirtyImage, psf2, 0.02/(cycle+1), 4);
                watchDeconv.Stop();
                FitsIO.Write(reconstruction, "reconstruction"+cycle+".fits");

                watchBackwards.Start();
                var modelVis = IDG.ToVisibilities(c, metadata, reconstruction, uvw, frequencies);
                residualVis = IDG.Substract(visibilities, modelVis, flags);
                watchBackwards.Stop();

                var imgRec = IDG.ToImage(c, metadata, modelVis, uvw, frequencies);
                FitsIO.Write(imgRec, "modelDirty" + cycle + ".fits");
            }
            watchBackwards.Stop();
            watchTotal.Stop();

            var timetable = "total elapsed: " + watchTotal.Elapsed;
            timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
            timetable += "\n" + "idg backwards elapsed: " + watchBackwards.Elapsed;
            timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
            File.WriteAllText("watches_single.txt", timetable);
        }
        #endregion

        #region helpers
        private static double[,] CutImg(double[,] image)
        {
            var output = new double[image.GetLength(0) / 2, image.GetLength(1) / 2];
            var yOffset = image.GetLength(0) / 2 - output.GetLength(0) / 2;
            var xOffset = image.GetLength(1) / 2 - output.GetLength(1) / 2;

            for (int y = 0; y < output.GetLength(0); y++)
                for (int x = 0; x < output.GetLength(0); x++)
                    output[y, x] = image[yOffset + y, xOffset + x];
            return output;
        }

        private static double[,] Convolve(double[,] image, double[,] kernel)
        {

            var output = new double[image.GetLength(0), image.GetLength(1)];
            for (int y = 0; y < image.GetLength(0); y++)
            {
                for (int x = 0; x < image.GetLength(1); x++)
                {
                    double sum = 0;
                    for (int yk = 0; yk < kernel.GetLength(0); yk++)
                    {
                        for (int xk = 0; xk < kernel.GetLength(1); xk++)
                        {
                            int ySrc = y + yk - ((kernel.GetLength(0) - 1) - kernel.GetLength(0) / 2);
                            int xSrc = x + xk - ((kernel.GetLength(1) - 1) - kernel.GetLength(1) / 2);
                            if (ySrc >= 0 & ySrc < image.GetLength(0) &
                                xSrc >= 0 & xSrc < image.GetLength(1))
                            {
                                sum += image[ySrc, xSrc] * kernel[kernel.GetLength(0) - 1 - yk, kernel.GetLength(1) - 1 - xk];
                            }
                        }
                    }
                    output[y, x] = sum;
                }
            }
            return output;
        }

        private static Complex[,,] Substract(Complex[,,] vis0, Complex[,,] vis1, bool[,,] flags)
        {
            var output = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        if (!flags[i, j, k])
                            output[i, j, k] = vis0[i, j, k] - vis1[i, j, k];
                        else
                            output[i, j, k] = 0;
            return output;
        }

        private static Complex[,,] Add(Complex[,,] vis0, Complex[,,] vis1)
        {
            var output = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];

            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        output[i, j, k] = vis0[i, j, k] + vis1[i, j, k];
            return output;
        }


        private static void Add(Complex[,] c0, Complex[,] c1)
        {
            for (int i = 0; i < c0.GetLength(0); i++)
            {
                for (int j = 0; j < c0.GetLength(1); j++)
                    c0[i, j] += c1[i, j];
            }
        }

        private static Complex[,] IFT(Complex vis, double u, double v, double freq, int gridSize, double imageSize)
        {
            u = u * freq / MathFunctions.SPEED_OF_LIGHT;
            v = v * freq / MathFunctions.SPEED_OF_LIGHT;

            var output = new Complex[gridSize, gridSize];
            var I = new Complex(0, 1);
            var cell = imageSize / gridSize;
            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    int xi = x - gridSize / 2;
                    int yi = y - gridSize / 2;
                    var d = Complex.Exp(2 * PI * I * (u * (xi) * cell + v * (yi) * cell));
                    var c = vis * d;
                    output[y, x] = c;
                }
            }
            return output;
        }

        private static int CountNonZero(double[,] image)
        {
            int count = 0;
            for(int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    if (image[y, x] > 0.0)
                        count++;
            return count;
        }
        #endregion


        public static void TestConvergence1()
        {
            var imSize = 32;
            var psfSize = 4;
            var psf = new double[psfSize, psfSize];

            var psfSum = 8.0;
            psf[1, 1] = 1 / psfSum;
            psf[1, 2] = 2 / psfSum;
            psf[1, 3] = 3 / psfSum;
            psf[2, 1] = 3 / psfSum;
            psf[2, 2] = 8 / psfSum;
            psf[2, 3] = 2 / psfSum;
            psf[3, 1] = 5 / psfSum;
            psf[3, 2] = 3 / psfSum;
            psf[3, 3] = 2 / psfSum;

            var groundTruth = new double[imSize, imSize];
            groundTruth[17, 17] = 15.0;
            groundTruth[16, 17] = 3.0;
            groundTruth[15, 17] = 2.0;
            groundTruth[16, 16] = 5.0;
            var convolved = Convolve(groundTruth, psf);
            var reconstruction = new double[imSize, imSize];
            CDClean.Deconvolve(reconstruction, convolved, psf, 0.1);
        }

        private static double ComputeL(int x, int subgridSize, float imageSize)
        {
            return (x - (subgridSize / 2)) * imageSize / subgridSize;
        }

    }
}
