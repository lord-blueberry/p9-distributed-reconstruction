using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using System.Numerics;
using static System.Math;


namespace Single_Reference
{
    class DebugMethods
    {
        #region IDG test
        public static void debie()
        {
            int max_nr_timesteps = 256;
            int gridSize = 16;
            int subgridsize = 8;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

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

        public static void DebugForwardBackward2()
        {
            int max_nr_timesteps = 256;
            int gridSize = 32;
            int subgridsize = 8;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

            double v = -50;
            double wavelength = -4 / imageSize / v;
            double u = 4 / imageSize / wavelength;
            double freq = wavelength * MathFunctions.SPEED_OF_LIGHT;
            double[] frequency = { freq };
            double u1 = 10 / imageSize / wavelength;
            double u2 = 9 / imageSize / wavelength;

            double visR0 = 3.9;
            double visR1 = 4.0;

            var visibilities = new Complex[1, 2, 1];
            visibilities[0, 0, 0] = new Complex(visR0, 0);
            visibilities[0, 1, 0] = new Complex(visR1, 0);

            var uvw = new double[1, 2, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = 0;
            uvw[0, 1, 0] = u1;
            uvw[0, 1, 1] = v;
            uvw[0, 1, 2] = 0;

            var subgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridsize, subgridsize);
            var metadata = Partitioner.CreatePartition(p, uvw, frequency);

            var gridded_subgrids = Gridder.ForwardHack(p, metadata, uvw, visibilities, frequency, subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(p, gridded_subgrids);
            var grid = Adder.AddHack(p, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);

            FFT.Shift(img);
            var grid2 = FFT.GridFFT(img);
            FFT.Shift(grid2);
            var ftGridded2 = Adder.SplitHack(p, metadata, grid2);
            var subgrids2 = FFT.SubgridIFFT(p, ftGridded2);
            var visibilities2 = Gridder.BackwardsHack(p, metadata, subgrids2, uvw, frequency, subgridSpheroidal);
        }

        public static void DebugForwardBackwardRealWorld()
        {
            int max_nr_timesteps = 256;
            int gridSize = 16;
            int subgridsize = 8;
            int kernelSize = 6;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);
  
            double[] frequency = { 857000000 };
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

            var visCount = 1;
            var subgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridsize, subgridsize);
            var metadata = Partitioner.CreatePartition(p, uvw, frequency);

            var gridded_subgrids = Gridder.ForwardHack(p, metadata, uvw, visibilities, frequency, subgridSpheroidal);
            
            var ftgridded = FFT.SubgridFFT(p, gridded_subgrids);
            var grid = Adder.AddHack(p, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);

            FFT.Shift(img);
            var grid2 = FFT.GridFFT(img, visCount);
            FFT.Shift(grid2);
            var ftGridded2 = Adder.SplitHack(p, metadata, grid2);
            var subgrids2 = FFT.SubgridIFFT(p, ftGridded2);
            
            var visibilities2 = Gridder.BackwardsHack(p, metadata, subgrids2, uvw, frequency, subgridSpheroidal);
        }
        #endregion

        #region full
        public static void DebugFullPipeline()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\uvw.fits");
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            var visibilitiesCount = visibilities.Length;

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

            int gridSize = 256;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 256;
            double cellSize = 0.5 / 3600.0 * PI / 180.0;

            var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            //visibilitiesCount = 1;
            var psf = IDG.CalculatePSF(c, metadata, uvw, frequencies, visibilitiesCount);
            var image = IDG.ToImage(c, metadata, visibilities, uvw, frequencies);
            var psfVis = IDG.ToVisibilities(c, metadata, psf, uvw, frequencies);
            var psf2 = CutImg(psf);
            FitsIO.Write(image, "dirty.fits");
            FitsIO.Write(psf, "psf.fits");
            
            var reconstruction = new double[gridSize, gridSize];
            CDClean.Deconvolve(reconstruction, image, psf2, 2.0, 5);
            FitsIO.Write(reconstruction, "reconstruction.fits");
            FitsIO.Write(image, "residual.fits");
            CDClean.Deconvolve(reconstruction, image, psf2, 1.0, 5);
            FitsIO.Write(reconstruction, "reconstruction.fits");
            FitsIO.Write(image, "residual.fits");

            var vis2 = IDG.ToVisibilities(c, metadata, image, uvw, frequencies);
            var diffVis = Substract(visibilities, vis2);
            var image2 = IDG.ToImage(c, metadata, diffVis, uvw, frequencies);
            var vis3 = IDG.ToVisibilities(c, metadata, image2, uvw, frequencies);
        }
        #endregion

        #region helpers
        private static double[,] CutImg(double[,] image)
        {
            var output = new double[128, 128];
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

        private static Complex[,,] Substract(Complex[,,] vis0, Complex[,,] vis1)
        {
            var output = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        output[i, j, k] = vis0[i, j, k] - vis1[i, j, k];
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
        #endregion

        public static void TestConvergence0()
        {
            var imSize = 64;
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
            groundTruth[33, 33] = 15.0;

            var image = Convolve(groundTruth, psf);
            var reconstruction = new double[imSize, imSize];
            CDClean.Deconvolve(reconstruction, image, psf, 0.1);

            var precision = 0.1;
        }

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

            var precision = 0.1;
        }

        private static double ComputeL(int x, int subgridSize, float imageSize)
        {
            return (x - (subgridSize / 2)) * imageSize / subgridSize;
        }

    }
}
