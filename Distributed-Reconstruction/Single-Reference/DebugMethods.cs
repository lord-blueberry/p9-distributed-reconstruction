using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDG;
using Single_Reference.Deconvolution;
using System.Numerics;
using static System.Math;


namespace Single_Reference
{
    class DebugMethods
    {
        #region IDG test
        public static void DebugForwardBackward()
        {
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 16;
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
            double visR1 = 5.2;
            double visR2 = 8.2;

            var visibilities = new Complex[1, 3, 1];
            visibilities[0, 0, 0] = new Complex(visR0, 0);
            visibilities[0, 1, 0] = new Complex(visR1, 0);
            visibilities[0, 2, 0] = new Complex(visR2, 0);
            var uvw = new double[1, 3, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = 0;
            uvw[0, 1, 0] = u1;
            uvw[0, 1, 1] = v;
            uvw[0, 1, 2] = 0;
            uvw[0, 2, 0] = u2;
            uvw[0, 2, 1] = v;
            uvw[0, 2, 2] = 0;

            var visCount = 1;
            var subgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridsize, subgridsize);
            var metadata = Partitioner.CreatePartition(p, uvw, frequency);

            var gridded_subgrids = Gridder.ForwardHack(p, metadata, uvw, visibilities, frequency, subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(p, gridded_subgrids);
            var grid = Adder.AddHack(p, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, visCount);
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
            var visibilitiesCount = uvw.GetLength(0) * uvw.GetLength(1) * frequencies.Length;

            int gridSize = 256;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 256;
            double cellSize = 0.5 / 3600.0 * PI / 180.0;

            var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            //visibilitiesCount = 1;
            var psf = NUFFT.CalculatePSF(c, metadata, uvw, frequencies, visibilitiesCount);
            var image = NUFFT.ToImage(c, metadata, visibilities, uvw, frequencies, visibilitiesCount);
            var psfVis = NUFFT.ToVisibilities(c, metadata, psf, uvw, frequencies, visibilitiesCount);
            var psf2 = CutImg(psf);
            FitsIO.Write(image, "dirty.fits");
            FitsIO.Write(psf, "psf.fits");
            /*
            var reconstruction = new double[gridSize, gridSize];
            CDClean.CoordinateDescent(reconstruction, image, psf2, 2.0, 5);
            FitsIO.Write(reconstruction, "reconstruction.fits");
            FitsIO.Write(image, "residual.fits");
            CDClean.CoordinateDescent(reconstruction, image, psf2, 1.0, 5);
            FitsIO.Write(reconstruction, "reconstruction.fits");
            FitsIO.Write(image, "residual.fits");*/

            var vis2 = NUFFT.ToVisibilities(c, metadata, image, uvw, frequencies, visibilitiesCount);
            var diffVis = Substract(visibilities, vis2);
            var image2 = NUFFT.ToImage(c, metadata, diffVis, uvw, frequencies, visibilitiesCount);
            var vis3 = NUFFT.ToVisibilities(c, metadata, image2, uvw, frequencies, visibilitiesCount);
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
            CDClean.CoordinateDescent(reconstruction, image, psf, 0.1);

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
            CDClean.CoordinateDescent(reconstruction, convolved, psf, 0.1);

            var precision = 0.1;
        }
    }
}
