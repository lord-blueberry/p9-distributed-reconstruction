using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Single_Machine.IDG;
using System.Numerics;
using static System.Math;
using System.IO;
using Single_Machine.Deconvolution;

using nom.tam.fits;
using nom.tam.util;

namespace Single_Machine
{
    class Debug
    {
        #region idg
        public static void SingleVisibility2()
        {
            /*  baseline 1036
                timestep 1
                channel 0 */
            double[] frequency = { 857000000f, 857000000f };

            //only xx polarization
            double visR = 3.8931689262390137;
            double visI = 0.061203371733427048;

            double u = -9.3063146568965749;
            double v = (-1) * -35.529046011622995;
            double w = 0;

            int nr_timeslots = 1;
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 64;
            int kernelSize = 16;
            float properCellSize = (float)(2.0 / 3600.0 * PI / 180.0);
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, properCellSize, 1, 0.0f);

            var visibilities = new Complex[1, 1, 2];
            visibilities[0, 0, 0] = new Complex(visR, visI);
            visibilities[0, 0, 1] = 0;
            var uvw = new double[1, 1, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = w;

            var subgridSpheroidal = IDG.Math.CalcIdentitySpheroidal(subgridsize, subgridsize);

            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, visibilities, frequency, subgridSpheroidal);
            var imgg = gridded[0][0];
            var ftgridded = FFT.SubgridFFT(p, gridded);
            var grid = Adder.AddHack(p, subgrids, ftgridded);
            //Write(grid);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);
            Write(img);
        }

        public static void SingleSubgrid()
        {
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 48;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

            double v = -50;
            double wavelength = -4 / imageSize / v;
            double u = 4 / imageSize / wavelength;
            double freq = wavelength * IDG.Math.SPEED_OF_LIGHT;
            double[] frequency = { freq };
            double u1 = 10 / imageSize / wavelength;

            double visR0 = 3.9;
            double visR1 = 5.2;

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


            var ift1 = IFT(new Complex(visR0, 0), u, v, freq, gridSize, imageSize);
            var ift2 = IFT(new Complex(visR1, 0), u1, v, freq, gridSize, imageSize);
            Add(ift1, ift2);
            Write(ift1, "iftOutput.fits");
            var fourierFT = FFT.GridFFT(ift1);
            Write(fourierFT, "iftOutput.fits");
            WriteImag(fourierFT);


            var subgridSpheroidal = IDG.Math.CalcIdentitySpheroidal(subgridsize, subgridsize);


            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, visibilities, frequency, subgridSpheroidal);

            var ftgridded = FFT.SubgridFFT(p, gridded);

            var img2 = ftgridded[0][0];
            Write(img2);
            WriteImag(img2);

            var grid = Adder.AddHack(p, subgrids, ftgridded);

            FFT.Shift(grid);
            Write(grid);
            WriteImag(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);
            Write(img);
        }
        #endregion

        public static void DebugCD()
        {
            var imSize = 16;
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

            var image = new double[imSize, imSize];
            image[9, 9] = 15.0;
            image[9, 8] = 2.0;
            var conv = Convolve(image, psf);

            var xImage = new double[imSize, imSize];
            CDClean.CoordinateDescent(xImage, conv, psf, 0.1, 100);
        }

        #region debug full pipeline
        public static void DebugFullPipeline()
        {
            #region reading
            Fits f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\freq.fits");
            ImageHDU h = (ImageHDU)f.ReadHDU();
            var frequencies = (double[])h.Kernel;
            f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\uvw.fits");
            h = (ImageHDU)f.ReadHDU();

            // Double Cube Dimensions: baseline, time, uvw
            var uvw_raw = (Array[])h.Kernel;
            var baselines = uvw_raw.Length;
            var time_samples = uvw_raw[0].Length;

            var uvw = new double[baselines, time_samples, 3];
            for (int i = 0; i < baselines; i++)
            {
                Array[] bl = (Array[])uvw_raw[i];
                for (int j = 0; j < time_samples; j++)
                {
                    double[] values = (double[])bl[j];
                    uvw[i, j, 0] = values[0]; //u
                    uvw[i, j, 1] = -values[1]; //v
                    uvw[i, j, 2] = values[2]; //w
                }

            }

            f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\vis.fits");
            h = (ImageHDU)f.ReadHDU();
            //Double cube Dimensions: baseline, time, channel, pol, complex_component
            var vis_raw = (Array[])h.Kernel;
            var channels = 8;
            var visibilities = new Complex[baselines, time_samples, channels];
            long visibilitiesCount = 0;
            for (int i = 0; i < baselines; i++)
            {
                Array[] bl = (Array[])vis_raw[i];
                for (int j = 0; j < time_samples; j++)
                {
                    Array[] times = (Array[])bl[j];
                    for (int k = 0; k < channels; k++)
                    {
                        Array[] channel = (Array[])times[k];
                        double[] pol_XX = (double[])channel[0];
                        double[] pol_YY = (double[])channel[3];

                        //add polarizations XX and YY together to form Intensity Visibilities only
                        visibilities[i, j, k] = new Complex(
                            (pol_XX[0] + pol_YY[0]) / 2.0,
                            (pol_XX[1] + pol_YY[1]) / 2.0
                            );
                        visibilitiesCount++;
                    }
                }
            }

            int gridSize = 256;
            int subgridsize = 16;
            int kernelSize = 4;
            //cell = image / grid
            int max_nr_timesteps = 256;
            double cellSize = 0.5 / 3600.0 * PI / 180.0;
            #endregion

            var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psf = NUFFT.CalculatePSF(c, metadata, uvw, frequencies, visibilitiesCount);
            var image = NUFFT.ToImage(c, metadata, visibilities, uvw, frequencies, visibilitiesCount);
            var psf2 = CutImg(psf);
            Write(image, "dirty.fits");
            var reconstruction = new double[gridSize, gridSize];
            CDClean.CoordinateDescent(reconstruction, image, psf2, 3.0, 10);
            Write(reconstruction, "reconstruction.fits");
            Write(image, "residual.fits");
            CDClean.CoordinateDescent(reconstruction, image, psf2, 2.5, 10);
            Write(reconstruction, "reconstruction.fits");
            Write(image, "residual.fits");

            var vis2 = NUFFT.ToVisibilities(c, metadata, image, uvw, frequencies, visibilitiesCount);
            var diffVis = Substract(visibilities, vis2);
            var image2 = NUFFT.ToImage(c, metadata, diffVis, uvw, frequencies, visibilitiesCount);
            var vis3 = NUFFT.ToVisibilities(c, metadata, image2, uvw, frequencies, visibilitiesCount);

        }
        #endregion

        #region helpers
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

        public static Complex[,,] Substract(Complex[,,] vis0, Complex[,,] vis1)
        {
            var output = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        output[i, j, k] = vis0[i, j, k] - vis1[i, j, k];
            return output;
        }

        public static Complex[,,] Add(Complex[,,] vis0, Complex[,,] vis1)
        {
            var output = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        output[i, j, k] = vis0[i, j, k] + vis1[i, j, k];
            return output;
        }


        public static void Add(Complex[,] c0, Complex[,] c1)
        {
            for (int i = 0; i < c0.GetLength(0); i++)
            {
                for (int j = 0; j < c0.GetLength(1); j++)
                    c0[i, j] += c1[i, j];
            }
        }

        public static Complex[,] IFT(Complex vis, double u, double v, double freq, int gridSize, double imageSize)
        {
            u = u * freq / IDG.Math.SPEED_OF_LIGHT;
            v = v * freq / IDG.Math.SPEED_OF_LIGHT;

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

        #region fits output
        public static void Write(Complex[,] img, string file = "Outputfile.fits")
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j].Real;
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file, FileMode.Create)))
            {
                f.Write(fstream);
            }
        }
        public static void Write(double[,] img, string file = "Outputfile.fits")
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j];
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file, FileMode.Create)))
            {
                f.Write(fstream);
            }
        }

        public static void WriteImag(Complex[,] img)
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j].Imaginary;
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream("Outputfile_imag.fits", FileMode.Create)))
            {
                f.Write(fstream);
            }
        }

        private static double[,] CutImg(double[,] image)
        {
            var output = new double[128, 128];
            var yOffset = image.GetLength(0) / 2 - output.GetLength(0) / 2;
            var xOffset = image.GetLength(1) / 2 - output.GetLength(1) / 2;

            for (int y = 0; y < output.GetLength(0); y++)
                for (int x = 0; x < output.GetLength(0); x++)
                    output[y, x] = image[yOffset+y, xOffset + x];
            return output;
        }
        #endregion
    }
}
