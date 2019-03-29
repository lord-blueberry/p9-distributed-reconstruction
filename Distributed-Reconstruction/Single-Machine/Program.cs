using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using nom.tam.fits;
using nom.tam.util;
using System.IO;

using static System.Math;
using System.Numerics;

namespace Single_Machine
{
    using IDG;
    class Program
    {
        static void Main(string[] args)
        {
            //SingleSubgrid();
            //SingleVisibility2();
            
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
            for(int i = 0; i < baselines; i++)
            {
                Array[] bl = (Array[])uvw_raw[i];
                for(int j =0; j < time_samples; j++)
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
            var vis_real = new double[baselines, time_samples, channels];
            var vis_imag = new double[baselines, time_samples, channels];
            long visibilitiesCount = 0;
            for (int i = 0; i < baselines; i++)
            {
                Array[] bl = (Array[])vis_raw[i];
                for(int j = 0; j < time_samples; j++)
                {
                    Array[] times = (Array[])bl[j];
                    for(int k = 0; k < channels; k++)
                    {
                        Array[] channel = (Array[])times[k];
                        double[] pol_XX = (double[])channel[0];
                        double[] pol_YY = (double[])channel[3];

                        //add polarizations XX and YY together to form Intensity Visibilities only
                        vis_real[i, j, k] = (pol_XX[0] + pol_YY[0]) / 2.0;
                        vis_imag[i, j, k] = (pol_XX[1] + pol_YY[1]) / 2.0;
                        visibilitiesCount++;
                    }
                }
            }

            //other input parameters:
            int gridSize = 512;
            int subgridsize = 32;
            int kernelSize = 16;

            //TODO: change Cacl_L to use cellsize instead and probably remove the [+0.5pixel] constant
            //cell = image / grid
            int properImageSize = 512;
            double properCellSize = 0.5 / 3600.0 * PI /180.0;
            double imagesize2 = properImageSize * properCellSize;

            int nr_timeslots = 1;
            int max_nr_timesteps = 256;
            float imagesize = 0.0025f;
            //float cellSize = imagesize / gridSize;
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)properCellSize, 1, 0.0f);
            var gridSpheroidal = Math.CalcIdentitySpheroidal(gridSize, gridSize);
            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);

            var subgrids = Partitioner.CreatePartition(p, uvw, frequencies);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequencies, subgridSpheroidal);
            var ftgridded = SubgridFFT.ForwardHack(p, gridded);
            var grid = Adder.AddHack(p, subgrids, ftgridded);
            SubgridFFT.Shift(grid);
            var img = SubgridFFT.ForwardiFFT(grid, visibilitiesCount);
            SubgridFFT.Shift(img);

            //remove spheroidal from grid
            for (int i = 0; i < img.GetLength(0); i++ )
                for(int j = 0; j < img.GetLength(1); j++)
                    img[i, j] = img[i,j] / gridSpheroidal[i,j];

            Write(img, "my_dirty.fits");
        }


        public static void SingleSubgrid()
        {
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 48;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

            double v = -50;
            double wavelength = -4 / imageSize / v;
            double u = 4 / imageSize / wavelength;
            double freq = wavelength * Math.SPEED_OF_LIGHT;
            double[] frequency = { freq };
            double u1 = 10 / imageSize / wavelength;

            double visR0 = 3.9;
            double visR1 = 5.2;

            var vis_real = new double[1, 2, 1];
            var vis_imag = new double[1, 2, 1];
            vis_real[0, 0, 0] = visR0;
            vis_imag[0, 0, 0] = 0;
            vis_real[0, 1, 0] = visR1;
            vis_imag[0, 1, 0] = 0;
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
            var fourierFT = SubgridFFT.ForwardFFT2(ift1);
            Write(fourierFT, "iftOutput.fits");
            WriteImag(fourierFT);
            

             var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);


            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequency, subgridSpheroidal);
            
            var ftgridded = SubgridFFT.ForwardHack(p, gridded);

            var img2 = ftgridded[0][0];
            Write(img2);
            WriteImag(img2);

            var grid = Adder.AddHack(p, subgrids, ftgridded);
            
            SubgridFFT.Shift(grid);
            Write(grid);
            WriteImag(grid);
            var img = SubgridFFT.ForwardiFFT(grid);
            SubgridFFT.Shift(img);
            Write(img);
        }


        public static void SingleSubgrid_orig()
        {
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 16;
            int kernelSize = 2;
            float imageSize = 0.0025f;
            float cellSize = imageSize / gridSize;
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);

            double v = -50;
            double wavelength = -4 / imageSize / v;
            double u = 4 / imageSize / wavelength;
            double freq = wavelength * Math.SPEED_OF_LIGHT;
            double[] frequency = { freq, freq };

            double visR = 3.9;
            double visI = 0.0;

            var vis_real = new double[1, 1, 2];
            var vis_imag = new double[1, 1, 2];
            vis_real[0, 0, 0] = visR;
            vis_imag[0, 0, 0] = visI;
            vis_real[0, 0, 1] = 0;
            vis_imag[0, 0, 1] = 0;
            var uvw = new double[1, 1, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = 0;

            //var imgIFT = IFT(new Complex(visR, visI), u, v, freq, gridSize, imageSize);
            //Write(imgIFT);

            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);


            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequency, subgridSpheroidal);
            //var img2 = gridded[0][0];
            //Write(img2);
            var ftgridded = SubgridFFT.ForwardHack(p, gridded);

            var grid = Adder.AddHack(p, subgrids, ftgridded);
            Write(grid);
            SubgridFFT.Shift(grid);
            var img = SubgridFFT.ForwardiFFT(grid);
            SubgridFFT.Shift(img);
            Write(img);
        }


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
            double v = (-1)*-35.529046011622995;
            double w = 0;

            int nr_timeslots = 1;
            int max_nr_timesteps = 256;
            int gridSize = 64;
            int subgridsize = 64;
            int kernelSize = 16;
            float properCellSize = (float)(2.0 / 3600.0 * PI / 180.0);
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, properCellSize, 1, 0.0f);

            var vis_real = new double[1, 1, 2];
            var vis_imag = new double[1, 1, 2];
            vis_real[0, 0, 0] = visR;
            vis_imag[0, 0, 0] = visI;
            vis_real[0, 0, 1] = 0;
            vis_imag[0, 0, 1] = 0;
            var uvw = new double[1, 1, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = w;

            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);

            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequency, subgridSpheroidal);
            var imgg = gridded[0][0];
            var ftgridded = SubgridFFT.ForwardHack(p, gridded);
            var grid = Adder.AddHack(p, subgrids, ftgridded);
            //Write(grid);
            SubgridFFT.Shift(grid);
            var img = SubgridFFT.ForwardiFFT(grid);
            SubgridFFT.Shift(img);
            Write(img);

        }

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


        public static Complex[,] IFT(Complex vis, double u, double v, double freq, int gridSize, double imageSize)
        {
            u = u * freq / Math.SPEED_OF_LIGHT;
            v = v * freq / Math.SPEED_OF_LIGHT;

            var output = new Complex[gridSize, gridSize];
            var I = new Complex(0, 1);
            var cell = imageSize / gridSize;
            for (int y = 0; y < gridSize; y++)
            {
                for(int x = 0; x < gridSize; x++)
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

        public static void Add(Complex[,] c0, Complex[,] c1)
        {
            for(int i = 0; i < c0.GetLength(0); i++)
            {
                for (int j = 0; j < c0.GetLength(1); j++)
                    c0[i, j] += c1[i, j];
            }
        }
    }
}
