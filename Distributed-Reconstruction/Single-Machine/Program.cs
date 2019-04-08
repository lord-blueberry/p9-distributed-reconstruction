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
            Debug.DebugForwardBackward();
            Debug.DebugFullPipeline();

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
            var visibilities = new Complex[baselines, time_samples, channels];
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
                        visibilities[i, j, k] = new Complex(
                            (pol_XX[0] + pol_YY[0]) / 2.0,
                            (pol_XX[1] + pol_YY[1]) / 2.0
                            );
                        visibilitiesCount++;
                    }
                }
            }

            //other input parameters:
            int gridSize = 512;
            int subgridsize = 32;
            int kernelSize = 16;

            //cell = image / grid
            int properImageSize = 512;
            double properCellSize = 0.5 / 3600.0 * PI /180.0;
            double imagesize2 = properImageSize * properCellSize;

            int nr_timeslots = 1;
            int max_nr_timesteps = 256;
            float imagesize = 0.0025f;
            //float cellSize = imagesize / gridSize;
            var p = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)properCellSize, 1, 0.0f);
            var gridSpheroidal = Math.CalcIdentitySpheroidal(gridSize, gridSize);
            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);

            var subgrids = Partitioner.CreatePartition(p, uvw, frequencies);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, visibilities, frequencies, subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(p, gridded);
            var grid = Adder.AddHack(p, subgrids, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, visibilitiesCount);
            FFT.Shift(img);

            //remove spheroidal from grid
            for (int i = 0; i < img.GetLength(0); i++ )
                for(int j = 0; j < img.GetLength(1); j++)
                    img[i, j] = img[i,j] / gridSpheroidal[i,j];

            Debug.Write(img, "my_dirty.fits");
        }

        public static void SingleSubgrid_orig()
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
            double freq = wavelength * Math.SPEED_OF_LIGHT;
            double[] frequency = { freq, freq };

            double visR = 3.9;
            double visI = 0.0;

            var visibilities = new Complex[1, 1, 2];
            visibilities[0, 0, 0] = new Complex(visR, visI);
            visibilities[0, 0, 1] = 0;
            var uvw = new double[1, 1, 3];
            uvw[0, 0, 0] = u;
            uvw[0, 0, 1] = v;
            uvw[0, 0, 2] = 0;

            //var imgIFT = IFT(new Complex(visR, visI), u, v, freq, gridSize, imageSize);
            //Write(imgIFT);

            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);


            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, visibilities, frequency, subgridSpheroidal);
            //var img2 = gridded[0][0];
            //Write(img2);
            var ftgridded = FFT.SubgridFFT(p, gridded);

            var grid = Adder.AddHack(p, subgrids, ftgridded);
            Debug.Write(grid);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);
            Debug.Write(img);
        }












    }
}
