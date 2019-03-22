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
    using NFFT;
    class Program
    {
        static void Main(string[] args)
        {
            ImagePhasor();
            Image();
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
                    uvw[i, j, 1] = values[1]; //v
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
                        vis_real[i, j, k] = pol_XX[0] + pol_YY[0];
                        vis_imag[i, j, k] = pol_XX[1] + pol_YY[1];
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
            float cellSize = imagesize / gridSize;
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);
            var gridSpheroidal = Math.CalcIdentitySpheroidal(gridSize, gridSize);
            var subgridSpheroidal = Math.CalcIdentitySpheroidal(subgridsize, subgridsize);

            var subgrids = Plan.CreatePlan(p, uvw, frequencies);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequencies, subgridSpheroidal);
            var ftgridded = SubgridFFT.ForwardHack(p, gridded);
            var grid = Adder.AddHack(p, subgrids, ftgridded);
            SubgridFFT.Shift(grid);
            
            var img = SubgridFFT.ForwardFFT(grid);
            //SubgridFFT.Shift(grid);

            //remove spheroidal from grid
            for (int i = 0; i < img.GetLength(0); i++ )
                for(int j = 0; j < img.GetLength(1); j++)
                    img[i, j] = img[i,j] / gridSpheroidal[i,j];

            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = grid[i, j].Real;// img[i, j];
                }
            }
                
            //write image
            f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream("Outputfile.fits", FileMode.Create)))
            {
                f.Write(fstream);
            }
        }

        private static void Image()
        {
            double[] frequency = { 857000000f };

            double visR = 3.214400053024292;
            double visI = 0.801982581615448;

            double u = 0.17073525427986169 * frequency[0] / Math.SPEED_OF_LIGHT;
            double v = -399.17929423647 * frequency[0] / Math.SPEED_OF_LIGHT;
            double w = -2.7543493956327438 * frequency[0] / Math.SPEED_OF_LIGHT;

            var vis = new Complex(visR, visI);
            var I = new Complex(0, 1);

            double yStep = 1.0 / 8.0;
            double xStep = 1.0 / 16.0;
            double cell = 8.0 / 3600 * PI / 180;

            for (int y = 0; y < 4; y++)
            {
                Complex[] row = new Complex[8];
                for(int x = 0; x< 8;x++ )
                {

                    var c = vis * Complex.Exp(2 * PI * I * (u* (x) * cell + v * (y) * cell));
                    row[x] = c;
                }
            }
        }

        private static void ImagePhasor()
        {
            double[] frequency = { 857000000f };
            
            double visR = 3.214400053024292;
            double visI = 0.801982581615448;
            
            double u = 0.17073525427986169;
            double v = -399.17929423647;
            double w = -2.7543493956327438;

            var wavenumbers = Math.FrequencyToWavenumber(frequency);

            var vis = new Complex(visR, visI);

            int subgridsize = 16;
            double cellSize = 8.0 / 3600 * PI / 180;
            double imagesize = subgridsize * cellSize;


           
            Complex[,] img = new Complex[16,16];
            for(int y = 0; y < 16; y++)
            {
                for (int x = 0; x < 16; x++)
                {
                    var l = ComputeL(x, subgridsize, imagesize);
                    var m = ComputeL(y, subgridsize, imagesize);

                    double phaseIndex = u * l + v * y;
                    double phase = (phaseIndex * wavenumbers[0]);
                    var tf = new Complex(Cos(phase), Sin(phase));
                    var c = vis * tf;
                    img[y, x] = c;
                }
            }

            


        }


        private static double ComputeL(int x, int subgridSize, double imageSize)
        {
            return (x - (subgridSize / 2)) * imageSize / subgridSize;
        }



    }
}
