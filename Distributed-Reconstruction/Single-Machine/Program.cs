using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using FFTW.NET;
using nom.tam.fits;
using System.IO;

namespace Single_Machine
{
    using NFFT;
    class Program
    {
        static void Main(string[] args)
        {
            /*
            using (var input = new AlignedArrayComplex(16, 64, 24))
            using (var output = new AlignedArrayComplex(16, input.GetSize()))
            {
                for (int row = 0; row < input.GetLength(0); row++)
                {
                    for (int col = 0; col < input.GetLength(1); col++)
                        input[row, col] = (double)row * col / input.Length;
                }

                DFT.FFT(input, output);
                DFT.IFFT(output, output);

                for (int row = 0; row < input.GetLength(0); row++)
                {
                    for (int col = 0; col < input.GetLength(1); col++)
                        Console.Write((output[row, col].Real / input[row, col].Real / input.Length).ToString("F2").PadLeft(6));
                    Console.WriteLine();
                }
            }
            */

            Fits f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\freq.fits");
            ImageHDU h = (ImageHDU)f.ReadHDU();
            var frequencies = (double[])h.Kernel;
            f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\uvw.fits");
            h = (ImageHDU)f.ReadHDU();

            // Double Cube Dimensions: baseline, time, uvw
            var uvw_raw = (Array[])h.Kernel;
            var baselines = uvw_raw.Length;
            var time_samples = uvw_raw[0].Length;

            //TODO: completely wrong order. I thought C#/C++ was column major, but it is row major
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

            int nr_timeslots = 1;
            int max_nr_timesteps = 256; //
            float cellSize = 1.0f;
            var p = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);
            var gridSpheroidal = Math.CalcIdentitySpheroidal(gridSize, gridSize);
            var subgridSpheroidal = Math.CalcExampleSpheroidal(subgridsize, subgridsize);


            var subgrids = Plan.CreatePlan(p, uvw, frequencies);
            Gridder.ForwardHack(p, subgrids, uvw, vis_real, vis_imag, frequencies, subgridSpheroidal);


        }



    }
}
