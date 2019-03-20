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
            var uvw = (System.Array)h.Kernel;
            f = new Fits(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\vis.fits");
            h = (ImageHDU)f.ReadHDU();
            //Double cube Dimensions: baseline, time, channel, pol, complex_component
            var vis = (System.Array)h.Kernel;

            //other input parameters:
            int gridSize = 512;
            int subgridsize = 32;
            int kernelSize = 16;

            int nr_timeslots = 1;
            int max_nr_timesteps = 256;
            float cellSize = 0.5f;

            var parameters = new GriddingParams(gridSize, subgridsize, kernelSize, max_nr_timesteps, cellSize, 1, 0.0f);
        }



    }
}
