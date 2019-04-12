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
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\uvw.fits");
            var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\vis.fits",uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);

            BinaryIO.WriteFrequency(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\binary\simulation_point\freq.binary", frequencies);
            BinaryIO.WriteUVW(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\binary\simulation_point\uvw.binary", uvw);
            BinaryIO.WriteVisibilities(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\binary\simulation_point\vis.binary", visibilities);
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
