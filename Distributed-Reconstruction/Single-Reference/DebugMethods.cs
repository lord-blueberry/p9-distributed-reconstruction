using System;
using System.Collections.Generic;
using System.Text;
using Single_Machine.IDG;
using System.Numerics;

namespace Single_Reference
{
    class DebugMethods
    {
        #region IDG test
        public static void DebugForwardBackward()
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
            double freq = wavelength * MathFunctions.SPEED_OF_LIGHT;
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

            var subgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridsize, subgridsize);
            var metadata = Partitioner.CreatePartition(p, uvw, frequency);

            var gridded_subgrids = Gridder.ForwardHack(p, metadata, uvw, visibilities, frequency, subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(p, gridded_subgrids);
            var grid = Adder.AddHack(p, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, 2);
            FFT.Shift(img);

            FFT.Shift(img);
            var grid2 = FFT.GridFFT(img, 2);
            FFT.Shift(grid2);
            var ftGridded2 = Adder.SplitHack(p, metadata, grid2);
            var subgrids2 = FFT.SubgridIFFT(p, ftGridded2);
            var visibilities2 = Gridder.BackwardsHack(p, metadata, subgrids2, uvw, frequency, subgridSpheroidal);
        }

        #endregion
    }
}
