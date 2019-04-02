using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Single_Machine.IDG;
using System.Numerics;
using static System.Math;

namespace Single_Machine
{
    class Debug
    {
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
            Program.Write(img);
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
            Program.Write(ift1, "iftOutput.fits");
            var fourierFT = FFT.GridFFT(ift1);
            Program.Write(fourierFT, "iftOutput.fits");
            Program.WriteImag(fourierFT);


            var subgridSpheroidal = IDG.Math.CalcIdentitySpheroidal(subgridsize, subgridsize);


            var subgrids = Partitioner.CreatePartition(p, uvw, frequency);
            var gridded = Gridder.ForwardHack(p, subgrids, uvw, visibilities, frequency, subgridSpheroidal);

            var ftgridded = FFT.SubgridFFT(p, gridded);

            var img2 = ftgridded[0][0];
            Program.Write(img2);
            Program.WriteImag(img2);

            var grid = Adder.AddHack(p, subgrids, ftgridded);

            FFT.Shift(grid);
            Program.Write(grid);
            Program.WriteImag(grid);
            var img = FFT.GridIFFT(grid);
            FFT.Shift(img);
            Program.Write(img);
        }

        #region helpers
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
    }
}
