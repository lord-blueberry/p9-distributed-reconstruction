using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Single_Machine.IDG
{
    class IDG
    {
        public static double[,] ToImage(GriddingConstants constants, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            var gridded = Gridder.ForwardHack(constants, metadata, uvw, visibilities, frequencies, constants.subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(constants, gridded);
            var grid = Adder.AddHack(constants, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, visibilitiesCount);
            FFT.Shift(img);

            return img;
        }

        public static Complex[,,] ToVisibilities(GriddingConstants constants, List<List<SubgridHack>> metadata, double[,] image)
        {
            FFT.Shift(image);
            var grid = FFT.GridFFT(image);
            FFT.Shift(grid);
            var gridded = Adder.SplitHack(constants, metadata, grid);
            //FFT
            //splitter
            //subgridFFT

            //De-Gridder
            return null;
        }
    }
}
