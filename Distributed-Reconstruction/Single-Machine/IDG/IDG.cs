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
        public static double[,] ToImage(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.subgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, visibilitiesCount);
            FFT.Shift(img);

            return img;
        }

        public static Complex[,,] ToVisibilities(GriddingConstants c, List<List<SubgridHack>> metadata, double[,] image)
        {
            FFT.Shift(image);
            var grid = FFT.GridFFT(image);
            FFT.Shift(grid);
            var gridded = Adder.SplitHack(c, metadata, grid);
            //FFT
            //splitter
            //subgridFFT

            //De-Gridder
            return null;
        }
    }
}
